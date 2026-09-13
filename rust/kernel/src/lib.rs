//! `rapidmesh_kernel` — the Rust collapse sweep behind RapidMesh's band
//! interface (WP-3.4b, Round 11).
//!
//! # The boundary
//!
//! `decimate.decimate_patch` is the interface. Above it nothing changes:
//! `decimate_tiles.decimate_generation` still drives one tile at a time and
//! still sees a `DecimatedPatch`. Below it, the scalar collapse sweep either
//! runs here or in `_PatchState`, and the two are required to agree bit for
//! bit — `decimate.py` is the equivalence reference and stays in the tree
//! (DEC-012's pattern, applied to the decimator).
//!
//! # Why everything crosses as `bytes`
//!
//! Arrays arrive as `bytes` and leave as `bytes` rather than through the buffer
//! protocol. That looks wasteful and is: it copies a few megabytes per tile,
//! against a tile that takes tens of seconds. What it buys is that there is no
//! buffer-format negotiation to get wrong — a NumPy `bool` array exports `?`,
//! which some buffer consumers reject, and a non-contiguous or byte-swapped
//! view would be a silent wrong answer rather than a loud one. The copy is
//! under 0.1% of the run and it removes a class of bug that would otherwise be
//! discovered as a bad measurement.
//!
//! Nothing here allocates anything station-sized: one tile is resident, which
//! is the DEC-021 window and is what `CLAUDE.md` §11 requires.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod quadric;
mod sweep;

use sweep::{PatchState, Settings};

/// Interpret a `bytes` blob as `f64`s in the platform's own order, which is
/// what NumPy's `tobytes()` writes for a native dtype.
fn decode_f64(raw: &[u8], expected: usize, what: &str) -> PyResult<Vec<f64>> {
    if raw.len() != expected * 8 {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {} bytes, got {}",
            expected * 8,
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().expect("chunks_exact(8) is 8 bytes")))
        .collect())
}

fn decode_i64(raw: &[u8], expected: usize, what: &str) -> PyResult<Vec<i64>> {
    if raw.len() != expected * 8 {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {} bytes, got {}",
            expected * 8,
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|c| i64::from_ne_bytes(c.try_into().expect("chunks_exact(8) is 8 bytes")))
        .collect())
}

fn encode_f64(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

fn encode_i64_from_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&(*v as i64).to_ne_bytes());
    }
    out
}

/// `array('i')` on the Python side, which is what `_PatchState.parent` holds.
fn encode_i32_from_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&(*v as i32).to_ne_bytes());
    }
    out
}

fn encode_bools(values: &[bool]) -> Vec<u8> {
    values.iter().map(|&b| u8::from(b)).collect()
}

/// Run one collapse sweep over one patch.
///
/// Takes the state `_PatchState.__init__` would have built — including the
/// per-vertex quadrics and the unique-edge list, which stay in NumPy so that
/// `np.bincount`'s accumulation order is never reproduced by hand — and returns
/// the state `_PatchState.finish()` reads. The caller re-uses the existing
/// `finish()` unchanged, so output compaction is the same NumPy code on both
/// paths and cannot be a source of disagreement.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    pos_x, pos_y, pos_z, quad, amin, locked, triangle_vertices, edges,
    vertex_count, triangle_count, target_triangles, error_limit, deviation_limit,
    cos_limit, placement_optimal,
))]
fn decimate_sweep<'py>(
    py: Python<'py>,
    pos_x: &[u8],
    pos_y: &[u8],
    pos_z: &[u8],
    quad: &[u8],
    amin: &[u8],
    locked: &[u8],
    triangle_vertices: &[u8],
    edges: &[u8],
    vertex_count: usize,
    triangle_count: usize,
    target_triangles: i64,
    error_limit: f64,
    deviation_limit: f64,
    cos_limit: f64,
    placement_optimal: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let px = decode_f64(pos_x, vertex_count, "pos_x")?;
    let py_positions = decode_f64(pos_y, vertex_count, "pos_y")?;
    let pz = decode_f64(pos_z, vertex_count, "pos_z")?;
    let quadrics = decode_f64(quad, vertex_count * 10, "quad")?;
    let min_areas = decode_f64(amin, vertex_count, "amin")?;
    if locked.len() != vertex_count {
        return Err(PyValueError::new_err(format!(
            "locked: expected {vertex_count} bytes, got {}",
            locked.len()
        )));
    }
    let lock: Vec<bool> = locked.iter().map(|&b| b != 0).collect();
    let tv_i64 = decode_i64(triangle_vertices, triangle_count * 3, "triangle_vertices")?;
    if edges.len() % 16 != 0 {
        return Err(PyValueError::new_err("edges: expected pairs of int64"));
    }
    let edge_pairs = decode_i64(edges, edges.len() / 8, "edges")?;

    if vertex_count > u32::MAX as usize || triangle_count > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "patch exceeds the kernel's 2^32 index range",
        ));
    }
    let mut tv = Vec::with_capacity(tv_i64.len());
    for &index in &tv_i64 {
        if index < 0 || index as usize >= vertex_count {
            return Err(PyValueError::new_err(
                "a triangle names a vertex outside the patch",
            ));
        }
        tv.push(index as u32);
    }
    for &index in &edge_pairs {
        if index < 0 || index as usize >= vertex_count {
            return Err(PyValueError::new_err("an edge names a vertex outside the patch"));
        }
    }

    let settings = Settings {
        target_triangles: if target_triangles < 0 {
            None
        } else {
            Some(target_triangles)
        },
        error_limit,
        deviation_limit,
        cos_limit,
        placement_optimal,
    };

    // The interpreter is detached for the sweep itself (`allow_threads` under
    // its pre-0.25 name). Nothing inside touches a Python object — every input
    // was copied into Rust above — so this is sound, and it is what lets a
    // caller overlap a sweep with I/O later without this kernel having to know
    // about threads. The kernel is **not** itself threaded: determinism under
    // threading would be a separate claim needing separate evidence, and the
    // bar does not need the speed.
    let state = py.detach(move || {
        let mut state = PatchState::new(
            px,
            py_positions,
            pz,
            quadrics,
            min_areas,
            lock,
            tv,
            &edge_pairs,
            settings,
        );
        state.run();
        state
    });

    let (out_x, out_y, out_z) = state.positions();
    let result = PyDict::new(py);
    result.set_item("pos_x", PyBytes::new(py, &encode_f64(out_x)))?;
    result.set_item("pos_y", PyBytes::new(py, &encode_f64(out_y)))?;
    result.set_item("pos_z", PyBytes::new(py, &encode_f64(out_z)))?;
    result.set_item(
        "triangle_vertices",
        PyBytes::new(py, &encode_i64_from_u32(state.triangle_vertices())),
    )?;
    result.set_item("alive", PyBytes::new(py, &encode_bools(state.alive_flags())))?;
    result.set_item("moved", PyBytes::new(py, &encode_bools(state.moved_flags())))?;
    result.set_item(
        "triangle_alive",
        PyBytes::new(py, &encode_bools(state.triangle_alive_flags())),
    )?;
    result.set_item("collapses", state.collapses)?;
    result.set_item("rejected_link", state.rejected_link)?;
    result.set_item("rejected_seam", state.rejected_seam)?;
    result.set_item("rejected_turn", state.rejected_turn)?;
    result.set_item("rejected_error", state.rejected_error)?;
    result.set_item("rejected_deviation", state.rejected_deviation)?;
    result.set_item("bound", PyBytes::new(py, &encode_f64(state.bounds())))?;
    result.set_item("parent", PyBytes::new(py, &encode_i32_from_u32(state.parents())))?;
    result.set_item("max_accepted", state.max_accepted)?;
    result.set_item("live_triangles", state.live_triangle_count())?;
    Ok(result)
}

/// The version of the kernel's own contract with `decimate.py`.
///
/// Bumped when the argument list or the returned keys change, so a stale `.pyd`
/// left on a path is refused by the Python side rather than silently producing
/// a different answer. It is not the crate version.
#[pyfunction]
fn contract_version() -> u32 {
    2
}

#[pymodule]
fn rapidmesh_kernel(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decimate_sweep, module)?)?;
    module.add_function(wrap_pyfunction!(contract_version, module)?)?;
    module.add("__doc__", "RapidMesh quadric collapse sweep (WP-3.4b).")?;
    Ok(())
}
