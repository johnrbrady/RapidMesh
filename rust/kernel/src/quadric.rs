//! The quadric algebra, transcribed from `decimate_quadrics.py`.
//!
//! Every expression here is a **literal transcription** of the Python one,
//! including the order of the additions and multiplications. That is not
//! stylistic: float addition is not associative, and bit-identity with the
//! Python reference (`decimate_patch`, which stays in the tree as the
//! equivalence reference) is only available if the operand order matches.
//! Rust's `+` and `*` are left-associative exactly as Python's are, and Rust
//! does not contract `a * b + c` into an FMA without an explicit `mul_add`, so
//! a transcription that looks the same *is* the same.
//!
//! Do not "simplify" any expression in this file. A Horner form, a common
//! subexpression lifted out, or a `mul_add` would each be a silent change to
//! the result in the last bits.

/// Relative determinant floor for the 3x3 optimal-placement solve.
/// `decimate_quadrics.SINGULAR_RELATIVE`.
pub const SINGULAR_RELATIVE: f64 = 1e-10;

/// `v^T Q v` for `v = (x, y, z, 1)`, written out.
///
/// Transcribed from `decimate_quadrics.quadric_error`.
#[inline]
pub fn quadric_error(q: &[f64; 10], x: f64, y: f64, z: f64) -> f64 {
    let (a, b, c, d, e, f, g, h, i, j) = (
        q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9],
    );
    a * x * x + 2.0 * b * x * y + 2.0 * c * x * z + 2.0 * d * x
        + e * y * y + 2.0 * f * y * z + 2.0 * g * y
        + h * z * z + 2.0 * i * z
        + j
}

/// The point where the quadric is least, or `None` if it has no unique one.
///
/// Transcribed from `decimate_quadrics.solve_optimal`.
#[inline]
pub fn solve_optimal(q: &[f64; 10]) -> Option<(f64, f64, f64)> {
    let (a, b, c, d, e, f, g, h, i) = (
        q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8],
    );
    let c00 = e * h - f * f;
    let c01 = c * f - b * h;
    let c02 = b * f - c * e;
    let det = a * c00 + b * c01 + c * c02;
    let scale = a + e + h;
    if scale <= 0.0 || det.abs() <= SINGULAR_RELATIVE * scale * scale * scale {
        return None;
    }
    let c11 = a * h - c * c;
    let c12 = b * c - a * f;
    let c22 = a * e - b * b;
    let inv = 1.0 / det;
    Some((
        -inv * (c00 * d + c01 * g + c02 * i),
        -inv * (c01 * d + c11 * g + c12 * i),
        -inv * (c02 * d + c12 * g + c22 * i),
    ))
}

/// Unit triangle normal, or `None` when the triangle has no area.
///
/// Transcribed from `decimate._normal`. `sqrt` is the hardware instruction in
/// both languages and is correctly rounded by IEEE-754, so it needs no special
/// handling to agree.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn normal(
    ax: f64, ay: f64, az: f64,
    bx: f64, by: f64, bz: f64,
    cx: f64, cy: f64, cz: f64,
) -> Option<(f64, f64, f64)> {
    let (ux, uy, uz) = (bx - ax, by - ay, bz - az);
    let (vx, vy, vz) = (cx - ax, cy - ay, cz - az);
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let length = (nx * nx + ny * ny + nz * nz).sqrt();
    if length <= 0.0 {
        return None;
    }
    Some((nx / length, ny / length, nz / length))
}
