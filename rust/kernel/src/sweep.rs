//! The collapse sweep, transcribed from `_PatchState` in `decimate.py`.
//!
//! # What this is and is not
//!
//! This is a **transcription**, not a reimplementation. `decimate.py` stays in
//! the tree as the equivalence reference (the same relation
//! `pipeline.mesh_station` has to the streamed path, DEC-012), and the claim
//! this kernel has to support is bit-identity with it. So every predicate is
//! evaluated in the same order, every float expression has the same operand
//! order, and every tie is broken the same way. Where a faster formulation
//! exists it is not taken, and the comment says why.
//!
//! # Why bit-identity is available at all
//!
//! Python's version iterates several `set`s, and set iteration order is not
//! something Rust could reproduce even if it wanted to. It does not have to:
//! **no set iteration in the Python sweep can affect the output.**
//!
//! * `shared` and the vertex stars are iterated to mark triangles dead, to
//!   rewrite indices and to test membership — all order-independent.
//! * `_turn_ok` and `_would_join_locked` reduce their iteration to one boolean
//!   with `any`/early return; which element triggers it varies, the answer does
//!   not.
//! * `_commit` pushes one queue entry per neighbour of the survivor. Push order
//!   would matter to a heap whose pop order depended on insertion, but the
//!   queue's key is the 4-tuple `(cost, u, v, versions)` and those tuples are
//!   **distinct**: within one commit the neighbours are distinct, and across
//!   commits the survivor's `version` has moved. A binary heap over a strict
//!   total order pops the unique minimum, so the pop sequence is fixed by the
//!   set of live entries and not by how they arrived.
//!
//! That argument is why this file can use `Vec` stars and a `BinaryHeap` and
//! still be expected to agree to the last bit. It is an argument, so it is
//! *checked* rather than trusted: the equivalence tests compare Rust and Python
//! output byte for byte on fixtures and on real tiles.
//!
//! # What deliberately stays in NumPy
//!
//! The per-vertex quadrics and the unique-edge list are **not** built here.
//! They are vectorised NumPy in `decimate_quadrics.vertex_quadrics` and
//! `decimate._unique_edges`, they are together about 6% of the run, and
//! reproducing `np.bincount`'s accumulation order bit-for-bit is the one part
//! of this port with real float-identity risk. They are computed once in Python
//! and handed in. This trades ~6% of an achievable 14x for the identity claim,
//! which is the right way round: the bar needs 1.15x.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::quadric::{normal, quadric_error, solve_optimal};

/// `decimate._HEAP_SLACK`.
pub const HEAP_SLACK: f64 = 3.0;

/// Everything that can move the output. `decimate.DecimationSettings`, reduced
/// to what the sweep actually reads: the caller has already turned degrees into
/// a cosine and metres into a squared limit, so no transcendental is evaluated
/// on this side of the boundary and the two languages' libm cannot disagree.
#[derive(Clone, Copy)]
pub struct Settings {
    pub target_triangles: Option<i64>,
    /// `max_error_m ** 2`, or `f64::INFINITY`.
    pub error_limit: f64,
    /// `cos(radians(max_normal_turn_deg))`, computed by the caller.
    pub cos_limit: f64,
    /// `placement == "optimal"`.
    pub placement_optimal: bool,
}

/// One queue entry. `Ord` is **reversed** so that `BinaryHeap`, a max-heap,
/// pops the same element `heapq` would: the minimum of the 4-tuple.
#[derive(Clone, Copy)]
struct Entry {
    cost: f64,
    u: u32,
    v: u32,
    versions: u64,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp` rather than `partial_cmp().unwrap()`: costs are finite and
        // non-negative here, but a total order that cannot panic is worth more
        // than an assertion that says so.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.u.cmp(&self.u))
            .then_with(|| other.v.cmp(&self.v))
            .then_with(|| other.versions.cmp(&self.versions))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A generation-stamped membership set over a fixed index range.
///
/// Replaces Python's transient `set`s without allocating one per candidate
/// edge. `bump` invalidates every mark in O(1); the counter is `u32` and is
/// re-zeroed on wrap, so a stale stamp cannot alias a live one.
struct Marks {
    stamp: Vec<u32>,
    generation: u32,
}

impl Marks {
    fn new(size: usize) -> Self {
        Self { stamp: vec![0; size], generation: 0 }
    }

    fn bump(&mut self) {
        if self.generation == u32::MAX {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.generation = 0;
        }
        self.generation += 1;
    }

    #[inline]
    fn set(&mut self, index: usize) {
        self.stamp[index] = self.generation;
    }

    #[inline]
    fn has(&self, index: usize) -> bool {
        self.stamp[index] == self.generation
    }
}

pub struct PatchState {
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    quad: Vec<f64>,
    locked: Vec<bool>,
    moved: Vec<bool>,
    alive: Vec<bool>,
    version: Vec<u32>,
    tv: Vec<u32>,
    tri_alive: Vec<bool>,
    live_triangles: i64,
    vtris: Vec<Vec<u32>>,
    heap: BinaryHeap<Entry>,
    heap_limit: usize,
    settings: Settings,

    pub collapses: u64,
    pub rejected_link: u64,
    pub rejected_seam: u64,
    pub rejected_turn: u64,
    pub rejected_error: u64,
    pub max_accepted: f64,

    mark_u: Marks,
    mark_v: Marks,
    mark_tri: Marks,
    nu: Vec<u32>,
    nv: Vec<u32>,
    shared: Vec<u32>,
    star: Vec<u32>,
}

impl PatchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        px: Vec<f64>,
        py: Vec<f64>,
        pz: Vec<f64>,
        quad: Vec<f64>,
        locked: Vec<bool>,
        tv: Vec<u32>,
        edges: &[i64],
        settings: Settings,
    ) -> Self {
        let count = px.len();
        let faces = tv.len() / 3;

        let mut vtris: Vec<Vec<u32>> = vec![Vec::new(); count];
        for t in 0..faces {
            let base = 3 * t;
            vtris[tv[base] as usize].push(t as u32);
            vtris[tv[base + 1] as usize].push(t as u32);
            vtris[tv[base + 2] as usize].push(t as u32);
        }

        let mut state = Self {
            px,
            py,
            pz,
            quad,
            moved: vec![false; count],
            alive: vec![true; count],
            version: vec![0; count],
            locked,
            tv,
            tri_alive: vec![true; faces],
            live_triangles: faces as i64,
            vtris,
            heap: BinaryHeap::new(),
            heap_limit: 0,
            settings,
            collapses: 0,
            rejected_link: 0,
            rejected_seam: 0,
            rejected_turn: 0,
            rejected_error: 0,
            max_accepted: 0.0,
            mark_u: Marks::new(count),
            mark_v: Marks::new(count),
            mark_tri: Marks::new(faces),
            nu: Vec::new(),
            nv: Vec::new(),
            shared: Vec::new(),
            star: Vec::new(),
        };
        state.heap_limit = state.heap_limit_now();

        for pair in edges.chunks_exact(2) {
            state.push(pair[0] as usize, pair[1] as usize);
        }
        state
    }

    // -- the sweep ----------------------------------------------------------

    /// `_PatchState.run`.
    ///
    /// Python checks `while heap` first and the target second; this checks the
    /// target first and treats an empty queue as the other exit. The two agree
    /// on every case: an empty queue ends the sweep either way, and a reached
    /// target ends it either way.
    pub fn run(&mut self) {
        let target = self.settings.target_triangles;
        loop {
            if let Some(t) = target {
                if self.live_triangles <= t {
                    return;
                }
            }
            let entry = match self.heap.pop() {
                Some(entry) => entry,
                None => return,
            };
            let (u, v) = (entry.u as usize, entry.v as usize);
            if !(self.alive[u] && self.alive[v]) {
                continue;
            }
            if entry.versions != self.versions(u, v) {
                continue; // superseded; the live entry is still queued
            }
            self.try_collapse(u, v);
        }
    }

    /// `_PatchState._try_collapse`.
    fn try_collapse(&mut self, u: usize, v: usize) {
        // shared = vtris[u] & vtris[v]
        self.mark_tri.bump();
        for &t in &self.vtris[u] {
            self.mark_tri.stamp[t as usize] = self.mark_tri.generation;
        }
        let mut shared = std::mem::take(&mut self.shared);
        shared.clear();
        for &t in &self.vtris[v] {
            if self.mark_tri.has(t as usize) {
                shared.push(t);
            }
        }
        if !(1..=2).contains(&shared.len()) {
            self.shared = shared;
            self.rejected_link += 1;
            return;
        }

        let mut nu = std::mem::take(&mut self.nu);
        let mut nv = std::mem::take(&mut self.nv);
        self.neighbours_into(u, &mut nu, true);
        self.neighbours_into(v, &mut nv, false);
        // len(nu & nv) != len(shared)
        let mut intersection = 0usize;
        for &w in &nu {
            if self.mark_v.has(w as usize) {
                intersection += 1;
            }
        }
        if intersection != shared.len() {
            self.nu = nu;
            self.nv = nv;
            self.shared = shared;
            self.rejected_link += 1;
            return;
        }

        let (keep, drop) = if !self.locked[v] { (u, v) } else { (v, u) };
        if self.locked[drop] {
            self.nu = nu;
            self.nv = nv;
            self.shared = shared;
            self.rejected_link += 1; // both ends locked: the edge is a seam
            return;
        }
        if self.locked[keep] {
            // `_would_join_locked(keep, keep_nbrs, drop_nbrs)`: is there a
            // vertex in the dropped end's star that is locked, is not the
            // survivor, and is not already the survivor's neighbour?
            let (drop_nbrs, keep_is_u) = if keep == u { (&nv, true) } else { (&nu, false) };
            let joins = drop_nbrs.iter().any(|&w| {
                let w = w as usize;
                w != keep
                    && self.locked[w]
                    && !(if keep_is_u { self.mark_u.has(w) } else { self.mark_v.has(w) })
            });
            if joins {
                self.nu = nu;
                self.nv = nv;
                self.shared = shared;
                self.rejected_seam += 1;
                return;
            }
        }

        let q = self.quadric_sum(u, v);
        let (x, y, z) = self.place(&q, u, v);
        if !self.turn_ok(keep, drop, &shared, x, y, z) {
            self.nu = nu;
            self.nv = nv;
            self.shared = shared;
            self.rejected_turn += 1;
            return;
        }
        let cost = quadric_error(&q, x, y, z);
        self.commit(keep, drop, &shared, x, y, z);
        self.collapses += 1;
        if cost > self.max_accepted {
            self.max_accepted = cost;
        }
        nu.clear();
        nv.clear();
        self.nu = nu;
        self.nv = nv;
        self.shared = shared;
    }

    /// `_PatchState._commit`.
    fn commit(&mut self, keep: usize, drop: usize, shared: &[u32], x: f64, y: f64, z: f64) {
        for &t in shared {
            let base = 3 * t as usize;
            for slot in base..base + 3 {
                let owner = self.tv[slot] as usize;
                if let Some(at) = self.vtris[owner].iter().position(|&e| e == t) {
                    self.vtris[owner].swap_remove(at);
                }
            }
            self.tri_alive[t as usize] = false;
        }
        self.live_triangles -= shared.len() as i64;

        // Every triangle still in the dropped vertex's star names `drop` and
        // not `keep` — one that named both was in `shared` and has just gone —
        // so these can be moved across without a membership test.
        let mut star = std::mem::take(&mut self.star);
        star.clear();
        star.extend_from_slice(&self.vtris[drop]);
        for &t in &star {
            let base = 3 * t as usize;
            for slot in base..base + 3 {
                if self.tv[slot] as usize == drop {
                    self.tv[slot] = keep as u32;
                }
            }
            self.vtris[keep].push(t);
        }
        self.vtris[drop].clear();
        self.star = star;
        self.alive[drop] = false;

        self.px[keep] = x;
        self.py[keep] = y;
        self.pz[keep] = z;
        if !self.locked[keep] {
            self.moved[keep] = true;
        }
        let (base_k, base_d) = (10 * keep, 10 * drop);
        for i in 0..10 {
            self.quad[base_k + i] += self.quad[base_d + i];
        }
        self.version[keep] += 1;

        let mut star = std::mem::take(&mut self.star);
        self.neighbours_into(keep, &mut star, true);
        for i in 0..star.len() {
            self.push(keep, star[i] as usize);
        }
        star.clear();
        self.star = star;

        if self.heap.len() > self.heap_limit {
            self.compact();
        }
    }

    // -- geometry -----------------------------------------------------------

    /// `_PatchState._neighbours`, into a caller-owned buffer.
    ///
    /// `into_u` selects which mark set records membership, because
    /// `_try_collapse` needs both endpoints' neighbour sets live at once.
    fn neighbours_into(&mut self, u: usize, out: &mut Vec<u32>, into_u: bool) {
        out.clear();
        let marks = if into_u { &mut self.mark_u } else { &mut self.mark_v };
        marks.bump();
        let star = std::mem::take(&mut self.vtris[u]);
        for &t in &star {
            let base = 3 * t as usize;
            for slot in base..base + 3 {
                let w = self.tv[slot] as usize;
                if w == u {
                    continue; // `out.discard(u)`
                }
                let marks = if into_u { &mut self.mark_u } else { &mut self.mark_v };
                if !marks.has(w) {
                    marks.set(w);
                    out.push(w as u32);
                }
            }
        }
        self.vtris[u] = star;
        // The invariant the callers depend on: after this returns,
        // `marks.has(x)` is true for exactly the members of `out`. `u` must
        // therefore NOT be marked, even though marking it would be a tidier way
        // to keep it out of `out` — `_try_collapse` counts `len(nu & nv)` by
        // testing `mark_v` over `nu`, and `v` is always a neighbour of `u`, so a
        // self-mark makes that intersection exactly one too large and the link
        // condition rejects nearly every collapse. That was a real defect here,
        // found by the equivalence test rather than by reading.
    }

    /// `_PatchState._quadric_sum`.
    #[inline]
    fn quadric_sum(&self, u: usize, v: usize) -> [f64; 10] {
        let (a, b) = (10 * u, 10 * v);
        let mut q = [0.0f64; 10];
        for i in 0..10 {
            q[i] = self.quad[a + i] + self.quad[b + i];
        }
        q
    }

    /// `_PatchState._place`.
    fn place(&self, q: &[f64; 10], u: usize, v: usize) -> (f64, f64, f64) {
        if self.locked[u] {
            return (self.px[u], self.py[u], self.pz[u]);
        }
        if self.locked[v] {
            return (self.px[v], self.py[v], self.pz[v]);
        }
        let ends = [
            (self.px[u], self.py[u], self.pz[u]),
            (self.px[v], self.py[v], self.pz[v]),
        ];
        if !self.settings.placement_optimal {
            return cheapest(q, &ends);
        }
        if let Some(best) = solve_optimal(q) {
            return best;
        }
        let midpoint = (
            0.5 * (self.px[u] + self.px[v]),
            0.5 * (self.py[u] + self.py[v]),
            0.5 * (self.pz[u] + self.pz[v]),
        );
        cheapest(q, &[ends[0], ends[1], midpoint])
    }

    /// `_PatchState._turn_ok`.
    fn turn_ok(
        &mut self,
        keep: usize,
        drop: usize,
        shared: &[u32],
        x: f64,
        y: f64,
        z: f64,
    ) -> bool {
        self.mark_tri.bump();
        for &t in shared {
            self.mark_tri.set(t as usize);
        }
        for source in [keep, drop] {
            let star = std::mem::take(&mut self.vtris[source]);
            for &t in &star {
                if self.mark_tri.has(t as usize) {
                    continue;
                }
                let base = 3 * t as usize;
                let (i, j, m) = (
                    self.tv[base] as usize,
                    self.tv[base + 1] as usize,
                    self.tv[base + 2] as usize,
                );
                let (mut ax, mut ay, mut az) = (self.px[i], self.py[i], self.pz[i]);
                let (mut bx, mut by, mut bz) = (self.px[j], self.py[j], self.pz[j]);
                let (mut cx, mut cy, mut cz) = (self.px[m], self.py[m], self.pz[m]);
                let nb = match normal(ax, ay, az, bx, by, bz, cx, cy, cz) {
                    None => continue, // already degenerate; nothing to turn
                    Some(n) => n,
                };
                // A triangle holding *both* ends is in `shared` and was skipped,
                // so exactly one corner moves and the chain is exhaustive.
                if i == keep || i == drop {
                    ax = x;
                    ay = y;
                    az = z;
                } else if j == keep || j == drop {
                    bx = x;
                    by = y;
                    bz = z;
                } else {
                    cx = x;
                    cy = y;
                    cz = z;
                }
                let na = match normal(ax, ay, az, bx, by, bz, cx, cy, cz) {
                    None => {
                        self.vtris[source] = star;
                        return false; // collapsed to a line or a point
                    }
                    Some(n) => n,
                };
                if nb.0 * na.0 + nb.1 * na.1 + nb.2 * na.2 < self.settings.cos_limit {
                    self.vtris[source] = star;
                    return false;
                }
            }
            self.vtris[source] = star;
        }
        true
    }

    // -- queue --------------------------------------------------------------

    /// `_PatchState._versions`.
    #[inline]
    fn versions(&self, u: usize, v: usize) -> u64 {
        ((self.version[u] as u64) << 32) | (self.version[v] as u64)
    }

    /// `_PatchState._push`.
    fn push(&mut self, u: usize, v: usize) {
        if !(self.alive[u] && self.alive[v]) {
            return;
        }
        if self.locked[u] && self.locked[v] {
            return;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        let q = self.quadric_sum(a, b);
        let (x, y, z) = self.place(&q, a, b);
        let mut cost = quadric_error(&q, x, y, z);
        if cost < 0.0 {
            cost = 0.0; // rounding under a flat quadric
        }
        if cost > self.settings.error_limit {
            self.rejected_error += 1;
            return;
        }
        self.heap.push(Entry {
            cost,
            u: a as u32,
            v: b as u32,
            versions: self.versions(a, b),
        });
    }

    /// `_PatchState._heap_limit`.
    fn heap_limit_now(&self) -> usize {
        let slack = (HEAP_SLACK * 1.5 * self.live_triangles as f64) as i64;
        std::cmp::max(1 << 16, slack) as usize
    }

    /// `_PatchState._compact`.
    fn compact(&mut self) {
        let entries = std::mem::take(&mut self.heap).into_vec();
        let kept: Vec<Entry> = entries
            .into_iter()
            .filter(|e| {
                let (u, v) = (e.u as usize, e.v as usize);
                self.alive[u] && self.alive[v] && e.versions == self.versions(u, v)
            })
            .collect();
        let size = kept.len();
        self.heap = BinaryHeap::from(kept);
        self.heap_limit = std::cmp::max(self.heap_limit_now(), 2 * size);
    }

    // -- output -------------------------------------------------------------

    /// The state `decimate.py`'s `finish()` reads, handed back for it to
    /// compact. The compaction itself is vectorised NumPy and stays there.
    pub fn positions(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.px, &self.py, &self.pz)
    }
    pub fn triangle_vertices(&self) -> &[u32] {
        &self.tv
    }
    pub fn alive_flags(&self) -> &[bool] {
        &self.alive
    }
    pub fn moved_flags(&self) -> &[bool] {
        &self.moved
    }
    pub fn triangle_alive_flags(&self) -> &[bool] {
        &self.tri_alive
    }
    pub fn live_triangle_count(&self) -> i64 {
        self.live_triangles
    }
}

/// Python's `min(candidates, key=...)`, which keeps the **first** minimum on a
/// tie. A `<=` here would silently prefer the last and is the kind of one-
/// character difference the equivalence test exists to catch.
#[inline]
fn cheapest(q: &[f64; 10], candidates: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let mut best = candidates[0];
    let mut best_cost = quadric_error(q, best.0, best.1, best.2);
    for &c in &candidates[1..] {
        let cost = quadric_error(q, c.0, c.1, c.2);
        if cost < best_cost {
            best = c;
            best_cost = cost;
        }
    }
    best
}
