"""Gate C — what a DEC-005 candidate must be *able* to do — WP-4.1.

Gate C is the capability gate the Round 15 brief states in advance, and it is
absolute: a candidate that fails any clause is disqualified whatever its size.

    1. express the DEC-004 tier distinction **structurally** — a reader can tell
       a client tier from a QC tier without parsing geometry and without
       trusting a filename;
    2. carry a tier that contains **no per-vertex source identity of any kind**;
    3. represent the LOD chain and per-tile error bounds.

How these are proven, and why the shape matters (standing rule 8)
------------------------------------------------------------------
The pre-change tree has no container at all, so "this test fails before the
change" would only establish that the capability is new. It would not establish
that the check observes the rule it names — a green that would stay green if the
rule were violated is worth nothing.

So every clause below is proven **by mutation, one rule at a time**. Each test
writes the same tier twice: once correctly, once through a writer broken on
exactly that clause and on no other (`container.Defects`), and asserts green on
the first and red on the second. A test that passed because the writer happens
to be correct in some *other* way cannot survive that.

`Defects` is unreachable from the pipeline: no production caller constructs it,
which `test_the_pipeline_never_constructs_a_defect` holds.

No real data is used here. The fixture is synthetic, which is the correct choice
for a capability test — capability is a property of the format, and the real
measurement lives in the Round 15 harness under `MEASUREMENTS/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidmesh.container import (
    CLIENT_VERTEX_ATTRIBUTES,
    ByteLedger,
    ContainerError,
    Defects,
    TierAudience,
    TierDescriptor,
    TileGeometry,
    strip_for_client,
)
from rapidmesh.container_audit import audit_identity
from rapidmesh.container_binary import (
    AUDIENCE_OFFSET,
    CODEC_FILTERED,
    CODEC_STORED,
    audience_byte,
    binary_stream_offsets,
    binary_vertex_attributes,
    read_binary_entries,
    read_binary_header,
    read_binary_tile,
    write_binary_tier,
)
from rapidmesh.container_codec import byte_shuffle, byte_unshuffle, delta_rows, undelta_rows
from rapidmesh.container_gltf import write_gltf_tier
from rapidmesh.container_gltf_read import (
    gltf_vertex_attributes,
    read_gltf_entries,
    read_gltf_header,
    read_gltf_tile,
)

# Source ids are deliberately large and irregular. Real global vertex indices
# run to tens of millions, so their top bytes are zero and a run of four is
# mostly zeros; a fixture of 0..V-1 would make the byte audit look better than
# it is by giving it an unrealistically distinctive needle.
_ID_BASE = 0x0000_0BAD_C0DE_0000


def _tile(tile_id: int, rows: int = 17, cols: int = 13) -> TileGeometry:
    """One synthetic tile: a warped grid, its triangles, and source identity."""
    y, x = np.meshgrid(
        np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij"
    )
    z = np.sin(x * 0.37 + tile_id) * 0.11 + np.cos(y * 0.23) * 0.07
    positions = np.stack([x * 0.05, y * 0.05, z], axis=-1).reshape(-1, 3).astype(np.float32)
    quads = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            quads.append((a, a + 1, a + cols))
            quads.append((a + 1, a + cols + 1, a + cols))
    triangles = np.asarray(quads, np.uint32)
    ids = (_ID_BASE + tile_id * 1_000_003 + np.arange(positions.shape[0], dtype=np.int64) * 7)
    return TileGeometry(
        tile_id=tile_id,
        positions=positions,
        triangles=triangles,
        plane_deviation_bound_m=0.0064 * (1.0 + tile_id * 0.01),
        source_vertex_id=ids,
    )


def _tier(audience: TierAudience = TierAudience.CLIENT) -> TierDescriptor:
    return TierDescriptor(
        audience=audience,
        lod_level=1,
        lod_count=3,
        error_budget_m=0.0064,
        station_ordinal=1,
    )


def _client_tiles(count: int = 3) -> list[TileGeometry]:
    return [strip_for_client(_tile(i)) for i in range(count)]


def _qc_tiles(count: int = 3) -> list[TileGeometry]:
    return [_tile(i) for i in range(count)]


def _id_arrays(tiles: list[TileGeometry]) -> list[np.ndarray]:
    """One id array per tile — the shape `audit_identity` needs to match in one piece."""
    return [_tile(t.tile_id).source_vertex_id for t in tiles]


# ---------------------------------------------------------------------------
# Gate C clause 2 — no per-vertex source identity in a client tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", [CODEC_STORED, CODEC_FILTERED])
def test_a_client_binary_tier_holds_no_source_identity_bytes(tmp_path, codec):
    """The DEC-004 clause, observed in bytes rather than asserted.

    Observes: that the int64 source ids do not occur in the file, raw or behind
    DEFLATE, that no per-vertex attribute outside the allow-list is declared,
    and that every byte written is attributed to a named section so there is no
    unaccounted region for them to be in.

    Would fail if: the writer carried the field under any name, or under none,
    or compressed — the mutant below does exactly that and is red.
    """
    tiles = _client_tiles()
    ids = _id_arrays(tiles)

    good = tmp_path / "client.bin"
    ledger = write_binary_tier(good, _tier(), tiles, codec=codec)
    audit = audit_identity(
        good.read_bytes(),
        ids,
        binary_vertex_attributes(good),
        ledger,
        stream_offsets=binary_stream_offsets(good),
    )
    assert audit.needles_tried > 0
    assert audit.raw_hits == 0
    assert audit.inflated_hits == 0
    assert audit.ledger_balanced
    assert audit.clean

    # The mutation: the same writer, broken on this clause alone.
    bad = tmp_path / "leaked.bin"
    bad_ledger = write_binary_tier(
        bad, _tier(), _qc_tiles(), codec=codec, defects=Defects(leak_source_identity=True)
    )
    bad_audit = audit_identity(
        bad.read_bytes(),
        ids,
        binary_vertex_attributes(bad),
        bad_ledger,
        stream_offsets=binary_stream_offsets(bad),
    )
    assert not bad_audit.clean
    if codec == CODEC_STORED:
        assert bad_audit.raw_hits > 0
    else:
        assert bad_audit.inflated_hits > 0, "a compressed leak must still be caught"


def test_a_client_gltf_tier_holds_no_source_identity_bytes(tmp_path):
    """Same clause, same standard of proof, on the standard route.

    Observes the same three things. The mutant writes identity as a custom
    `_SOURCE_VERTEX_ID` attribute — which is what a glTF author would naturally
    do — so the check is not passing merely because glTF has no int64 type.
    """
    tiles = _client_tiles()
    ids = _id_arrays(tiles)

    good = tmp_path / "client.glb"
    ledger = write_gltf_tier(good, _tier(), tiles)
    audit = audit_identity(good.read_bytes(), ids, gltf_vertex_attributes(good), ledger)
    assert audit.needles_tried > 0
    assert audit.raw_hits == 0
    assert audit.inflated_hits == 0
    assert audit.ledger_balanced
    assert audit.clean
    assert set(gltf_vertex_attributes(good)) <= set(CLIENT_VERTEX_ATTRIBUTES)

    bad = tmp_path / "leaked.glb"
    bad_ledger = write_gltf_tier(
        bad, _tier(), _qc_tiles(), defects=Defects(leak_source_identity=True)
    )
    bad_audit = audit_identity(bad.read_bytes(), ids, gltf_vertex_attributes(bad), bad_ledger)
    assert bad_audit.raw_hits > 0
    assert not bad_audit.clean
    assert "_SOURCE_VERTEX_ID" in gltf_vertex_attributes(bad)


def test_both_writers_refuse_a_client_tier_that_still_carries_identity(tmp_path):
    """The refusal is at the writer, so an unstripped tile never reaches bytes.

    Would fail if the guard moved to the reader: a reader-side filter puts the
    field on the wire and then declines to look at it, which is not what DEC-004
    says.
    """
    with pytest.raises(ContainerError, match="client tier carries no per-vertex"):
        write_gltf_tier(tmp_path / "a.glb", _tier(), _qc_tiles())
    with pytest.raises(ContainerError, match="client tier carries no per-vertex"):
        write_binary_tier(tmp_path / "a.bin", _tier(), _qc_tiles())
    assert not (tmp_path / "a.glb").exists()
    assert not (tmp_path / "a.bin").exists()


def test_a_qc_tier_may_carry_identity_because_it_never_leaves_the_server(tmp_path):
    """DEC-004 keeps the QC tier server-side; it does not pretend it has no data.

    A format with nowhere to put source identity would force the QC tier into a
    second format. Both candidates carry it on the QC tier and neither on the
    client tier, which is the distinction DEC-004 actually draws.
    """
    qc = _tier(TierAudience.QC)
    binary = tmp_path / "qc.bin"
    write_binary_tier(binary, qc, _qc_tiles())
    assert "_SOURCE_VERTEX_ID" in binary_vertex_attributes(binary)
    assert read_binary_tile(binary, 0).source_vertex_id is not None

    gltf = tmp_path / "qc.glb"
    write_gltf_tier(gltf, qc, _qc_tiles())
    assert read_gltf_header(gltf).audience is TierAudience.QC
    # glTF 2.0 has no 64-bit integer component type, so the standard route can
    # only carry this under a private attribute as paired uint32. Recorded as a
    # limitation of that route on the QC tier, not worked around.
    assert "_SOURCE_VERTEX_ID" in gltf_vertex_attributes(gltf)


# ---------------------------------------------------------------------------
# Gate C clause 1 — the tier distinction is structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("audience", [TierAudience.CLIENT, TierAudience.QC])
def test_the_binary_audience_survives_a_lying_filename(tmp_path, audience):
    """Observes: the audience comes from a byte at a fixed offset, not the name.

    The file is deliberately named after the *other* audience. Would fail if any
    reader inferred the audience from the path, or if the byte moved.
    """
    other = TierAudience.QC if audience is TierAudience.CLIENT else TierAudience.CLIENT
    path = tmp_path / f"definitely-a-{other}-tier.bin"
    tiles = _client_tiles() if audience is TierAudience.CLIENT else _qc_tiles()
    write_binary_tier(path, _tier(audience), tiles)

    header = read_binary_header(path)
    assert header.audience is audience
    assert header.header_bytes == 96
    assert header.header_bytes % 4 == 0, "an unaligned header forces a copy of every block"
    assert header.header_bytes < header.total_bytes / 10
    assert audience_byte(path) == (1 if audience is TierAudience.CLIENT else 2)
    assert AUDIENCE_OFFSET == 10


def test_the_gltf_audience_survives_a_lying_filename(tmp_path):
    """Same clause on the standard route, and the JSON chunk is all that is read.

    Would fail if the audience were absent, if it were listed only in
    `extensionsUsed` — a loader may ignore an optional extension, which would
    put DEC-004 in the loader's gift — or if reading it required the BIN chunk.
    """
    path = tmp_path / "definitely-a-qc-tier.glb"
    write_gltf_tier(path, _tier(TierAudience.CLIENT), _client_tiles())
    header = read_gltf_header(path)
    assert header.audience is TierAudience.CLIENT
    assert header.header_bytes < header.total_bytes
    # The JSON chunk and nothing after it.
    assert header.header_bytes < header.total_bytes / 2


@pytest.mark.parametrize("suffix", [".glb", ".bin"])
def test_an_unattributed_tier_is_refused_by_the_reader(tmp_path, suffix):
    """The mutation for clause 1, broken on that clause alone.

    Observes: a tier whose audience is missing cannot be read as if it were a
    client tier. Would fail if a reader defaulted the audience to anything.
    """
    path = tmp_path / f"unattributed{suffix}"
    defects = Defects(omit_audience=True)
    if suffix == ".glb":
        write_gltf_tier(path, _tier(), _client_tiles(), defects=defects)
        with pytest.raises(ContainerError, match="audience"):
            read_gltf_header(path)
    else:
        write_binary_tier(path, _tier(), _client_tiles(), defects=defects)
        with pytest.raises(ContainerError, match="unattributed"):
            read_binary_header(path)


def test_reading_the_audience_does_not_touch_the_geometry(tmp_path):
    """Observes literally that: the geometry is removed and the answer is unchanged.

    The file is truncated to its declared header length. If either reader
    touched a byte of geometry to answer the audience question, it would now
    fail. Would fail if a reader validated the buffer, or hashed the file.
    """
    glb = tmp_path / "tier.glb"
    write_gltf_tier(glb, _tier(), _client_tiles())
    head = read_gltf_header(glb)
    (tmp_path / "head.glb").write_bytes(glb.read_bytes()[: head.header_bytes])
    assert read_gltf_header(tmp_path / "head.glb").audience is TierAudience.CLIENT

    binary = tmp_path / "tier.bin"
    write_binary_tier(binary, _tier(), _client_tiles())
    head_b = read_binary_header(binary)
    (tmp_path / "head.bin").write_bytes(binary.read_bytes()[: head_b.header_bytes])
    assert read_binary_header(tmp_path / "head.bin").audience is TierAudience.CLIENT


# ---------------------------------------------------------------------------
# Gate C clause 3 — the LOD chain and per-tile error bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".glb", ".bin"])
def test_the_lod_chain_and_budget_are_carried(tmp_path, suffix):
    """Observes: which level of how many, and the budget the level was built to.

    Would fail if a tier could not say where it sits in its chain — a viewer
    switching LODs has to know that from the file and not from a sidecar.
    """
    path = tmp_path / f"tier{suffix}"
    tier = TierDescriptor(TierAudience.CLIENT, 1, 3, 0.0064, 1)
    if suffix == ".glb":
        write_gltf_tier(path, tier, _client_tiles())
        header = read_gltf_header(path)
    else:
        write_binary_tier(path, tier, _client_tiles())
        header = read_binary_header(path)
    assert (header.lod_level, header.lod_count) == (1, 3)
    assert header.error_budget_m == pytest.approx(0.0064)
    assert header.tile_count == 3


@pytest.mark.parametrize("suffix", [".glb", ".bin"])
def test_every_tile_carries_its_own_plane_bound(tmp_path, suffix):
    """Observes: a per-tile bound, readable without decoding that tile's geometry.

    The bound is over the **planes** of the merged triangles and not over the
    surface (ITEM-027); the field name carries that scope. Mutation below omits
    the bound and only the bound.
    """
    tiles = _client_tiles(4)
    path = tmp_path / f"tier{suffix}"
    reader = read_gltf_entries if suffix == ".glb" else read_binary_entries
    writer = write_gltf_tier if suffix == ".glb" else write_binary_tier

    writer(path, _tier(), tiles)
    entries = reader(path)
    assert len(entries) == 4
    for entry, tile in zip(entries, tiles, strict=True):
        assert entry.plane_deviation_bound_m == pytest.approx(tile.plane_deviation_bound_m)
        assert entry.vertex_count == tile.vertex_count
        assert entry.triangle_count == tile.triangle_count
        assert entry.byte_length > 0

    bad = tmp_path / f"unbounded{suffix}"
    writer(bad, _tier(), tiles, defects=Defects(omit_bounds=True))
    with pytest.raises(ContainerError, match="no error bound"):
        reader(bad)


# ---------------------------------------------------------------------------
# properties both candidates owe the product rather than the gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", [CODEC_STORED, CODEC_FILTERED])
def test_binary_geometry_round_trips_identity_exact(tmp_path, codec):
    """Bit-for-bit, not to a tolerance. A container that moves a vertex is a
    silent alignment (§4 rule 4) wearing a compression ratio.

    Would fail if the codec were lossy, if the filter were not exactly
    invertible, or if a dtype narrowed anywhere on the path.
    """
    tiles = _client_tiles(4)
    path = tmp_path / "tier.bin"
    write_binary_tier(path, _tier(), tiles, codec=codec)
    for index, tile in enumerate(tiles):
        back = read_binary_tile(path, index)
        assert back.tile_id == tile.tile_id
        assert back.positions.dtype == np.float32
        assert back.triangles.dtype == np.uint32
        assert np.array_equal(
            back.positions.view(np.uint32), tile.positions.view(np.uint32)
        ), "positions must round-trip on their bits, not to a tolerance"
        assert np.array_equal(back.triangles, tile.triangles)
        assert back.source_vertex_id is None


def test_gltf_geometry_round_trips_identity_exact(tmp_path):
    tiles = _client_tiles(4)
    path = tmp_path / "tier.glb"
    write_gltf_tier(path, _tier(), tiles)
    for index, tile in enumerate(tiles):
        back = read_gltf_tile(path, index)
        assert np.array_equal(back.positions.view(np.uint32), tile.positions.view(np.uint32))
        assert np.array_equal(back.triangles, tile.triangles)
        assert back.source_vertex_id is None


@pytest.mark.parametrize("suffix", [".glb", ".bin"])
def test_one_tile_reads_from_a_tier_whose_later_tiles_are_gone(tmp_path, suffix):
    """ITEM-028, as a property rather than as an intention.

    The tier is truncated just past tile 0's last byte. If either reader
    materialised the tier to hand back one tile, this read would fail. Would
    fail if a reader joined tiles, or validated a trailing digest over geometry.
    """
    tiles = _client_tiles(5)
    path = tmp_path / f"tier{suffix}"
    if suffix == ".glb":
        write_gltf_tier(path, _tier(), tiles)
        entries = read_gltf_entries(path)
        reader = read_gltf_tile
    else:
        write_binary_tier(path, _tier(), tiles)
        entries = read_binary_entries(path)
        reader = read_binary_tile
    end = entries[0].byte_offset + entries[0].byte_length
    cut = tmp_path / f"cut{suffix}"
    cut.write_bytes(path.read_bytes()[:end])
    assert cut.stat().st_size < path.stat().st_size

    back = reader(cut, 0)
    assert np.array_equal(back.positions.view(np.uint32), tiles[0].positions.view(np.uint32))
    assert np.array_equal(back.triangles, tiles[0].triangles)


def test_the_lossless_filter_is_exactly_invertible():
    """The codec's own correctness, separate from any container.

    Would fail if the shuffle transposed the wrong way, or if the index delta
    relied on signed arithmetic and wrapped differently on the way back.
    """
    rng = np.random.default_rng(20260915)
    positions = rng.standard_normal((977, 3)).astype(np.float32)
    assert np.array_equal(
        byte_unshuffle(byte_shuffle(positions), np.float32, positions.shape).view(np.uint32),
        positions.view(np.uint32),
    )
    triangles = rng.integers(0, 1 << 20, size=(1013, 3)).astype(np.uint32)
    assert np.array_equal(undelta_rows(delta_rows(triangles)), triangles)
    # A descending run makes the delta wrap; the inverse must wrap with it.
    descending = np.stack(
        [np.arange(500, 0, -1, dtype=np.uint32)] * 3, axis=-1
    ).astype(np.uint32)
    assert np.array_equal(undelta_rows(delta_rows(descending)), descending)


def test_a_balanced_ledger_leaves_nowhere_for_an_unaccounted_array(tmp_path):
    """Observes: written bytes equal the sum of the named sections, exactly.

    This is the exhaustive half of the DEC-004 evidence. The byte search can
    only say "I did not find it"; a balanced ledger says there is no region it
    could have been in. Would fail if a writer emitted any block it did not
    name — which the leak mutant does, and which is why its ledger still
    balances only by naming the leak.
    """
    tiles = _client_tiles(4)
    for name, ledger in (
        ("glb", write_gltf_tier(tmp_path / "t.glb", _tier(), tiles)),
        ("bin", write_binary_tier(tmp_path / "t.bin", _tier(), tiles)),
    ):
        assert ledger.balanced, f"{name} ledger does not balance"
        assert ledger.total_bytes == (tmp_path / f"t.{name}").stat().st_size
        assert ledger.named("source-identity") == 0
        assert ledger.named("positions") > 0
        assert ledger.named("indices") > 0


def test_the_pipeline_never_constructs_a_defect():
    """`Defects` exists for sensitivity tests and must stay unreachable otherwise.

    Would fail the moment a production module imported it — which is the failure
    mode that turns a test-only switch into a shipping option.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "rapidmesh"
    users = [
        path.name
        for path in src.glob("*.py")
        if "Defects(" in path.read_text(encoding="utf-8")
    ]
    assert sorted(users) == ["container.py"], users
    assert not Defects().any_set


def test_an_empty_tier_is_writable_and_readable(tmp_path):
    """A station that meshes nothing is not a failed station (§4 rule 7)."""
    for path, writer, header_reader, entry_reader in (
        (tmp_path / "e.glb", write_gltf_tier, read_gltf_header, read_gltf_entries),
        (tmp_path / "e.bin", write_binary_tier, read_binary_header, read_binary_entries),
    ):
        ledger = writer(path, _tier(), [])
        assert isinstance(ledger, ByteLedger)
        assert ledger.balanced
        assert header_reader(path).tile_count == 0
        assert entry_reader(path) == []
