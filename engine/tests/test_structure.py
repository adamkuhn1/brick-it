import numpy as np

from engine.colors import colorize
from engine.hollow import hollow
from engine.layout import layout
from engine.structure import (
    _State,
    adjacency,
    com_warning,
    components,
    repair,
    weak_bridges,
    weak_joints,
)
from engine.types import Brick
from engine.voxelize import voxelize
from meshes import arch, cube, l_shape, overhanging_t, sphere


def _built(mesh, size):
    g = hollow(colorize(voxelize(mesh, size, up="z")), size)
    return repair(layout(g), g), g


def test_bridge_detection():
    # Tower of three 1x2s, each overlapping the next by 1 stud; plus one more on top.
    b = lambda i, x, z: Brick(i, "3004", 1, 2, x, 0, z * 3, 90, 0)
    bricks = [b(0, 0, 0), b(1, 1, 1), b(2, 2, 2), b(3, 2, 3)]
    adj = adjacency(bricks, (4, 1, 4))
    assert len(components(bricks, adj=adj)) == 1
    # 0-1 splits off 1 brick (ignored); 1-2 splits 2|2 (weak); 2-3 overlap 2 (not weak)
    assert weak_bridges(adj) == [(1, 2)] or weak_bridges(adj) == [(2, 1)]


def test_shapes_single_component_no_floating():
    for mesh, size in [(cube(), 12), (sphere(), 16), (l_shape(), 16), (arch(), 16), (overhanging_t(), 16)]:
        (bricks, warnings), g = _built(mesh, size)
        assert len(components(bricks, g.shape)) == 1, (mesh, warnings)
        # ids are sequential
        assert [b.id for b in bricks] == list(range(len(bricks)))


def test_overhanging_t_warns_or_balances():
    (bricks, warnings), _ = _built(overhanging_t(), 16)
    assert com_warning(bricks) is None  # symmetric T is balanced


def test_com_warning_for_lopsided():
    b = lambda i, x, z: Brick(i, "3008", 1, 8, x, 0, z * 3, 90, 0)
    base = Brick(0, "3005", 1, 1, 0, 0, 0, 0, 0)
    arm = b(1, 0, 1)
    assert com_warning([base, arm]) is not None


def test_floating_part_gets_connected():
    g = hollow(colorize(voxelize(cube(), 8, up="z")), 8)
    g = g.copy()
    # carve a gap so the top two layers float
    from engine.types import EMPTY
    g[:, :, 3] = EMPTY
    bricks, warnings = repair(layout(g), g)
    assert len(components(bricks, g.shape)) == 1
    assert any("support" in w for w in warnings)


def _ordered(adj):
    return [(k, list(v.items())) for k, v in adj.items()]


def test_state_graph_tracks_adjacency():
    """The graph _State keeps up to date matches a full rebuild after every re-tile, undo and
    support, down to dict order (repair's searches depend on it)."""
    g = hollow(colorize(voxelize(sphere(), 12, up="z")), 12)
    state = _State(layout(g), g, graph=True)
    rng = np.random.default_rng(0)
    X, Y, Z = g.shape
    changes = 0
    for _ in range(300):
        z, x, y = int(rng.integers(Z)), int(rng.integers(X)), int(rng.integers(Y))
        sx, sy = (int(v) for v in rng.choice([1, 2, 3, 4], 2))
        if state.force(z, x, y, sx, sy) is None:
            continue
        changes += 1
        assert _ordered(state.adj) == _ordered(adjacency(state.list(), g.shape))
        if rng.random() < 0.5:
            state.undo()
            assert _ordered(state.adj) == _ordered(adjacency(state.list(), g.shape))
    assert changes > 20

    # A support column joining a floating 1x2 to the one on the ground two courses down.
    b = lambda i, z: Brick(i, "3004", 1, 2, 0, 0, z * 3, 90, 0)
    g = np.zeros((2, 1, 4), dtype=np.int16)
    state = _State([b(0, 0), b(1, 3)], g, graph=True)
    assert state.add_support(0, 0, 3) == 2
    assert _ordered(state.adj) == _ordered(adjacency(state.list(), g.shape))
    assert len(components(state.list(), adj=state.adj)) == 1


def test_repair_skips_known_weak_joints():
    # The tower from test_bridge_detection, in a grid with room to fix its 1-stud joint.
    b = lambda i, x, z: Brick(i, "3004", 1, 2, x, 0, z * 3, 90, 0)
    tower = lambda: [b(0, 0, 0), b(1, 1, 1), b(2, 2, 2), b(3, 2, 3)]
    g = np.zeros((4, 1, 4), dtype=np.int16)
    known = weak_joints(tower(), g.shape)
    assert len(known) == 1
    fixed, _ = repair(tower(), g)
    assert not weak_joints(fixed, g.shape)
    kept, warnings = repair(tower(), g, skip=known)
    assert weak_joints(kept, g.shape) == known
    assert "1 weak joint held by a single stud." in warnings
