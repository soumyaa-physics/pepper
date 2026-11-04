import awkward as ak
import hist
import numpy as np
import pytest
import uproot
from coffea.nanoevents.methods import vector

from pepper import kinreco_sonnenschein as kr

MT, MW, MB = 172.5, 80.3, 4.8


def boost(p, parent):
    """Boost 4-vectors p (E, px, py, pz) from parent rest frame to lab"""
    beta = parent[:, 1:] / parent[:, :1]
    b2 = (beta ** 2).sum(axis=1, keepdims=True)
    gamma = 1 / np.sqrt(1 - b2)
    bp = (beta * p[:, 1:]).sum(axis=1, keepdims=True)
    gamma2 = np.where(b2 > 0, (gamma - 1) / np.where(b2 > 0, b2, 1), 0)
    p3 = p[:, 1:] + gamma2 * bp * beta + gamma * beta * p[:, :1]
    e = gamma * (p[:, :1] + bp)
    return np.concatenate([e, p3], axis=1)


def two_body(parent, m1, m2, rng):
    M = np.sqrt(parent[:, 0] ** 2 - (parent[:, 1:] ** 2).sum(axis=1))
    p = np.sqrt((M ** 2 - (m1 + m2) ** 2) * (M ** 2 - (m1 - m2) ** 2)) / (2 * M)
    cost = rng.uniform(-1, 1, len(M))
    phi = rng.uniform(0, 2 * np.pi, len(M))
    sint = np.sqrt(1 - cost ** 2)
    d = np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1)
    p1 = np.concatenate([np.sqrt(p ** 2 + m1 ** 2)[:, None], p[:, None] * d],
                        axis=1)
    p2 = np.concatenate([np.sqrt(p ** 2 + m2 ** 2)[:, None], -p[:, None] * d],
                        axis=1)
    return boost(p1, parent), boost(p2, parent)


def lv(p):
    return ak.zip({"t": p[:, 0], "x": p[:, 1], "y": p[:, 2], "z": p[:, 3]},
                  with_name="LorentzVector", behavior=vector.behavior)


def make_events(n, seed=1):
    """Dileptonic ttbar with exact masses, random kinematics"""
    rng = np.random.default_rng(seed)
    mtt = rng.uniform(400, 900, n)
    pz = rng.normal(0, 300, n)
    ptt = rng.normal(0, 30, (n, 2))
    tt = np.concatenate([np.sqrt(mtt ** 2 + pz ** 2 + (ptt ** 2).sum(1))[
        :, None], ptt, pz[:, None]], axis=1)
    t, at = two_body(tt, MT, MT, rng)
    wp, b = two_body(t, MW, MB, rng)
    wm, ab = two_body(at, MW, MB, rng)
    alep, nu = two_body(wp, 0., 0., rng)
    lep, anu = two_body(wm, 0., 0., rng)
    nunu = nu + anu
    met = ak.zip({"pt": np.hypot(nunu[:, 1], nunu[:, 2]),
                  "phi": np.arctan2(nunu[:, 2], nunu[:, 1]),
                  "eta": np.zeros(n), "mass": np.zeros(n)},
                 with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    return {"lep": lv(lep), "antilep": lv(alep), "b": lv(b),
            "antib": lv(ab), "met": met}, lv(t), lv(at)


@pytest.fixture(scope="module")
def events():
    return make_events(50)


def test_truth_events_have_solutions(events):
    inputs, _, _ = events
    t, at = kr.sonnenschein(**inputs)
    assert len(t) == 50
    # Exact kinematics: the truth is always a solution
    assert ak.mean(ak.num(t) == 1) > 0.95
    np.testing.assert_allclose(ak.flatten(t.mass), MT, rtol=1e-6)
    np.testing.assert_allclose(ak.flatten(at.mass), MT, rtol=1e-6)


def test_transverse_momentum_balance(events):
    inputs, _, _ = events
    t, at = kr.sonnenschein(**inputs)
    has = ak.num(t) == 1
    vis = (inputs["lep"] + inputs["antilep"] + inputs["b"] + inputs["antib"])
    met = inputs["met"]
    np.testing.assert_allclose(
        ak.flatten((t + at).x), (vis.x + met.x)[has], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(
        ak.flatten((t + at).y), (vis.y + met.y)[has], rtol=1e-5, atol=1e-3)


def test_finds_truth_mostly(events):
    """Smallest-mtt solution is the true one in most events"""
    inputs, t_true, _ = events
    t, _ = kr.sonnenschein(**inputs)
    has = ak.num(t) == 1
    dist = np.abs(ak.flatten(t.z) - t_true.z[has])
    assert np.mean(dist < 1.) > 0.5


def test_jagged_input(events):
    inputs, _, _ = events
    jagged = {k: ak.unflatten(v, np.ones(len(v), dtype=int))
              for k, v in inputs.items()}
    t1, _ = kr.sonnenschein(**inputs)
    t2, _ = kr.sonnenschein(**jagged)
    assert ak.all(ak.flatten(t1.x) == ak.flatten(t2.x))


def test_chunking_consistent(events):
    inputs, _, _ = events
    t1, at1 = kr.sonnenschein(**inputs)
    t2, at2 = kr.sonnenschein(**inputs, chunksize=7)
    assert t1.x.tolist() == t2.x.tolist()
    assert at1.z.tolist() == at2.z.tolist()


def test_trivial_smearing_equals_none(events, smear_hists):
    inputs, _, _ = events
    t1, _ = kr.sonnenschein(**inputs)
    t2, _ = kr.sonnenschein(**inputs, num_smear=5, energyfl=1., energyfj=1.,
                            alphal=0., alphaj=0.,
                            hist_mlb=smear_hists["hist_mlb"],
                            rng=np.random.default_rng(1))
    np.testing.assert_allclose(ak.flatten(t2.x), ak.flatten(t1.x),
                               rtol=1e-6)
    # num_smear=0 is the same as None
    t3, _ = kr.sonnenschein(**inputs, num_smear=0)
    assert t3.x.tolist() == t1.x.tolist()


@pytest.fixture(scope="module")
def smear_hists(tmp_path_factory):
    path = tmp_path_factory.mktemp("kr") / "kinreco.root"
    rng = np.random.default_rng(3)
    with uproot.recreate(path) as f:
        h = hist.Hist(hist.axis.Regular(50, 0.8, 1.2))
        h.fill(rng.normal(1, 0.05, 10000))
        f["energyf"] = h
        h = hist.Hist(hist.axis.Regular(50, 0, 0.1))
        h.fill(np.abs(rng.normal(0, 0.02, 10000)))
        f["alpha"] = h
        h = hist.Hist(hist.axis.Regular(50, 0, 180))
        h.fill(rng.uniform(0, 180, 100000))
        f["mlb"] = h
    f = uproot.open(path)
    return {"energyfl": f["energyf"], "energyfj": f["energyf"],
            "alphal": f["alpha"], "alphaj": f["alpha"], "hist_mlb": f["mlb"]}


def test_smearing_reproducible(events, smear_hists):
    inputs, _, _ = events
    res = [kr.sonnenschein(**inputs, num_smear=20, **smear_hists,
                           rng=np.random.default_rng(42))[0]
           for _ in range(2)]
    assert res[0].x.tolist() == res[1].x.tolist()
    assert ak.mean(ak.num(res[0]) == 1) > 0.8
    other = kr.sonnenschein(**inputs, num_smear=20, **smear_hists,
                            rng=np.random.default_rng(43))[0]
    assert res[0].x.tolist() != other.x.tolist()


def test_random_orthogonal_single_vector():
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(1, 1, 3))
    u = kr._random_orthogonal(vec, rng)
    np.testing.assert_allclose((u * vec).sum(-1), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1)


def test_rotate_axis():
    vec = np.array([[1., 0., 0.]])
    axis = np.array([[0., 0., 1.]])
    np.testing.assert_allclose(kr._rotate_axis(vec, axis, np.pi / 2),
                               [[0., 1., 0.]], atol=1e-12)


@pytest.mark.xfail(strict=True, reason="Bug: np.linalg.norm without axis=-1 "
                   "in _random_orthogonal normalises over all vectors")
def test_random_orthogonal_and_rotate():
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(1, 10, 3))
    u = kr._random_orthogonal(vec, rng)
    np.testing.assert_allclose((u * vec).sum(-1), 0, atol=1e-10)
    rot = kr._rotate_axis(vec, u, np.full((1, 10), 0.3))
    np.testing.assert_allclose(np.linalg.norm(rot, axis=-1),
                               np.linalg.norm(vec, axis=-1))
    cos = (rot * vec).sum(-1) / np.linalg.norm(vec, axis=-1) ** 2
    np.testing.assert_allclose(cos, np.cos(0.3))


def test_roots_vectorized():
    poly = np.array([[1., -3., 2.], [1., 0., -4.]])
    roots = np.sort(kr._roots_vectorized(poly).real, axis=-1)
    np.testing.assert_allclose(roots, [[1., 2.], [-2., 2.]])
