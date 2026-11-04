import pickle

import awkward as ak
import numpy as np
import pytest

from pepper import Selector


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, data, systematics, cut, done_steps, cats):
        self.calls.append({"n": len(data), "cut": cut,
                           "steps": set(done_steps), "cats": dict(cats),
                           "systematics": systematics})


def test_init(events):
    sel = Selector(events)
    assert sel.num == 6
    assert sel.cutnames == ["BeforeCuts"]
    assert sel.systematics is None
    assert "cut:BeforeCuts" in sel.done_steps


def test_init_weight(events):
    sel = Selector(events, weight=np.full(6, 2.))
    assert ak.all(sel.systematics.weight == 2.)


def test_add_cut_mask(events):
    sel = Selector(events)
    sel.add_cut("met", lambda d: d.MET.pt > 25)
    assert sel.num == 4
    assert ak.all(sel.data.MET.pt > 25)
    assert sel.cutnames == ["BeforeCuts", "met"]
    assert "cut:met" in sel.done_steps


def test_add_cut_array(events):
    sel = Selector(events)
    sel.add_cut("first", np.array([True, False] * 3))
    assert sel.num == 3


def test_add_cut_masked_values_fail(events):
    sel = Selector(events)
    accept = ak.mask(np.full(6, True), np.array([True] * 3 + [False] * 3))
    sel.add_cut("masked", accept)
    assert sel.num == 3


def test_add_cut_weighted(events):
    sel = Selector(events, weight=np.ones(6))
    sel.add_cut("sf", np.array([0., 0.5, 1., 2., 0., 1.]))
    assert sel.num == 4
    np.testing.assert_allclose(sel.systematics.weight, [0.5, 1., 2., 1.])


def test_add_cut_duplicate_name(events):
    sel = Selector(events)
    sel.add_cut("a", np.full(6, True))
    with pytest.raises(ValueError):
        sel.add_cut("a", np.full(6, True))


def test_add_cut_with_systematics_tuple(events):
    sel = Selector(events, weight=np.ones(6))
    mask = np.array([True, True, False, True, True, False])
    up = np.arange(6) + 1.
    down = np.arange(6) + 10.
    sel.add_cut("c", (mask, {"sf": (up, down)}))
    np.testing.assert_allclose(sel.systematics.sf_up, up[mask])
    np.testing.assert_allclose(sel.systematics.sf_down, down[mask])
    assert sel.cut_systematic_map["c"] == ["sf_up", "sf_down"]


def test_add_cut_systematics_after_cut_length(events):
    sel = Selector(events, weight=np.ones(6))
    mask = np.array([True, True, False, True, True, False])
    sel.add_cut("c", mask, systematics={"s": np.full(4, 3.)})
    np.testing.assert_allclose(sel.systematics.s, 3.)


def test_add_cut_systematics_list_rejected(events):
    sel = Selector(events, weight=np.ones(6))
    with pytest.raises(ValueError):
        sel.add_cut("c", np.full(6, True),
                    systematics={"s": [np.ones(6), np.ones(6)]})


def test_add_cut_categories(events):
    sel = Selector(events)
    sel.set_cat("channel", {"is_ee", "is_mm", "is_em"})
    seen = []

    def cut(data):
        seen.append(len(data))
        return np.full(len(data), False)
    sel.add_cut("ee_only", cut, categories={"channel": ["is_ee"]})
    # Only ee events were evaluated and removed, others kept
    assert seen == [2]
    assert sel.num == 4
    assert not ak.any(sel.data.is_ee)


def test_unknown_category(events):
    sel = Selector(events)
    with pytest.raises(ValueError):
        sel.add_cut("c", np.full(6, True), categories={"nope": ["is_ee"]})


def test_set_column(events):
    sel = Selector(events)
    sel.set_column("nmu", lambda d: ak.num(d.Muon))
    assert sel.data.nmu.tolist() == [2, 1, 0, 3, 1, 2]
    assert "column:nmu" in sel.done_steps
    with pytest.raises(ValueError):
        sel.set_column("bad", lambda d: None)


def test_set_column_categories(events):
    sel = Selector(events)
    sel.set_cat("channel", {"is_ee", "is_mm", "is_em"})
    sel.set_column("x", lambda d: d.MET.pt * 2,
                   categories={"channel": ["is_mm"]})
    assert sel.data.x.tolist() == [None, 40., None, None, 100., 120.]


def test_set_multiple_columns(events):
    sel = Selector(events)
    rec = Recorder()
    sel.on_update.append(rec)
    sel.set_multiple_columns(lambda d: {"a": d.MET.pt, "b": d.MET.pt + 1})
    assert ak.all(sel.data.b - sel.data.a == 1)
    # Only one callback for all columns
    assert len(rec.calls) == 1


def test_deferred_cuts(events):
    sel = Selector(events, weight=np.ones(6), applying_cuts=False)
    sel.add_cut("a", np.array([True, True, True, False, False, True]))
    sel.add_cut("b", np.array([1., 0., 0.5, 1., 1., 1.]))
    # Nothing removed yet
    assert sel.num == 6
    assert sel.num_final == 3
    assert ak.sum(~ak.is_none(sel.final)) == 3
    np.testing.assert_allclose(
        ak.fill_none(sel.final_systematics.weight, -1),
        [1., -1, 0.5, -1, -1, 1.])
    names, cuts = sel.get_cuts()
    assert names == ["BeforeCuts", "a", "b"]
    assert ak.fields(cuts) == ["a", "b"]
    sel.applying_cuts = True
    assert sel.num == 3
    np.testing.assert_allclose(sel.systematics.weight, [1., 0.5, 1.])
    assert len(sel.unapplied_cuts) == 0


def test_deferred_set_column_all_cuts(events):
    sel = Selector(events, applying_cuts=False)
    sel.add_cut("a", np.array([True, False, True, False, True, False]))
    seen = []

    def col(data):
        seen.append(len(data))
        return data.MET.pt
    sel.set_column("x", col, all_cuts=True)
    assert seen == [3]
    assert sel.data.x.tolist() == [10., None, 30., None, 50., None]


def test_deferred_systematics_weight_only(events):
    sel = Selector(events, weight=np.ones(6), applying_cuts=False)
    sel.add_cut("sf", np.full(6, 0.5), systematics={"s": np.full(6, 2.)})
    sel.applying_cuts = True
    np.testing.assert_allclose(sel.systematics.s, 2.)
    np.testing.assert_allclose(sel.systematics.weight, 0.5)


def test_deferred_systematics_post_cut_length(events):
    sel = Selector(events, weight=np.ones(6), applying_cuts=False)
    mask = np.array([True, False, True, False, True, False])
    sel.add_cut("a", mask, systematics={"s": np.full(3, 2.)})
    sel.applying_cuts = True
    np.testing.assert_allclose(sel.systematics.s, 2.)


@pytest.mark.xfail(strict=True, reason="Bug: ak.mask result passed to "
                   "np.array in set_systematic (selector.py:374)")
def test_deferred_systematics_full_length(events):
    sel = Selector(events, weight=np.ones(6), applying_cuts=False)
    mask = np.array([True, False, True, False, True, False])
    sel.add_cut("a", mask, systematics={"s": np.full(6, 2.)})
    sel.applying_cuts = True
    np.testing.assert_allclose(sel.systematics.s, 2.)


def test_on_update_args(events):
    rec = Recorder()
    sel = Selector(events, on_update=rec)
    sel.add_cut("a", np.array([True] * 5 + [False]))
    sel.add_cut("b", np.full(5, True), no_callback=True)
    sel.set_column("c", lambda d: d.MET.pt)
    assert [c["cut"] for c in rec.calls] == ["BeforeCuts", "a", "b"]
    assert rec.calls[1]["n"] == 5
    assert "column:c" in rec.calls[-1]["steps"]


def test_copy_independent(events):
    sel = Selector(events, weight=np.ones(6))
    cp = sel.copy()
    cp.add_cut("a", np.array([True] * 3 + [False] * 3))
    cp.set_column("x", lambda d: d.MET.pt)
    assert sel.num == 6
    assert sel.cutnames == ["BeforeCuts"]
    assert "x" not in ak.fields(sel.data)
    assert len(sel.systematics) == 6


def test_pickle(events):
    sel = Selector(events, weight=np.ones(6), rng_seed=1)
    sel.add_cut("a", np.array([True] * 3 + [False] * 3))
    sel2 = pickle.loads(pickle.dumps(sel))
    assert sel2.num == 3
    assert sel2.cutnames == sel.cutnames


def test_rng_seed(events):
    a = Selector(events, rng_seed=(1, 2)).rng.random(3)
    b = Selector(events, rng_seed=(1, 2)).rng.random(3)
    np.testing.assert_equal(a, b)
