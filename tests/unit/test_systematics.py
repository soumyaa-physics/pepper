import numpy as np
import pytest

from pepper import Selector
from pepper.hist_defns import HistDefinition
from pepper.output_filler import OutputFiller


def make(events, hist_config, sys_enabled=True, **fkw):
    defs = {"met": HistDefinition(hist_config())}
    filler = OutputFiller(defs, True, "ds", "ds", sys_enabled, **fkw)
    sel = Selector(events, weight=np.ones(6),
                   on_update=filler.get_callbacks(), output_filler=filler)
    return sel, filler


def test_schemes(events):
    sel = Selector(events, weight=np.ones(6))
    sel.set_systematic("one", np.full(6, 2.))
    sel.set_systematic("ud", np.full(6, 1.1), np.full(6, 0.9))
    sel.set_systematic("num", *[np.full(6, i) for i in range(3)])
    sel.set_systematic("forced", np.ones(6), np.ones(6), scheme="numeric")
    assert set(sel.systematics.fields) == {
        "weight", "one", "ud_up", "ud_down", "num_0", "num_1", "num_2",
        "forced_0", "forced_1"}


def test_schemes_errors(events):
    sel = Selector(events, weight=np.ones(6))
    with pytest.raises(ValueError):
        sel.set_systematic("weight", np.ones(6))
    with pytest.raises(ValueError):
        sel.set_systematic("x", np.ones(6), scheme="bad")
    with pytest.raises(ValueError):
        sel.set_systematic("x", np.ones((6, 2)))


def test_systematics_follow_cuts(events):
    sel = Selector(events, weight=np.ones(6))
    sel.set_systematic("s", np.arange(6.))
    sel.add_cut("c", np.array([False, True, True, False, True, True]))
    np.testing.assert_allclose(sel.systematics.s, [1, 2, 4, 5])


def test_hists_have_sys_axis(events, hist_config):
    sel, filler = make(events, hist_config)
    sel.set_systematic("s", np.full(6, 1.5), np.full(6, 0.5))
    sel.add_cut("c", np.full(6, True))
    h = filler.output["hists"]["ds"][("c", "met")]
    assert set(h.axes["sys"]) == {"nominal", "s_up", "s_down"}
    assert h["ds", "s_up", :].sum(flow=True).value == pytest.approx(9)
    assert h["ds", "s_down", :].sum(flow=True).value == pytest.approx(3)
    assert h["ds", "nominal", :].sum(flow=True).value == pytest.approx(6)


def test_sys_disabled(events, hist_config):
    sel, filler = make(events, hist_config, sys_enabled=False)
    sel.set_systematic("s", np.full(6, 1.5))
    sel.add_cut("c", np.full(6, True))
    h = filler.output["hists"]["ds"][("c", "met")]
    assert "sys" not in h.axes.name


def test_systs_to_histogram(events, hist_config):
    sel, filler = make(events, hist_config, systs_to_histogram=["a", "n"])
    sel.set_systematic("a", np.ones(6), np.ones(6))
    sel.set_systematic("b", np.ones(6), np.ones(6))
    sel.set_systematic("n", np.ones(6), np.ones(6), np.ones(6))
    sel.add_cut("c", np.full(6, True))
    h = filler.output["hists"]["ds"][("c", "met")]
    assert set(h.axes["sys"]) == {"nominal", "a_up", "a_down", "n_0", "n_1",
                                  "n_2"}


def test_sys_overwrite(events, hist_config):
    """Variation runs (e.g. JES) fill only their sys bin, no cutflow"""
    sel, filler = make(events, hist_config, sys_overwrite="jes_up")
    sel.add_cut("c", np.full(6, True))
    h = filler.output["hists"]["ds"][("c", "met")]
    assert list(h.axes["sys"]) == ["jes_up"]
    assert filler.output["cutflows"]["ds"] == {}


def test_cut_sf_with_variations(events, hist_config):
    sel, filler = make(events, hist_config)
    sf = np.full(6, 0.9)
    sel.add_cut("sf", (sf, {"sf": (np.full(6, 1.1), np.full(6, 0.8))}))
    h = filler.output["hists"]["ds"][("sf", "met")]
    assert h["ds", "nominal", :].sum(flow=True).value == pytest.approx(5.4)
    assert h["ds", "sf_up", :].sum(flow=True).value == pytest.approx(5.94)


def test_norm_post(events, hist_config):
    sel, filler = make(events, hist_config)
    sel.set_systematic("ps", np.full(6, 2.), norm_post=True)
    assert filler.output["gen_sumws"]["ds"]["ps"] == pytest.approx(12)
    sel.add_cut("c", np.full(6, True))
    with pytest.raises(RuntimeError):
        sel.set_systematic("late", np.full(6, 2.), norm_post=True)


def test_norm_post_needs_filler(events):
    sel = Selector(events, weight=np.ones(6))
    with pytest.raises(UnboundLocalError):
        sel.set_systematic("ps", np.full(6, 2.), norm_post=True)
