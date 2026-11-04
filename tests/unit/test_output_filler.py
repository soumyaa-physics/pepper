import awkward as ak
import hist
import numpy as np
import pytest

from pepper import Selector
from pepper.hist_defns import HistDefinition
from pepper.output_filler import AddableDict, OutputFiller


def make_filler(hist_config, **kwargs):
    defs = {"met": HistDefinition(hist_config())}
    args = dict(hist_dict=defs, is_mc=True, dsname="ds",
                dsname_in_hist="ds", sys_enabled=False)
    args.update(kwargs)
    return OutputFiller(**args)


def run(events, filler, weight=None):
    sel = Selector(events, weight=weight, on_update=filler.get_callbacks())
    sel.add_cut("met", lambda d: d.MET.pt > 15)
    sel.add_cut("mu", lambda d: ak.num(d.Muon) > 0)
    return sel


def cutflow_val(filler, cut):
    return filler.output["cutflows"]["ds"][cut].values().sum()


def test_cutflow(events, hist_config):
    filler = make_filler(hist_config)
    run(events, filler, weight=np.full(6, 2.))
    assert cutflow_val(filler, "BeforeCuts") == 12
    assert cutflow_val(filler, "met") == 10
    assert cutflow_val(filler, "mu") == 8


def test_cutflow_unweighted(events, hist_config):
    filler = make_filler(hist_config)
    run(events, filler)
    assert cutflow_val(filler, "mu") == 4


def test_hists_per_cut_match_cutflow(events, hist_config):
    filler = make_filler(hist_config)
    run(events, filler, weight=np.arange(6) + 1.)
    hists = filler.output["hists"]["ds"]
    assert set(hists) == {("BeforeCuts", "met"), ("met", "met"),
                          ("mu", "met")}
    for cut in ["BeforeCuts", "met", "mu"]:
        assert hists[(cut, "met")].sum(flow=True).value == \
            pytest.approx(cutflow_val(filler, cut))


def test_cuts_to_histogram(events, hist_config):
    filler = make_filler(hist_config, cuts_to_histogram=["mu"])
    run(events, filler)
    assert set(filler.output["hists"]["ds"]) == {("mu", "met")}
    # Cutflows still filled for all cuts
    assert set(filler.output["cutflows"]["ds"]) == \
        {"BeforeCuts", "met", "mu"}


def test_step_requirement(events, hist_config):
    defs = {"x": HistDefinition(hist_config(
        fill={"pt": ["x"]}, step_requirement="column:x"))}
    filler = OutputFiller(defs, True, "ds", "ds", False)
    sel = Selector(events, on_update=filler.get_callbacks())
    sel.add_cut("a", np.full(6, True))
    assert filler.output["hists"]["ds"] == {}
    # Setting the column fills for the last cut
    sel.set_column("x", lambda d: d.MET.pt)
    sel.add_cut("b", np.full(6, True))
    assert set(filler.output["hists"]["ds"]) == {("a", "x"), ("b", "x")}


def test_blind_data(events, hist_config):
    defs = {"met": HistDefinition(hist_config(blind_data=True))}
    filler = OutputFiller(defs, False, "data", "data", False)
    Selector(events, on_update=filler.get_callbacks())
    assert filler.output["hists"]["data"] == {}
    filler = OutputFiller(defs, True, "mc", "mc", False)
    Selector(events, on_update=filler.get_callbacks())
    assert len(filler.output["hists"]["mc"]) == 1


def test_dsname_in_hist(events, hist_config):
    filler = make_filler(hist_config, dsname_in_hist="other")
    run(events, filler)
    h = filler.output["hists"]["ds"][("mu", "met")]
    assert list(h.axes["dataset"]) == ["other"]


def test_missing_fill_ignored(events, hist_config):
    defs = {"bad": HistDefinition(hist_config(fill={"pt": ["Nope"]}))}
    filler = OutputFiller(defs, True, "ds", "ds", False)
    run(events, filler)
    assert filler.output["hists"]["ds"] == {}


def test_gen_sumws(hist_config):
    filler = make_filler(hist_config)
    filler.fill_gen_sumws(np.array([1., 2., -0.5]))
    assert filler.output["gen_sumws"]["ds"]["gen_sumw"] == 2.5


def test_addable_dict():
    a = AddableDict(x=1, y=2)
    b = AddableDict(y=3, z=4)
    c = a + b
    assert c == {"x": 1, "y": 5, "z": 4}
    assert isinstance(c, AddableDict)
    with pytest.raises(ValueError):
        a + 1


def test_hists_accumulate(events, hist_config):
    """Outputs of two chunks can be summed like coffea does"""
    f1, f2 = make_filler(hist_config), make_filler(hist_config)
    run(events, f1)
    run(events, f2)
    total = AddableDict(f1.output["hists"]["ds"]) + \
        AddableDict(f2.output["hists"]["ds"])
    assert total[("mu", "met")].sum(flow=True).value == 8
    assert isinstance(total[("mu", "met")], hist.Hist)
