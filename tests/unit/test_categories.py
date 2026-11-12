import awkward as ak
import numpy as np
import pytest

from pepper import Selector
from pepper.hist_defns import HistDefinition
from pepper.output_filler import OutputFiller

CHANNELS = {"is_ee", "is_mm", "is_em"}


def setup(events, hist_config, **hkw):
    defs = {"met": HistDefinition(hist_config(**hkw))}
    filler = OutputFiller(defs, True, "ds", "ds", False)
    sel = Selector(events, on_update=filler.get_callbacks())
    return sel, filler


def test_set_cat_safe(events):
    sel = Selector(events)
    sel.set_cat("channel", CHANNELS)
    assert sel.cats == {"channel": CHANNELS}
    with pytest.raises(ValueError):
        sel.set_cat("bad", {"is_ee", "is_mm"})
    sel2 = Selector(events)
    sel2.set_cat("loose", {"is_ee", "is_mm"}, safe=False)
    with pytest.raises(ValueError):
        # Category names must be unique across categorizations
        sel2.set_cat("other", {"is_ee", "is_em"}, safe=False)


def test_set_cat_none_filled(events):
    sel = Selector(events)
    sel.set_column("sub", ak.mask(events.is_ee, events.is_ee))
    sel.set_column("rest", ~events.is_ee)
    sel.set_cat("c", {"sub", "rest"})
    assert sel.data.sub.tolist() == events.is_ee.tolist()


def test_cutflow_per_category(events, hist_config):
    sel, filler = setup(events, hist_config)
    sel.set_cat("channel", CHANNELS)
    sel.add_cut("met", lambda d: d.MET.pt > 15)
    cf = filler.output["cutflows"]["ds"]["met"]
    assert cf.axes.name[1:] == ("channel",)
    assert cf[0, "is_ee"].value == 1
    assert cf[0, "is_mm"].value == 3
    assert cf[0, "is_em"].value == 1
    # Before categorisation: no category axis
    assert len(filler.output["cutflows"]["ds"]["BeforeCuts"].axes) == 1


def test_hists_split_by_category(events, hist_config):
    sel, filler = setup(events, hist_config)
    sel.set_cat("channel", CHANNELS)
    sel.add_cut("all", np.full(6, True))
    h = filler.output["hists"]["ds"][("all", "met")]
    assert set(h.axes["channel"]) == CHANNELS
    assert h["ds", :, "is_mm"].sum(flow=True).value == 3


def test_two_categorizations(events, hist_config):
    sel, filler = setup(events, hist_config)
    sel.set_cat("channel", CHANNELS)
    sel.set_column("hi", lambda d: d.MET.pt > 35)
    sel.set_column("lo", lambda d: d.MET.pt <= 35)
    sel.set_cat("met_region", {"hi", "lo"})
    sel.add_cut("all", np.full(6, True))
    cf = filler.output["cutflows"]["ds"]["all"]
    assert cf.axes.name[1:] == ("channel", "met_region")
    assert cf[0, "is_mm", "hi"].value == 2
    assert cf.values().sum() == 6


def test_dataset_category(events, hist_config):
    """'dataset' categorisation overrides the dataset axis"""
    sel, filler = setup(events, hist_config)
    sel.set_column("part_a", lambda d: d.MET.pt < 35)
    sel.set_column("part_b", lambda d: d.MET.pt >= 35)
    sel.set_cat("dataset", {"part_a", "part_b"})
    sel.add_cut("all", np.full(6, True))
    h = filler.output["hists"]["ds"][("all", "met")]
    assert set(h.axes["dataset"]) == {"part_a", "part_b"}
    assert h["part_a", :].sum(flow=True).value == 3


def test_category_restricted_cut(events, hist_config):
    sel, filler = setup(events, hist_config)
    sel.set_cat("channel", CHANNELS)
    # Z-window-like cut only for same flavour
    sel.add_cut("sf_only", lambda d: d.MET.pt > 45,
                categories={"channel": ["is_ee", "is_mm"]})
    cf = filler.output["cutflows"]["ds"]["sf_only"]
    assert cf[0, "is_em"].value == 1
    assert cf[0, "is_ee"].value == 0
    assert cf[0, "is_mm"].value == 2
