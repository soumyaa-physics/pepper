
import awkward as ak
import hist
import numpy as np
import pytest

from pepper.hist_defns import (
    DataPicker, HistDefinition, HistDefinitionError, HistFillError)
from conftest import TESTREPO


def fill(hdef, data, cats=None, weight=None, dsname="ds"):
    return hdef(data, cats or {}, dsname, True, weight)


# --- DataPicker ---

def test_picker_attribute(events):
    assert DataPicker(["MET", "pt"])(events).tolist() == \
        events.MET.pt.tolist()


def test_picker_missing_returns_none(events):
    assert DataPicker(["Electron", "pt"])(events) is None
    assert DataPicker([{"attribute": "nope"}])(events) is None


@pytest.mark.xfail(strict=True, reason="Bug: 'key' only catches KeyError, "
                   "awkward raises FieldNotFoundError")
def test_picker_missing_key_returns_none(events):
    assert DataPicker([{"key": "nope"}])(events) is None


def test_picker_leading(events):
    res = DataPicker(["Muon", "pt", {"leading": 2}])(events)
    assert res.tolist() == [[20.], [], [], [40.], [], [35.]]
    res = DataPicker(["Muon", "pt", {"leading": [1, 3]}])(events)
    assert ak.num(res).tolist() == [2, 1, 0, 2, 1, 2]
    with pytest.raises(HistDefinitionError):
        DataPicker(["Muon", "pt", {"leading": [3, 1]}])(events)


def test_picker_functions(events):
    assert DataPicker(["Muon", {"function": "num"}])(events).tolist() == \
        [2, 1, 0, 3, 1, 2]
    s = DataPicker(["Jet", "pt", {"function": "sum"}])(events)
    assert s.tolist() == [140., 80., 50., 0., 280., 45.]
    sq = DataPicker([{"function": "sqrt", "args": [["MET", "pt"]]}])(events)
    np.testing.assert_allclose(sq, np.sqrt(events.MET.pt))
    eq = DataPicker([{"function": "equal",
                      "args": [["MET", "pt"], 30.]}])(events)
    assert eq.tolist() == [False, False, True, False, False, False]
    # Args that cannot be picked propagate None
    assert DataPicker([{"function": "sqrt",
                        "args": [["Nope"]]}])(events) is None


def test_picker_leaddiff_concatenate(events):
    ld = DataPicker(["Muon", "pt", {"function": "leaddiff"}])(events)
    assert ld.tolist() == [[30.], [], [], [30.], [], [25.]]
    c = DataPicker([{"function": "concatenate",
                     "args": [["Muon", "pt"], ["Jet", "pt"]],
                     "kwargs": {"axis": 1}}])(events)
    assert ak.num(c).tolist() == [4, 2, 2, 3, 4, 3]


def test_picker_list_of_attrs(events):
    res = DataPicker(["Muon", ["pt", "eta"]])(events)
    assert set(ak.fields(res)) == {"pt", "eta"}


def test_picker_errors(events):
    with pytest.raises(HistDefinitionError):
        DataPicker("MET")(events)
    with pytest.raises(HistDefinitionError):
        DataPicker(["MET", 1])(events)
    with pytest.raises(HistDefinitionError):
        DataPicker(["MET", {"function": "nofunc"}])(events)


def test_picker_name():
    assert DataPicker(["Muon", "pt", {"leading": 1}]).name == \
        "Muon_pt_leading1"
    assert DataPicker(["Jet", {"function": "num"}]).name == "Jet_num"


# --- HistDefinition ---

def test_axes(hist_config):
    conf = hist_config(bins=[
        {"name": "a", "label": "A", "n_or_arr": 5, "lo": 0, "hi": 10,
         "unit": "GeV"},
        {"name": "b", "label": "B", "n_or_arr": [0, 1, 5]},
        {"name": "c", "label": "C", "n_or_arr": 3, "lo": 0, "hi": 3,
         "type": "int"},
    ], cats=[{"name": "cat", "label": "Cat"}],
        fill={"a": ["MET", "pt"], "b": ["MET", "pt"], "c": ["MET", "pt"],
              "cat": {"x": ["is_ee"]}})
    hdef = HistDefinition(conf)
    h = hdef.create_hist({"channel": ["is_ee"]}, has_systematic=True)
    assert h.axes.name == ("dataset", "sys", "a", "b", "c", "cat",
                           "channel")
    assert isinstance(h.axes["a"], hist.axis.Regular)
    assert isinstance(h.axes["b"], hist.axis.Variable)
    assert isinstance(h.axes["c"], hist.axis.Integer)
    assert h.axes["a"].label == "A (GeV)"


def test_label(hist_config):
    assert HistDefinition(hist_config()).label == "Events / $10$ units"
    assert HistDefinition(hist_config(label="foo")).label == "foo"
    var = hist_config(bins=[{"name": "pt", "n_or_arr": [0, 1, 5]}])
    assert HistDefinition(var).label == "Events / bin"


def test_definition_errors(hist_config):
    with pytest.raises(HistDefinitionError):
        HistDefinition(hist_config(fill={}))
    with pytest.raises(HistDefinitionError):
        HistDefinition(hist_config(fill={"pt": ["x"], "nope": ["x"]}))
    with pytest.raises(HistDefinitionError):
        HistDefinition(hist_config(
            bins=[{"name": "pt", "n_or_arr": 1, "lo": 0, "hi": 1,
                   "type": "bad"}]))
    with pytest.raises(HistDefinitionError):
        HistDefinition(hist_config(weight=["a"], weight_factor=["b"]))


def test_fill_unweighted(events, hist_config):
    h = fill(HistDefinition(hist_config()), events)
    np.testing.assert_allclose(h["ds", :].values(), [0, 1, 1, 1, 1, 1, 1,
                                                     0, 0, 0])


def test_fill_jagged_with_flat_weight(events, hist_config):
    conf = hist_config(fill={"pt": ["Muon", "pt"]})
    w = np.arange(6) + 1.
    h = fill(HistDefinition(conf), events, weight=w)
    # Each muon gets the event weight
    exp = sum(len(m) * wi for m, wi in zip(events.Muon.pt.tolist(), w))
    assert h.sum(flow=True).value == pytest.approx(exp)


def test_fill_mismatched_jagged_dropped(events, hist_config):
    conf = hist_config(
        bins=[{"name": "a", "n_or_arr": 10, "lo": 0, "hi": 300},
              {"name": "b", "n_or_arr": 10, "lo": 0, "hi": 300}],
        fill={"a": ["Muon", "pt"], "b": ["Jet", "pt"]})
    h = fill(HistDefinition(conf), events)
    # Only events with same number of muons and jets: 0, 1
    assert h.sum(flow=True).value == 2 + 1


def test_fill_missing_raises(events, hist_config):
    conf = hist_config(fill={"pt": ["Electron", "pt"]})
    with pytest.raises(HistFillError):
        fill(HistDefinition(conf), events)


def test_fill_masked_events(events, hist_config):
    data = ak.mask(events, np.array([True, False] * 3))
    h = fill(HistDefinition(hist_config()), data)
    assert h.sum(flow=True).value == 3


def test_fill_systematics(events, hist_config):
    w = {"nominal": np.ones(6), "sf_up": np.full(6, 2.)}
    h = fill(HistDefinition(hist_config()), events, weight=w)
    assert set(h.axes["sys"]) == {"nominal", "sf_up"}
    assert h["ds", "sf_up", :].sum().value == 12
    # do_systs off: only nominal, no sys axis
    h = fill(HistDefinition(hist_config(do_systs=False)), events, weight=w)
    assert "sys" not in h.axes.name
    assert h.sum().value == 6
    with pytest.raises(HistFillError):
        fill(HistDefinition(hist_config(do_systs=False)), events,
             weight={"sf_up": np.ones(6)})


def test_fill_weight_options(events, hist_config):
    h = fill(HistDefinition(hist_config(weight=["MET", "pt"])), events,
             weight=np.ones(6))
    assert h.sum(flow=True).value == pytest.approx(210)
    h = fill(HistDefinition(hist_config(weight_factor=["MET", "pt"])),
             events, weight=np.full(6, 2.))
    assert h.sum(flow=True).value == pytest.approx(420)


def test_fill_selector_categories(events, hist_config):
    cats = {"channel": ["is_ee", "is_mm", "is_em"]}
    h = fill(HistDefinition(hist_config()), events, cats=cats)
    assert h["ds", :, "is_mm"].sum(flow=True).value == 3
    assert h["ds", :, "is_ee"].sum(flow=True).value == 2
    h = fill(HistDefinition(hist_config(selector_cats=False)), events,
             cats=cats)
    assert "channel" not in h.axes.name
    h = fill(HistDefinition(hist_config(
        only_for_cats={"channel": "is_ee"})), events, cats=cats)
    assert list(h.axes["channel"]) == ["is_ee"]
    with pytest.raises(HistFillError):
        fill(HistDefinition(hist_config(only_for_cats={"other": "x"})),
             events, cats=cats)


def test_fill_dataset_category(events, hist_config):
    cats = {"dataset": ["is_ee", "is_mm", "is_em"]}
    h = fill(HistDefinition(hist_config()), events, cats=cats)
    assert set(h.axes["dataset"]) == {"is_ee", "is_mm", "is_em"}
    assert h.sum(flow=True).value == 6


def test_fill_config_categories_default_key(events, hist_config):
    conf = hist_config(
        cats=[{"name": "flav", "label": "f", "default_key": "all"}],
        fill={"pt": ["MET", "pt"], "flav": {"b": ["nope"]}})
    h = fill(HistDefinition(conf), events)
    assert list(h.axes["flav"]) == ["all"]
    conf = hist_config(
        cats=[{"name": "flav", "label": "f"}],
        fill={"pt": ["MET", "pt"], "flav": {"b": ["nope"]}})
    with pytest.raises(HistFillError):
        fill(HistDefinition(conf), events)


def test_histograms_json_parses():
    """All definitions shipped with the test configs must parse"""
    path = TESTREPO / "configs" / "histograms.json"
    if not path.exists():
        pytest.skip("test submodule not checked out")
    import hjson
    with open(path) as f:
        defs = hjson.load(f)
    for name, conf in defs.items():
        HistDefinition(conf)
