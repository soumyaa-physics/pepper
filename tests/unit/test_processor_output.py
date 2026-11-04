import json
import types

import awkward as ak
import h5py
import hist
import numpy as np
import pytest
import uproot

import pepper
from pepper.hist_collection import HistCollection
from pepper.processor import EventOutputFile, Processor


class Proc(Processor):
    def process_selection(self, selector, dsname, is_mc, filler):
        pass


def make_proc(**config):
    p = Proc.__new__(Proc)
    conf = {"mc_lumifactors": {"tt": 1.}, "mc_datasets": {"tt": []},
            "dataset_for_systematics": {}}
    conf.update(config)
    p.config = conf
    p.eventdir = None
    return p


def cf_hist(value, cats=None):
    axes = [hist.axis.Integer(0, 1)]
    for name, vals in (cats or {}).items():
        axes.append(hist.axis.StrCategory(vals, name=name))
    h = hist.Hist(*axes, storage="Weight")
    if cats:
        for i, c in enumerate(next(iter(cats.values()))):
            h.fill(0, **{next(iter(cats)): c}, weight=value * (i + 1))
    else:
        h.fill(0, weight=value)
    return h


def h1(ds, val, sys=False):
    axes = [hist.axis.StrCategory([], name="dataset", growth=True)]
    if sys:
        axes.append(hist.axis.StrCategory([], name="sys", growth=True))
    axes.append(hist.axis.Regular(2, 0, 2, name="x"))
    h = hist.Hist(*axes, storage="Weight")
    kw = {"sys": "nominal"} if sys else {}
    h.fill(dataset=ds, x=[0.5], weight=val, **kw)
    if sys:
        h.fill(dataset=ds, sys="s_up", x=[0.5], weight=2 * val)
    return h


@pytest.fixture
def output():
    return {
        "cutflows": {
            "tt": {"BeforeCuts": cf_hist(10.), "A": cf_hist(6.),
                   "B": cf_hist(2., {"ch": ["ee", "mm"]})},
            "data": {"BeforeCuts": cf_hist(5.), "B": cf_hist(1.)},
        },
        "hists": {
            "tt": {("A", "h"): h1("tt", 6.), ("B", "h"): h1("tt", 2.)},
            "data": {("B", "h"): h1("data", 1.)},
        },
        "gen_sumws": {"tt": {"gen_sumw": 20.}},
    }


def test_get_cuts(output):
    assert Processor._get_cuts(output) == ["BeforeCuts", "A", "B"]
    output["cutflows"]["x"] = {"B": cf_hist(1.), "A": cf_hist(1.)}
    with pytest.raises(ValueError):
        Processor._get_cuts(output)


def test_prepare_cutflows(output):
    cf = Processor._prepare_cutflows(output)
    assert cf["tt"]["all"] == {"BeforeCuts": 10., "A": 6., "B": 6.}
    assert cf["tt"]["ee"]["B"] == 2. and cf["tt"]["mm"]["B"] == 4.
    assert cf["data"]["all"]["B"] == 1.


@pytest.mark.parametrize("fmt", ["hist", "root"])
def test_save_output(tmp_path, output, fmt):
    p = make_proc(histogram_format=fmt)
    p.save_output(output, str(tmp_path))
    with open(tmp_path / "cutflows.json") as f:
        assert json.load(f)["tt"]["all"]["A"] == 6.
    with open(tmp_path / "hists/hists.json") as f:
        coll = HistCollection.from_json(f)
    assert coll.userdata["cuts"] == ["BeforeCuts", "A", "B"]
    ext = "coffea" if fmt == "hist" else "root"
    assert coll[("B", "h")] == f"Cut_002_B_h.{ext}"
    h = coll.load(("B", "h"))
    assert set(h.axes["dataset"]) == {"tt", "data"}
    assert h.sum().value == 3.
    assert not (tmp_path / "gen_sumws.json").exists()


def test_save_output_gen_sumws_with_eventdir(tmp_path, output):
    p = make_proc()
    p.eventdir = str(tmp_path / "ev")
    p.save_output(output, str(tmp_path))
    with open(tmp_path / "gen_sumws.json") as f:
        assert json.load(f) == {"tt": {"gen_sumw": 20.}}


@pytest.mark.xfail(strict=True, reason="Bug: 'coffea' is accepted by "
                   "save_output but rejected in _save_histograms_inner")
def test_save_output_coffea_format(tmp_path, output):
    make_proc(histogram_format="coffea").save_output(output, str(tmp_path))


def test_apply_lumifactors_post(output):
    p = make_proc(mc_lumifactors=False, crosssections={"tt": 5.},
                  luminosity=8.)
    output["hists"]["tt"] = {("A", "h"): h1("tt", 6., sys=True),
                             ("B", "h"): h1("tt", 2., sys=True)}
    output["gen_sumws"]["tt"]["s_up"] = 40.
    out = p.apply_lumifactors_post(output)
    factor = 5. * 8. / 20.
    assert out["cutflows"]["tt"]["A"].sum().value == pytest.approx(6 * factor)
    assert out["cutflows"]["data"]["B"].sum().value == 1.
    h = out["hists"]["tt"][("A", "h")]
    assert h["tt", "nominal", :].sum().value == pytest.approx(6 * factor)
    # Normalised systematic: scaled by gen_sumw / sumw_sys
    assert h["tt", "s_up", :].sum().value == \
        pytest.approx(12 * factor * 20. / 40.)


def test_get_identifier():
    data = types.SimpleNamespace(metadata={
        "filename": "f.root", "entrystart": 0, "entrystop": 10})
    assert Processor.get_identifier(data) == ("f.root", 0, 10)


def test_rng_seed_file(tmp_path):
    p = make_proc(rng_seed_file=str(tmp_path / "seed.txt"))
    seed = p._load_rng_seed()
    assert int((tmp_path / "seed.txt").read_text()) == seed
    assert p._load_rng_seed() == seed
    (tmp_path / "seed.txt").write_text("nope")
    with pytest.raises(pepper.config.ConfigError):
        p._load_rng_seed()


def test_hists_to_do(hist_config):
    from pepper.hist_defns import HistDefinition
    conf = {"hists": {"a": HistDefinition(hist_config()),
                      "b": HistDefinition(hist_config())},
            "hists_to_do": ["b", "missing"]}
    assert list(Processor._get_hists_from_config(
        conf, "hists", "hists_to_do")) == ["b"]
    assert Processor._get_hists_from_config({}, "hists", "x") == {}


# --- Event output ---

@pytest.mark.parametrize("filetype", ["root", "hdf5"])
@pytest.mark.parametrize("use_temp", [False, True])
def test_event_output_file(tmp_path, filetype, use_temp):
    out = EventOutputFile(str(tmp_path), "ds/x", ("f", 0, 1), filetype,
                          use_temp)
    with out as f:
        if filetype == "root":
            f["Events"] = {"a": np.arange(3)}
        else:
            f["a"] = np.arange(3)
    files = list((tmp_path / "ds_x").iterdir())
    assert len(files) == 1
    assert files[0].suffix == (".root" if filetype == "root" else ".h5")
    if filetype == "root":
        with uproot.open(files[0]) as f:
            assert f["Events"]["a"].array().tolist() == [0, 1, 2]
    else:
        with h5py.File(files[0]) as f:
            assert list(f["a"]) == [0, 1, 2]
    if use_temp:
        assert not out.actual_filepath.startswith(str(tmp_path))


def test_event_output_two_files(tmp_path):
    for _ in range(2):
        with EventOutputFile(str(tmp_path), "ds", ("f", 0, 1), "root",
                             False) as f:
            f["x"] = "y"
    assert len(list((tmp_path / "ds").iterdir())) == 2


def test_event_output_bad_type(tmp_path):
    with pytest.raises(ValueError):
        EventOutputFile(str(tmp_path), "ds", ("f",), "csv", False)


def test_separate_masks_for_root():
    arrays = {
        "plain": ak.Array([1, 2, 3]),
        "masked": ak.Array([1, None, 3]),
        "jagged_masked": ak.Array([[1], None, [2, 3]]),
        "rec_masked": ak.Array([{"a": 1}, None, {"a": 3}]),
        "fake_option": ak.mask(ak.Array([1, 2, 3]), [True] * 3),
        "scalar": "x",
    }
    res = Processor._separate_masks_for_root(arrays)
    assert res["scalar"] == "x"
    assert "maskplain" not in res and "maskfake_option" not in res
    assert res["masked"].tolist() == [1, 0, 3]
    assert res["maskmasked"].tolist() == [True, False, True]
    assert res["jagged_masked"].tolist() == [[1], [], [2, 3]]
    assert res["rec_masked"].a.tolist() == [1, 0, 3]
    with pytest.raises(ValueError):
        Processor._separate_masks_for_root(
            {"x": ak.Array([[[1]]])})
