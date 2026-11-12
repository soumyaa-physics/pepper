"""Tests for the scripts in scripts/. Most scripts have module-level argparse
and are therefore run as subprocesses."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import awkward as ak
import hist
import numpy as np
import pytest
import uproot

import pepper
from pepper import Selector
from pepper.hdffile import HDF5File
from pepper.hist_collection import HistCollection
from pepper.hist_defns import HistDefinition
from pepper.output_filler import OutputFiller
from pepper.processor import Processor
from conftest import REPO, SCRIPTS, TESTREPO

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except ImportError:  # e.g. libjpeg missing in the CI image
    HAS_MPL = False
needs_mpl = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")


def run_script(script, *args, cwd=None, check=True, stdin=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)], cwd=cwd,
        env=env, capture_output=True, text=True, timeout=900, input=stdin)
    if check and res.returncode != 0:
        raise AssertionError(
            f"{script} failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    return res


def import_script(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("/", "_")[:-3], SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Synthetic pepper output ---

HISTS = {
    "met": {"bins": [{"name": "met", "label": "MET", "n_or_arr": 10,
                      "lo": 0, "hi": 200, "unit": "GeV"}],
            "fill": {"met": ["MET", "pt"]}},
    "jet_pt": {"bins": [{"name": "pt", "label": "Jet pt", "n_or_arr": 10,
                         "lo": 0, "hi": 200}],
               "fill": {"pt": ["Jet", "pt"]}},
    "met_njet": {"bins": [{"name": "met", "label": "MET", "n_or_arr": 5,
                           "lo": 0, "hi": 200},
                          {"name": "njet", "label": "N", "n_or_arr": 4,
                           "lo": 0, "hi": 4, "type": "int"}],
                 "fill": {"met": ["MET", "pt"],
                          "njet": ["Jet", {"function": "num"}]}},
}
DATASETS = {"MuonEG": False, "TT": True, "DY": True}


def synth_events(n, seed):
    rng = np.random.default_rng(seed)
    njet = rng.integers(0, 4, n)
    chn = rng.integers(0, 3, n)
    return ak.Array({
        "MET": ak.zip({"pt": rng.exponential(50, n)}),
        "Jet": ak.zip({"pt": ak.unflatten(
            rng.exponential(60, njet.sum()) + 20, njet)}),
        "is_ee": chn == 0, "is_em": chn == 1, "is_mm": chn == 2,
    })


def make_output(seed=0):
    out = {"hists": {}, "cutflows": {}, "gen_sumws": {}}
    for i, (ds, is_mc) in enumerate(DATASETS.items()):
        defs = {k: HistDefinition(json.loads(json.dumps(v)))
                for k, v in HISTS.items()}
        filler = OutputFiller(defs, is_mc, ds, ds, True)
        ev = synth_events(500, seed + i)
        sel = Selector(ev, weight=np.full(500, 0.5 if is_mc else 1.),
                       on_update=filler.get_callbacks())
        sel.set_cat("channel", {"is_ee", "is_em", "is_mm"})
        if is_mc:
            sel.set_systematic("toy", np.full(500, 1.1), np.full(500, 0.9))
        sel.add_cut("MET", lambda d: d.MET.pt > 20)
        sel.add_cut("HasJets", lambda d: ak.num(d.Jet) > 0)
        for key in out:
            out[key].update(filler.output[key])
    return out


class Proc(Processor):
    def process_selection(self, selector, dsname, is_mc, filler):
        pass


def save_output(dest, fmt="hist", seed=0):
    p = Proc.__new__(Proc)
    p.config = {"mc_lumifactors": True, "histogram_format": fmt}
    p.eventdir = None
    os.makedirs(dest, exist_ok=True)
    p.save_output(make_output(seed), str(dest))
    return dest


@pytest.fixture(scope="module")
def pepper_out(tmp_path_factory):
    return save_output(tmp_path_factory.mktemp("pepper_out"))


@pytest.fixture
def plot_config(tmp_path):
    conf = {
        "backgrounds": {
            "TT": {"label": "tt", "color": "tab:red", "datasets": ["TT"]},
            "DY": {"label": "DY", "color": "tab:blue", "datasets": ["DY"]}},
        "data": ["MuonEG"], "signals": {}, "categories": {},
        "sum_categories": False,
        "rebin": {"met": {"n_or_arr": [0, 40, 200]}},
        "year": "2018", "luminosity": 59.7, "com_energy": 13,
    }
    path = tmp_path / "plot.json"
    path.write_text(json.dumps(conf))
    return path


# --- comparison scripts ---

def test_compare_json(tmp_path):
    a = {"x": 1.0, "y": [1, {"z": "s"}], "b": True}
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(a))
    run_script("comparison/compare_json.py", tmp_path / "a.json",
               tmp_path / "b.json")
    a["x"] = 1.1
    (tmp_path / "b.json").write_text(json.dumps(a))
    res = run_script("comparison/compare_json.py", tmp_path / "a.json",
                     tmp_path / "b.json", check=False)
    assert res.returncode == 1 and "root.x" in res.stdout
    run_script("comparison/compare_json.py", tmp_path / "a.json",
               tmp_path / "b.json", "--rtol", "0.2")


def test_compare_json_function():
    mod = import_script("comparison/compare_json.py")
    diffs = []
    mod.compare({"a": [1, 2], "b": 1, "c": True},
                {"a": [1], "b": "1", "c": 1, "d": 0}, "r", 0, 0, diffs)
    # True == 1 is not reported
    assert len(diffs) == 3


def test_compare_hists(tmp_path, pepper_out):
    js = pepper_out / "hists/hists.json"
    run_script("comparison/compare_hists.py", js, js)
    other = save_output(tmp_path / "other", seed=5)
    res = run_script("comparison/compare_hists.py", js,
                     other / "hists/hists.json", check=False)
    assert res.returncode != 0


def write_event_dir(base, values, ext):
    for ds in ("A", "B"):
        d = base / ds
        d.mkdir(parents=True)
        if ext == ".h5":
            with HDF5File(str(d / "f.h5"), "w") as f:
                f["events"] = ak.Array({"x": values, "j": [[1, 2]] * 3})
                f["dsname"] = ds
        else:
            with uproot.recreate(d / "f.root") as f:
                f.mktree("Events", {"x": "f8"})
                f["Events"].extend({"x": np.asarray(values)})


@pytest.mark.parametrize("ext", [".h5", ".root"])
def test_compare_events(tmp_path, ext):
    write_event_dir(tmp_path / "a", [1., 2., 3.], ext)
    write_event_dir(tmp_path / "b", [1., 2., 3.], ext)
    write_event_dir(tmp_path / "c", [1., 2., 4.], ext)
    run_script("comparison/compare_events.py", tmp_path / "a",
               tmp_path / "b")
    res = run_script("comparison/compare_events.py", tmp_path / "a",
                     tmp_path / "c", check=False)
    assert res.returncode == 1 and "mismatch" in res.stderr
    key = "events/x" if ext == ".h5" else "Events/x"
    run_script("comparison/compare_events.py", tmp_path / "a",
               tmp_path / "c", "--ignore", key)


def test_compare_ak_function():
    mod = import_script("comparison/compare_events.py")
    a = ak.Array({"x": [[1., 2.], []], "s": ["a", "b"]})
    assert mod.compare_ak(a, a, "", 1e-5, 0, set()) == []
    b = ak.Array({"x": [[1., 2.], [3.]], "s": ["a", "b"]})
    assert "jagged" in mod.compare_ak(a, b, "", 1e-5, 0, set())[0]


# --- histogram handling scripts ---

def load_hist(js, key):
    with open(js) as f:
        return HistCollection.from_json(f).load(key)


def test_merge_histograms(tmp_path, pepper_out):
    js = pepper_out / "hists/hists.json"
    shutil.copytree(pepper_out / "hists", tmp_path / "copy")
    run_script("merge_histograms.py", tmp_path / "merged", js,
               tmp_path / "copy/hists.json")
    merged = load_hist(tmp_path / "merged/hists.json", ("HasJets", "met"))
    orig = load_hist(js, ("HasJets", "met"))
    assert merged.sum(flow=True).value == \
        pytest.approx(2 * orig.sum(flow=True).value)
    with open(tmp_path / "merged/hists.json") as f:
        assert set(json.load(f)["userdata"]["cuts"]) == \
            {"BeforeCuts", "MET", "HasJets"}


def test_export_hists_from_state(tmp_path):
    state = {"accumulator": {"out": make_output()}, "items_done": [],
             "version": 4, "userdata": {}}
    pepper.misc.save(state, str(tmp_path / "state.coffea"))
    run_script("export_hists_from_state.py", tmp_path / "state.coffea",
               "root", tmp_path / "out", "-t", "2")
    h = load_hist(tmp_path / "out/hists.json", ("MET", "met"))
    assert set(h.axes["dataset"]) == set(DATASETS)


@pytest.mark.xfail(strict=True, reason="Bug: hist_split_strcat called "
                   "without cats_present")
def test_convert_root_hists_split_sys(tmp_path, pepper_out):
    run_script("convert_root_hists_split_sys.py",
               pepper_out / "hists/hists.json", tmp_path / "out",
               "--processes", "2")
    with open(tmp_path / "out/hists.json") as f:
        keys, files = json.load(f)
    assert ["HasJets", "met", "toy_up"] in keys
    idx = keys.index(["HasJets", "met", "nominal"])
    with uproot.open(tmp_path / "out" / files[idx]) as f:
        assert "TT_is_ee" in [k.split(";")[0] for k in f.keys()]


@pytest.mark.xfail(strict=True, reason="Bug: arithmetic on uproot TH1 "
                   "models is not supported")
def test_average_sf_hists(tmp_path):
    def write(path, val):
        h = hist.Hist(hist.axis.Regular(2, 0, 2, name="x"),
                      storage="Weight")
        h[...] = np.stack([np.full(2, val), np.zeros(2)], axis=-1)
        with uproot.recreate(path) as f:
            f["sf"] = h
    write(tmp_path / "a.root", 1.)
    write(tmp_path / "b.root", 4.)
    run_script("average_sf_hists.py", tmp_path / "out.root", "avg",
               "-i", 2, tmp_path / "a.root", "sf",
               "-i", 1, tmp_path / "b.root", "sf")
    with uproot.open(tmp_path / "out.root") as f:
        np.testing.assert_allclose(f["avg"].values(), 2.)


def eff_collection(path, name, axes, fill):
    h = hist.Hist(hist.axis.StrCategory(["TT"], name="dataset"),
                  hist.axis.StrCategory(["nominal", "s_up"], name="sys"),
                  *axes, storage="Weight")
    for sys_ in ("nominal", "s_up"):
        fill(h, sys_)
    coll = HistCollection(str(path), ["cut", "hist"])
    coll.save(("Cut", name), h, "h.coffea", "hist",
              HistCollection.get_cats_present(h))
    with open(path / "hists.json", "w") as f:
        coll.save_metadata_json(f)
    return path / "hists.json"


def test_generate_btag_efficiencies(tmp_path):
    def fill(h, sys_):
        for flav in (0, 4, 5):
            h.fill(dataset="TT", sys=sys_, flav=flav, pt=50., abseta=1.,
                   btagged="yes", weight=1. + (flav == 5) * 2)
            h.fill(dataset="TT", sys=sys_, flav=flav, pt=50., abseta=1.,
                   btagged="no", weight=1.)
    js = eff_collection(tmp_path, "btageff", [
        hist.axis.IntCategory([0, 4, 5], name="flav"),
        hist.axis.Variable([20, 100], name="pt"),
        hist.axis.Variable([0, 2.5], name="abseta"),
        hist.axis.StrCategory(["yes", "no"], name="btagged")], fill)
    run_script("generate_btag_efficiencies.py", js, tmp_path / "eff.root",
               "--cut", "Cut")
    with uproot.open(tmp_path / "eff.root") as f:
        assert {k.split(";")[0] for k in f.keys()} == {"central", "s_up"}
        np.testing.assert_allclose(f["central"].values().ravel(),
                                   [0.5, 0.5, 0.75])
    # Existing output requires confirmation
    res = run_script("generate_btag_efficiencies.py", js,
                     tmp_path / "eff.root", "--cut", "Cut", stdin="n\n",
                     check=False)
    assert res.returncode == 1


def test_generate_jet_puid_efficiencies(tmp_path):
    def fill(h, sys_):
        for gen in ("yes", "no"):
            h.fill(dataset="TT", sys=sys_, pt=30., eta=0., has_gen_jet=gen,
                   pass_pu_id="yes", weight=3. if gen == "yes" else 1.)
            h.fill(dataset="TT", sys=sys_, pt=30., eta=0., has_gen_jet=gen,
                   pass_pu_id="no", weight=1.)
    js = eff_collection(tmp_path, "jet_pu_id_eff", [
        hist.axis.Variable([20, 50], name="pt"),
        hist.axis.Variable([-5, 5], name="eta"),
        hist.axis.StrCategory(["yes", "no"], name="has_gen_jet"),
        hist.axis.StrCategory(["yes", "no"], name="pass_pu_id")], fill)
    run_script("generate_jet_puid_efficiencies.py", js,
               tmp_path / "eff.root", "--cut", "Cut")
    with uproot.open(tmp_path / "eff.root") as f:
        np.testing.assert_allclose(f["eff"].values(), [[0.75]])
        np.testing.assert_allclose(f["mis"].values(), [[0.5]])


def dy_collection(path, data_scale):
    axes = [hist.axis.StrCategory(["MuonEG", "DYJets"], name="dataset"),
            hist.axis.StrCategory(["is_ee", "is_em", "is_mm"],
                                  name="channel"),
            hist.axis.StrCategory(["in_Z_win", "out_Z_win"],
                                  name="in_Z_window"),
            hist.axis.StrCategory(["0b", "1b"], name="btags"),
            hist.axis.Variable([20, 50, 200], name="pt")]
    h = hist.Hist(*axes, storage="Weight")
    rng = np.random.default_rng(1)
    for ch in ("is_ee", "is_mm"):
        for z in ("in_Z_win", "out_Z_win"):
            for b in ("0b", "1b"):
                n = rng.uniform(50, 100, 2)
                for ds, scale in (("DYJets", 1.), ("MuonEG", data_scale)):
                    h.fill(dataset=ds, channel=ch, in_Z_window=z, btags=b,
                           pt=[30., 100.], weight=n * scale)
    coll = HistCollection(str(path), ["cut", "hist"])
    coll.save(("HasJets", "Leptonpt"), h, "h.coffea", "hist",
              HistCollection.get_cats_present(h))
    with open(path / "hists.json", "w") as f:
        coll.save_metadata_json(f)
    conf = path / "conf.json"
    conf.write_text(json.dumps({"exp_datasets": {"MuonEG": []},
                                "mc_datasets": {"DYJets": []}}))
    return conf, path / "hists.json"


@pytest.mark.xfail(strict=True, reason="Bug: Hist.integrate(axis, 'cat') "
                   "integrates from 'cat' to the end of the axis; get_subs "
                   "must pass lists")
@pytest.mark.parametrize("integrate", [False, True])
def test_ttbarll_dy_sf_calculate(tmp_path, integrate):
    """Data = 2 x DY in ee/mm and no emu: all SFs are 2"""
    conf, js = dy_collection(tmp_path, 2.)
    args = ["-i"] if integrate else []
    run_script("ttbarll_dy_sf_calculate.py", conf, js, tmp_path / "sf.json",
               *args)
    with open(tmp_path / "sf.json") as f:
        res = json.load(f)
    np.testing.assert_allclose(np.asarray(res["factors"], dtype=float), 2.)
    assert np.all(np.asarray(res["factors_up"]) >
                  np.asarray(res["factors"]))
    # The output can be read by the ttbarll config
    sf = pepper.scale_factors.ScaleFactors(
        bins=res["bins"], factors=np.array(res["factors"]),
        factors_up=np.array(res["factors_up"]),
        factors_down=np.array(res["factors_down"]))
    assert sf is not None


# --- plotting ---

@needs_mpl
def test_make_plots(tmp_path, pepper_out, plot_config):
    run_script("make_plots.py", plot_config, pepper_out / "hists/hists.json",
               "-o", tmp_path, "--ext", "png", "--cut", "HasJets", "--log")
    pngs = list(tmp_path.rglob("*.png"))
    names = {p.name for p in pngs}
    assert "met_HasJets.png" in names or any(
        n.startswith("met_HasJets") for n in names)
    assert any(n.startswith("jet_pt_HasJets") for n in names)
    # 2d hists are skipped
    assert not any(n.startswith("met_njet") for n in names)


@needs_mpl
def test_make_plots_single_hist(tmp_path, pepper_out, plot_config):
    hfile = pepper_out / "hists/Cut_002_HasJets_met.coffea"
    run_script("make_plots.py", plot_config, hfile, "-o", tmp_path,
               "--ext", "png")
    assert list(tmp_path.rglob("met_HasJets*.png"))


@needs_mpl
def test_make_plots_2d(tmp_path, pepper_out, plot_config):
    run_script("make_plots_2D.py", plot_config,
               pepper_out / "hists/hists.json", "-o", tmp_path, "--ext",
               "png", "--cut", "HasJets")
    assert any(p.name.startswith("met_njet")
               for p in tmp_path.rglob("*.png"))


@needs_mpl
def test_plot_cutflow(tmp_path, pepper_out):
    run_script("plot_cutflow.py", pepper_out / "cutflows.json", "-o",
               tmp_path, "--ext", "png")
    assert (tmp_path / "cutflow_ratio.png").exists()
    assert list((tmp_path / "efficiencies").glob("*.png"))


# --- event output scripts ---

def write_hdf5_events(path, n, offset=0, systematics=False):
    with HDF5File(str(path), "w") as f:
        f["events"] = ak.Array({
            "met": np.arange(n, dtype=float) + offset,
            "jet": ak.Array([[1., 2.]] * n)})
        f["cutflags"] = ak.Array({"a": np.full(n, True),
                                  "b": np.arange(n) % 2 == 0})
        if systematics:
            f["systematics"] = ak.Array({"weight": np.ones(n),
                                         "s_up": np.full(n, 1.1)})
        else:
            f["weight"] = np.ones(n)
        f["identifier"] = ("file", offset, offset + n)


@pytest.mark.xfail(strict=True, reason="uproot>=5.7 writes RNTuple for "
                   "file[name] = dict; script should use mktree")
def test_hdf5_to_ttree_writes_ttree(tmp_path):
    mod = import_script("hdf5_to_ttree.py")
    d = tmp_path / "TT"
    d.mkdir()
    write_hdf5_events(d / "a.h5", 4)
    mod.process(str(d), {}, False, ignore=[])
    with uproot.open(tmp_path / "TT.root") as f:
        assert isinstance(f["Events"], uproot.TTree)


@pytest.mark.parametrize("mergesys", [False, True])
def test_hdf5_to_ttree(tmp_path, mergesys):
    mod = import_script("hdf5_to_ttree.py")
    d = tmp_path / "TT"
    d.mkdir()
    write_hdf5_events(d / "a.h5", 4, systematics=True)
    write_hdf5_events(d / "b.h5", 6, offset=10, systematics=True)
    mod.process(str(d), {}, mergesys, ignore=["jet"])
    with uproot.open(tmp_path / "TT.root") as f:
        ev = f["Events"].arrays()
        assert len(ev) == 10
        assert "jet" not in ev.fields
        assert ak.sum(ev.cuts) == 5
        if mergesys:
            assert "Systematics" not in f
        else:
            assert len(f["Systematics"]["s_up"].array()) == 10


@pytest.mark.xfail(strict=True, reason="Bug: systree reset to {} instead "
                   "of defaultdict(list) after size flush")
def test_hdf5_to_ttree_flush(tmp_path):
    mod = import_script("hdf5_to_ttree.py")
    d = tmp_path / "TT"
    d.mkdir()
    write_hdf5_events(d / "a.h5", 4)
    write_hdf5_events(d / "b.h5", 6, offset=10)
    mod.process(str(d), {}, False, ignore=[], max_size_bytes=1)
    with uproot.open(tmp_path / "TT.root") as f:
        assert f["Events"].num_entries == 10


@pytest.mark.parametrize("delete", [False, True])
def test_delete_duplicate_outputs(tmp_path, delete):
    mod = import_script("delete_duplicate_outputs.py")
    write_hdf5_events(tmp_path / "a.h5", 2)
    shutil.copy(tmp_path / "a.h5", tmp_path / "dup.h5")
    write_hdf5_events(tmp_path / "b.h5", 2, offset=5)
    with uproot.recreate(tmp_path / "r.root") as f:
        f["identifier"] = "x"
    (tmp_path / "broken.root").write_text("garbage")
    _, (dups, corrupt) = mod.process_dir(str(tmp_path), delete=delete)
    assert len(dups) == 1 and dups[0] in ("a.h5", "dup.h5")
    assert corrupt == ["broken.root"]
    remaining = {p.name for p in tmp_path.iterdir()}
    if delete:
        assert remaining == {"a.h5", "b.h5", "r.root"} - {dups[0]} | (
            {"a.h5", "dup.h5"} - {dups[0]})
    else:
        assert dups[0] + ".duplicate" in remaining
        assert "broken.root.corrupted" in remaining


# --- config conversion ---

def test_json_to_toml_roundtrip(tmp_path):
    pytest.importorskip("tomli")
    pytest.importorskip("tomli_w")
    conf = {"a": 1, "b": {"c": [1, 2]}, "s": "$CONFDIR/x"}
    (tmp_path / "c.json").write_text(json.dumps(conf))
    run_script("json_to_toml.py", tmp_path / "c.json")
    (tmp_path / "c.json").rename(tmp_path / "orig.json")
    run_script("json_to_toml.py", tmp_path / "c.toml")
    assert json.loads((tmp_path / "c.json").read_text()) == conf


# --- known broken scripts ---

@pytest.mark.xfail(strict=True, reason="Bug: reads args.input but the "
                   "argument is called 'file'")
def test_produce_met_xy_nums(tmp_path):
    header = tmp_path / "met.h"
    header.write_text(
        "if(!usemetv2){\n"
        "  if(runera==yUL2018A) METxcorr = -(0.263*npv +-1.91);\n"
        "  if(runera==yUL2018A) METycorr = -(0.0431*npv +-0.112);\n"
        "}\n")
    run_script("produce_met_xy_nums.py", "UL2018", header, "-o",
               tmp_path / "out.json")


@pytest.mark.xfail(strict=True, reason="Uses undefined args.plotdir and the "
                   "coffea.hist API")
def test_calculate_stitching_factors(tmp_path, pepper_out):
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"Data": ["MuonEG"], "Labels": ["TT"]}))
    run_script("calculate_stitching_factors.py", conf,
               pepper_out / "hists/Cut_002_HasJets_met.coffea")


# --- scripts that run over NanoAOD ---

@pytest.mark.needs_data
def test_compute_mc_lumifactors(tmp_path):
    out = tmp_path / "lumi.json"
    run_script("compute_mc_lumifactors.py",
               TESTREPO / "configs/test_config_ul18.json", out, "-p",
               cwd=REPO)
    with open(out) as f:
        res = json.load(f)
    with open(TESTREPO / "configs/lumifactors_ul18.json") as f:
        ref = json.load(f)
    for key in res:
        np.testing.assert_allclose(res[key], ref[key], rtol=1e-6)


@pytest.mark.needs_data
def test_compute_pileup_weights(tmp_path):
    rng = np.random.default_rng(0)
    for suffix, mu in (("", 30), ("_up", 32), ("_down", 28)):
        h = hist.Hist(hist.axis.Regular(100, 0, 100, name="pileup"))
        h.fill(rng.normal(mu, 8, 100000))
        with uproot.recreate(tmp_path / f"pu{suffix}.root") as f:
            f["pileup"] = h
    conf = {
        "import": str(TESTREPO / "configs/test_config_ul18.json"),
        # $CONFDIR refers to this file's directory, not the imported one's
        "mc_datasets": {"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8": [
            str(TESTREPO / "nanoaods_v9_ul18/nanoAOD_test_TTTo2L2Nu.root")]},
        "crosssections": str(TESTREPO / "configs/crosssections_run2.json"),
        "data_pu_hist": str(tmp_path / "pu.root"),
        "data_pu_hist_up": str(tmp_path / "pu_up.root"),
        "data_pu_hist_down": str(tmp_path / "pu_down.root")}
    (tmp_path / "c.json").write_text(json.dumps(conf))
    run_script("compute_pileup_weights.py", tmp_path / "c.json", "-o",
               tmp_path / "out", "--debug", "--metadata",
               tmp_path / "meta.coffea", cwd=REPO)
    with uproot.open(tmp_path / "out/pileup.root") as f:
        assert {k.split(";")[0] for k in f.keys()} == {
            "all_datasets", "all_datasets_up", "all_datasets_down"}
        w = f["all_datasets"].values()
        assert np.all(w >= 0) and np.any(w > 0)
    # Usable by the PileupWeighter
    with uproot.open(tmp_path / "out/pileup.root") as f:
        pepper.scale_factors.PileupWeighter(f)


@pytest.mark.needs_data
def test_ttbarll_kinreco_hists_produce(tmp_path):
    run_script("ttbarll_kinreco_hists_produce.py",
               TESTREPO / "configs/test_config_ul18.json", "-o",
               tmp_path / "out", "--debug", "--metadata",
               tmp_path / "meta.coffea", cwd=REPO)
    with uproot.open(tmp_path / "out/kinreco.root") as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert keys == {"mlb", "mw", "mt", "alphal", "energyfl", "alphaj",
                        "energyfj"}
        assert f["mlb"].values().sum() > 0


@pytest.mark.needs_data
@pytest.mark.xfail(strict=True, reason="Bug: set_cat is called inside the "
                   "column function, before the category columns exist")
def test_ttbarll_dy_sf_produce(tmp_path):
    run_script("ttbarll_dy_sf_produce.py",
               TESTREPO / "configs/test_config_ul18.json", "-o",
               tmp_path / "out", "--debug", "--metadata",
               tmp_path / "meta.coffea", cwd=REPO)
    with open(tmp_path / "out/cutflows.json") as f:
        cf = json.load(f)
    # Only data and DY are processed
    assert set(cf) == {"MuonEG"}
    with open(tmp_path / "out/hists/hists.json") as f:
        coll = HistCollection.from_json(f)
    h = coll.load(next(k for k in coll if k[0] == "HasJets"))
    assert {"in_Z_window", "btags", "channel"} <= set(h.axes.name)
