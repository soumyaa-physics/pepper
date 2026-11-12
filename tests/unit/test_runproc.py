"""End-to-end runs of pepper.runproc with a minimal processor on the
NanoAODs of the test submodule. Covers NanoAOD reading, histogram and
cutflow output, event output, metadata caching and resuming."""
import json
import os
import subprocess
import sys
from pathlib import Path

import awkward as ak
import coffea.util
import numpy as np
import pytest
import uproot

from pepper.hdffile import HDF5File
from pepper.hist_collection import HistCollection
from pepper.executor import STATEFILE_VERSION
from conftest import REPO, UL18_NANO

FILES = Path(__file__).resolve().parent / "files"

pytestmark = pytest.mark.needs_data

PROC = str(FILES / "mini_processor.py")
TRIGGER = "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL"


def make_config(path, **extra):
    conf = {
        "exp_datasets": {"MuonEG": [str(UL18_NANO / "nanoAOD_test_MuonEG.root")]},
        "mc_datasets": {"TT": [str(UL18_NANO / "nanoAOD_test_TTTo2L2Nu.root")]},
        "mc_lumifactors": {"TT": 2.0},
        "dataset_trigger_map": {"MuonEG": [TRIGGER]},
        "dataset_trigger_order": ["MuonEG"],
        "dataset_for_systematics": {},
        "compute_systematics": True,
        "hists": str(FILES / "mini_hists.json"),
        "histogram_format": "hist",
    }
    conf.update(extra)
    path.write_text(json.dumps(conf))
    return str(path)


def runproc(cwd, config, out, *args, env=None, check=True):
    cmd = [sys.executable, "-m", "pepper.runproc", PROC, config, "-o",
           str(out), "--metadata", str(cwd / "meta.coffea"),
           "--chunksize", "400", "--force_chunksize", *args]
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO) + os.pathsep + e.get("PYTHONPATH", "")
    e.update(env or {})
    res = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True,
                         text=True, timeout=600)
    if check and res.returncode != 0:
        raise AssertionError(f"runproc failed:\n{res.stdout}\n{res.stderr}")
    return res


def expected_counts():
    """Selection redone directly with uproot"""
    counts = {}
    for ds, fname in [("MuonEG", "nanoAOD_test_MuonEG.root"),
                      ("TT", "nanoAOD_test_TTTo2L2Nu.root")]:
        ev = uproot.open(UL18_NANO / fname)["Events"].arrays(
            [TRIGGER, "Muon_pt", "Muon_tightId", "MET_pt", "genWeight"]
            if ds == "TT" else [TRIGGER, "Muon_pt", "Muon_tightId",
                                "MET_pt"])
        w = ev.genWeight if ds == "TT" else np.ones(len(ev))
        trig = ev[TRIGGER]
        nmu = ak.sum((ev.Muon_pt > 20) & ev.Muon_tightId, axis=1)
        met = ev.MET_pt > 30
        counts[ds] = {
            "BeforeCuts": ak.sum(w),
            "Trigger": ak.sum(w[trig]),
            "HasMuon": ak.sum(w[trig & (nmu > 0)]),
            "MET": ak.sum(w[trig & (nmu > 0) & met]),
            "MET_2mu": ak.sum(w[trig & (nmu > 1) & met]),
        }
    return counts


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("full")
    config = make_config(cwd / "config.json")
    runproc(cwd, config, cwd / "out")
    return cwd


def load_cutflows(out):
    with open(out / "cutflows.json") as f:
        return json.load(f)


def test_cutflows(full_run):
    cf = load_cutflows(full_run / "out")
    exp = expected_counts()
    for ds in ("MuonEG", "TT"):
        for cut in ("BeforeCuts", "Trigger", "HasMuon", "MET"):
            assert cf[ds]["all"][cut] == pytest.approx(exp[ds][cut]), \
                (ds, cut)
        assert cf[ds]["is_2mu"]["MET"] == pytest.approx(exp[ds]["MET_2mu"])


def test_histograms(full_run):
    with open(full_run / "out/hists/hists.json") as f:
        coll = HistCollection.from_json(f)
    assert coll.userdata["cuts"] == ["BeforeCuts", "Trigger", "HasMuon",
                                     "MET"]
    # One histogram per cut and definition
    assert set(coll) == {(c, h) for c in coll.userdata["cuts"]
                         for h in ("met", "mu_pt")}
    h = coll.load(("MET", "met"))
    assert set(h.axes["dataset"]) == {"MuonEG", "TT"}
    assert set(h.axes["sys"]) == {"nominal", "toy_up", "toy_down"}
    assert set(h.axes["nmu"]) == {"is_1mu", "is_2mu"}
    cf = load_cutflows(full_run / "out")
    assert h["TT", "nominal", ...].sum(flow=True).value == \
        pytest.approx(cf["TT"]["all"]["MET"])
    assert h["TT", "toy_up", ...].sum(flow=True).value == \
        pytest.approx(1.1 * cf["TT"]["all"]["MET"])
    for f in os.listdir(full_run / "out/hists"):
        assert f == "hists.json" or f.startswith("Cut_")


def test_root_histograms(tmp_path):
    config = make_config(tmp_path / "c.json", histogram_format="root",
                         exp_datasets={})
    runproc(tmp_path, config, tmp_path / "out")
    files = os.listdir(tmp_path / "out/hists")
    assert "Cut_003_MET_met.root" in files
    with open(tmp_path / "out/hists/hists.json") as f:
        h = HistCollection.from_json(f).load(("MET", "met"))
    assert h.sum(flow=True).value > 0


def test_lumifactors_post(tmp_path):
    xs = tmp_path / "xs.json"
    xs.write_text(json.dumps({"TT": 10.}))
    config = make_config(tmp_path / "c.json", mc_lumifactors=False,
                         crosssections=str(xs), luminosity=2.,
                         exp_datasets={})
    runproc(tmp_path, config, tmp_path / "out")
    cf = load_cutflows(tmp_path / "out")
    assert cf["TT"]["all"]["BeforeCuts"] == pytest.approx(20.)


@pytest.mark.parametrize("fmt", ["root", pytest.param(
    "hdf5", marks=pytest.mark.xfail(
        strict=True, reason="Bug: cutflags is None without deferred cuts, "
        "HDF5File cannot write None (processor.py _save_per_event_info_hdf5)"
    ))])
def test_event_output(tmp_path, fmt):
    config = make_config(
        tmp_path / "c.json", column_output_format=fmt,
        columns_to_save={"met": ["MET", "pt"], "mupt": ["Muon", "pt"]})
    runproc(tmp_path, config, tmp_path / "out", "--eventdir",
            str(tmp_path / "ev"))
    cf = load_cutflows(tmp_path / "out")
    for ds in ("MuonEG", "TT"):
        files = sorted((tmp_path / "ev" / ds).iterdir())
        # One file per chunk
        assert len(files) >= 2
        nevents, sumw = 0, 0.
        for path in files:
            if fmt == "root":
                with uproot.open(path) as f:
                    if "Events" not in f:
                        continue
                    ev = f["Events"].arrays()
                    assert ak.all(ev.met > 30)
                    assert ak.all(ak.num(ev.mupt) > 0)
                    nevents += len(ev)
                    sumw += ak.sum(ev.weight)
                    assert "Categories/nmu" in f
            else:
                with HDF5File(str(path), "r") as f:
                    ev = f["events"]
                    assert ak.all(ev.met > 30)
                    assert f["dsname"] == ds
                    assert f["cutnames"][-1] == "MET"
                    assert set(ak.fields(f["categories"])) == {"nmu"}
                    nevents += len(ev)
                    if ds == "TT":
                        assert "toy_up" in ak.fields(f["systematics"])
                        sumw += ak.sum(f["systematics"]["weight"])
                    else:
                        sumw += ak.sum(f["weight"])
        assert nevents > 0
        assert sumw == pytest.approx(cf[ds]["all"]["MET"])


def test_metadata_cache(full_run):
    state = coffea.util.load(full_run / "meta.coffea")
    assert state["version"] == STATEFILE_VERSION
    files = {fm.filename for fm in state["accumulator"]}
    assert files == {str(UL18_NANO / "nanoAOD_test_MuonEG.root"),
                     str(UL18_NANO / "nanoAOD_test_TTTo2L2Nu.root")}


def test_statefile_requires_resume(full_run):
    res = runproc(full_run, str(full_run / "config.json"), full_run / "out",
                  check=False)
    assert res.returncode == 1
    assert "--resume" in res.stdout


def test_metadata_equals_statedata(tmp_path):
    config = make_config(tmp_path / "c.json")
    res = runproc(tmp_path, config, tmp_path / "out", "--statedata",
                  str(tmp_path / "meta.coffea"), check=False)
    assert res.returncode == 1


def test_resume_chunksize_mismatch(full_run):
    res = runproc(full_run, str(full_run / "config.json"), full_run / "out",
                  "-R", "--chunksize", "300", check=False)
    assert res.returncode != 0
    assert "chunksize" in res.stderr


def test_resume_partial(tmp_path, full_run):
    """Run only MC, then resume with all datasets. Must match a full run"""
    config = make_config(tmp_path / "c.json")
    runproc(tmp_path, config, tmp_path / "out", "--mc")
    state = coffea.util.load(tmp_path / "out/pepper_state.coffea")
    assert {i.dataset for i in state["items_done"]} == {"TT"}
    runproc(tmp_path, config, tmp_path / "out", "-R")
    assert load_cutflows(tmp_path / "out") == \
        load_cutflows(full_run / "out")


def test_resume_after_failure(tmp_path, full_run):
    config = make_config(tmp_path / "c.json")
    res = runproc(tmp_path, config, tmp_path / "out",
                  env={"MINIPROC_FAIL": "MuonEG"}, check=False)
    assert res.returncode != 0
    assert "Requested failure" in res.stderr
    runproc(tmp_path, config, tmp_path / "out", "-R")
    assert load_cutflows(tmp_path / "out") == \
        load_cutflows(full_run / "out")
