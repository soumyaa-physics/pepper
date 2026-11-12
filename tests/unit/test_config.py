import json
import pickle
import subprocess
import sys

import pytest

import pepper
from pepper.config import Config, ConfigError
from pepper.hist_defns import HistDefinition
from conftest import REPO, SCRIPTS, TESTREPO


def write(path, obj):
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
    return str(path)


@pytest.fixture
def confdir(tmp_path):
    (tmp_path / "datadir").mkdir()
    (tmp_path / "store").mkdir()
    write(tmp_path / "lumi.json", {"ttbar": 0.5})
    write(tmp_path / "hists.json", {"met": {
        "bins": [{"name": "x", "n_or_arr": 2, "lo": 0, "hi": 1}],
        "fill": {"x": ["MET", "pt"]}}})
    write(tmp_path / "base.json", {
        "datadir": str(tmp_path / "datadir"),
        "store": str(tmp_path / "store"),
        "exp_datasets": {"data": [str(tmp_path / "store/*.root")]},
        "mc_datasets": {},
        "mc_lumifactors": "$CONFDIR/lumi.json",
        "value": 1,
        "nested": {"path": "$DATADIR/f.txt", "list": ["$STOREDIR/x"]},
    })
    return tmp_path


def test_load_basic(confdir):
    conf = Config(str(confdir / "base.json"))
    conf.check_required_args()
    assert conf["value"] == 1
    assert conf["mc_lumifactors"] == {"ttbar": 0.5}
    assert conf["configdir"] == str(confdir)
    assert conf["nested"]["path"] == str(confdir / "datadir/f.txt")
    assert conf["nested"]["list"] == [str(confdir / "store/x")]
    assert conf[["value", "store"]] == [1, str(confdir / "store")]
    assert "value" in conf and "nothing" not in conf
    assert set(conf) >= {"value", "store", "configdir"}


def test_hjson_comments(confdir):
    p = write(confdir / "c.hjson", '''{
        # comment
        import: BASE
        value: 2
    }'''.replace("BASE", str(confdir / "base.json")))
    conf = Config(p)
    assert conf["value"] == 2


def test_import_override(confdir, monkeypatch):
    # Relative import paths are relative to the process' working dir
    monkeypatch.chdir(confdir)
    p = write(confdir / "child.json", {"import": "base.json", "value": 5,
                                       "new": "x"})
    conf = Config(p)
    assert conf["value"] == 5
    assert conf["new"] == "x"
    assert conf["mc_lumifactors"] == {"ttbar": 0.5}


def test_import_chain_and_circular(confdir, monkeypatch):
    monkeypatch.chdir(confdir)
    write(confdir / "a.json", {"import": "b.json", "a": 1})
    write(confdir / "b.json", {"import": "base.json", "b": 2})
    conf = Config(str(confdir / "a.json"))
    assert conf["a"] == 1 and conf["b"] == 2 and conf["value"] == 1
    write(confdir / "c1.json", {"import": "c2.json"})
    write(confdir / "c2.json", {"import": "c1.json"})
    with pytest.raises(ConfigError):
        Config(str(confdir / "c1.json"))


def test_toml(confdir):
    pytest.importorskip("tomli")
    p = write(confdir / "c.toml",
              f'import = "{confdir}/base.json"\nvalue = 7\n')
    assert Config(p)["value"] == 7


def test_json_to_toml_equivalent(confdir, monkeypatch):
    pytest.importorskip("tomli")
    pytest.importorskip("tomli_w")
    monkeypatch.chdir(confdir)
    src = confdir / "base.json"
    dst = confdir / "base.toml"
    subprocess.run([sys.executable, str(SCRIPTS / "json_to_toml.py"),
                    str(src), str(dst)], check=True)
    a, b = Config(str(src)), Config(str(dst))
    for key in a:
        assert a[key] == b[key]


def test_missing_datadir(confdir):
    p = write(confdir / "c.json", {"datadir": "/does/not/exist"})
    with pytest.raises(ConfigError):
        Config(p)


def test_missing_special_var(tmp_path):
    p = write(tmp_path / "c.json", {"x": "$STOREDIR/a"})
    with pytest.raises(ConfigError):
        Config(p)["x"]


def test_required_args(confdir):
    p = write(confdir / "c.json", {"value": 1})
    with pytest.raises(ConfigError):
        Config(p).check_required_args()


def test_hists_behavior(confdir):
    p = write(confdir / "h.json", {"import": str(confdir / "base.json"),
                                   "hists": "$CONFDIR/hists.json"})
    hists = Config(p)["hists"]
    assert isinstance(hists["met"], HistDefinition)


def test_set_and_delete(confdir):
    conf = Config(str(confdir / "base.json"))
    conf["value"] = 3
    assert conf["value"] == 3
    del conf["value"]
    assert "value" not in conf
    with pytest.raises(KeyError):
        conf["value"]
    with pytest.raises(KeyError):
        del conf["nothing"]
    conf["store"] = str(confdir)
    assert conf["nested"]["list"] == [str(confdir / "x")]


def test_pickle(confdir):
    conf = Config(str(confdir / "base.json"))
    conf["mc_lumifactors"]
    conf["value"] = 9
    state = conf.__getstate__()
    assert state["_cache"] == {} and state["_config_loaded"] is None
    conf2 = pickle.loads(pickle.dumps(conf))
    assert conf2["value"] == 9
    assert conf2["mc_lumifactors"] == {"ttbar": 0.5}


def make_store(store, files):
    for f in files:
        p = store / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


def test_get_datasets(confdir):
    make_store(confdir / "store", ["a.root", "b.root"])
    mc_ds = "/TT/RunIISummer20UL18NanoAODv9-v1/NANOAODSIM"
    make_store(confdir / "store",
               ["mc/RunIISummer20UL18NanoAODv9/TT/NANOAODSIM/v1/0/1.root"])
    p = write(confdir / "c.json", {
        "import": str(confdir / "base.json"),
        "mc_datasets": {"tt": [mc_ds], "none": None},
        "file_blacklist": [str(confdir / "store/b.root")]})
    conf = Config(p)
    ds, inv = conf.get_datasets(return_inverse=True)
    assert ds["data"] == [str(confdir / "store/a.root")]
    assert ds["tt"] == [
        "cmslfn://store/mc/RunIISummer20UL18NanoAODv9/TT/NANOAODSIM/v1/0/"
        "1.root"]
    assert inv[ds["tt"][0]] == "tt"
    assert set(conf.get_datasets(dstype="mc")) == {"tt"}
    assert set(conf.get_datasets(dstype="data")) == {"data"}
    assert set(conf.get_datasets(include=["t"])) == {"tt"}
    assert set(conf.get_datasets(exclude=["tt"])) == {"data"}
    with pytest.raises(ValueError):
        conf.get_datasets(dstype="bad")


def test_get_datasets_errors(confdir):
    p = write(confdir / "c.json", {
        "import": str(confdir / "base.json"),
        "mc_datasets": {"data": []}})
    with pytest.raises(ConfigError):
        Config(p).get_datasets()
    p = write(confdir / "d.json", {
        "import": str(confdir / "base.json"),
        "exp_datasets": {"x": [str(confdir / "store/none*.root")]}})
    with pytest.raises(ConfigError):
        Config(p).get_datasets()


def test_get_paths_for_lfn(confdir, monkeypatch):
    calls = []

    def fake_resolve(lfn, store, xrootddomain, blacklist, eos, priority):
        calls.append((store, xrootddomain, blacklist))
        return ["/local/a.root", "root://x//store/a.root"]
    monkeypatch.setattr(pepper.datasets, "resolve_lfn", fake_resolve)
    conf = Config(str(confdir / "base.json"))
    assert conf.get_paths_for_lfn("/plain/file.root") == ["/plain/file.root"]
    conf.get_paths_for_lfn("cmslfn://store/a.root")
    assert calls[-1] == (str(confdir / "store"), None, None)
    conf["file_mode"] = "xrootd"
    conf["xrootddomain"] = "dom"
    conf["xrootd_url_blacklist"] = ["bad"]
    conf.get_paths_for_lfn("cmslfn://store/a.root")
    assert calls[-1] == (None, "dom", ["bad"])
    conf["file_mode"] = "local+xrootd"
    conf["local_file_blacklist"] = ["/local/a.root"]
    assert conf.get_paths_for_lfn("cmslfn://store/a.root") == \
        ["root://x//store/a.root"]
    assert calls[-1][:2] == (str(confdir / "store"), "dom")


# --- Real configs shipped with the integration tests ---

def load_all_keys(conf):
    for key in conf:
        conf[key]


@pytest.mark.needs_data
def test_load_ul18_test_config(monkeypatch):
    monkeypatch.chdir(REPO)
    conf = pepper.ConfigTTbarLL(str(TESTREPO / "configs/test_config_ul18.json"))
    conf.check_required_args()
    load_all_keys(conf)
    assert conf["year"] == "ul2018"


@pytest.mark.needs_data
@pytest.mark.needs_cvmfs
def test_load_2024_test_config(monkeypatch):
    monkeypatch.chdir(REPO)
    conf = pepper.ConfigTTbarLL(str(TESTREPO / "configs/test_config_2024.json"))
    load_all_keys(conf)
