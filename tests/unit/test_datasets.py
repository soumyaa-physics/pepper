import sys
import types

import pytest

from pepper import datasets

MC_DS = ("/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9"
         "-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DATA_DS = "/MuonEG/Run2018A-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD"
MC_DIR = ("mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-"
          "pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/0000")
DATA_DIR = "data/Run2018A/MuonEG/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v1/0000"


@pytest.fixture
def store(tmp_path):
    store = tmp_path / "store"
    for f in [f"{MC_DIR}/a.root", f"{MC_DIR}/b.root", f"{DATA_DIR}/c.root",
              f"{DATA_DIR}/ignore.txt",
              "user/me/Prim/tag/220101_000000/0000/u.root"]:
        p = store / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return store


@pytest.fixture
def fake_xrootd(monkeypatch):
    """Replaces XRootD.client; set .locations to control locate()"""
    mod = types.SimpleNamespace(locations=["site1:1094", "site2:1094"])

    class FileSystem:
        def __init__(self, url):
            mod.url = url

        def locate(self, lfn, flags):
            if mod.locations is None:
                return types.SimpleNamespace(message="not found"), None
            return None, [types.SimpleNamespace(address=a)
                          for a in mod.locations]
    client = types.SimpleNamespace(
        FileSystem=FileSystem,
        flags=types.SimpleNamespace(
            OpenFlags=types.SimpleNamespace(MAKEPATH=0)))
    xrd = types.ModuleType("XRootD")
    xrd.client = client
    monkeypatch.setitem(sys.modules, "XRootD", xrd)
    monkeypatch.setitem(sys.modules, "XRootD.client", client)
    return mod


def test_dataset_to_lfn_local(store):
    assert datasets.dataset_to_lfn_local(MC_DS, str(store)) == {
        f"/store/{MC_DIR}/a.root", f"/store/{MC_DIR}/b.root"}
    assert datasets.dataset_to_lfn_local(DATA_DS, str(store)) == {
        f"/store/{DATA_DIR}/c.root"}
    assert datasets.dataset_to_lfn_local(
        "/Prim/me-tag-v1/USER", str(store)) == {
            "/store/user/me/Prim/tag/220101_000000/0000/u.root"}
    with pytest.raises(ValueError):
        datasets.dataset_to_lfn_local("/a/b/MINIAOD", str(store))


def test_lfn_conversions():
    assert datasets.lfn_to_local_path("/store/mc/x.root", "/st/") == \
        "/st/mc/x.root"
    assert datasets.lfn_to_xrootd_path("/store/x.root", "dom") == \
        "root://dom//store/x.root"


def test_dataset_to_lfns(store, monkeypatch):
    lfns = datasets.dataset_to_lfns(MC_DS, str(store))
    assert lfns == [f"cmslfn://store/{MC_DIR}/a.root",
                    f"cmslfn://store/{MC_DIR}/b.root"]
    monkeypatch.setattr(datasets, "dataset_to_lfn_dbs",
                        lambda ds: {f"/store/{MC_DIR}/z.root"})
    assert len(datasets.dataset_to_lfns(MC_DS, mode="xrootd")) == 1
    assert len(datasets.dataset_to_lfns(MC_DS, str(store),
                                        mode="local+xrootd")) == 3
    with pytest.raises(ValueError):
        datasets.dataset_to_lfns(MC_DS, mode="bad")
    with pytest.raises(ValueError):
        datasets.dataset_to_lfns(MC_DS, None, mode="local")


def test_read_paths(store, tmp_path):
    # Glob
    assert len(datasets.read_paths(f"{store}/{MC_DIR}/*.root")) == 2
    # Dataset name
    assert len(datasets.read_paths(DATA_DS, str(store))) == 1
    # File listing globs and dataset names
    listing = tmp_path / "list.txt"
    listing.write_text(f"{store}/{MC_DIR}/a.root\n\n{DATA_DS}\n")
    paths = datasets.read_paths(str(listing), str(store))
    assert paths == [f"{store}/{MC_DIR}/a.root",
                     f"cmslfn://store/{DATA_DIR}/c.root"]


def test_read_paths_eos():
    assert datasets.read_paths("/eos/nonexistent/*.root") == []


def test_expand_datasetdict(store):
    ds, inv = datasets.expand_datasetdict(
        {"tt": [MC_DS, MC_DS], "data": [f"{store}/{DATA_DIR}/*.root"]},
        str(store), ignore_path=lambda p: p.endswith("b.root"))
    assert ds["tt"] == [f"cmslfn://store/{MC_DIR}/a.root"]
    assert ds["data"] == [f"{store}/{DATA_DIR}/c.root"]
    assert inv[ds["tt"][0]] == "tt"
    with pytest.raises(RuntimeError):
        datasets.expand_datasetdict({"a": [MC_DS], "b": [MC_DS]}, str(store))


def test_apply_url_priority_and_blacklist():
    pfns = ["/local/f.root", "root://a.de//f", "root://b.it//f",
            "root://c.ch//f"]
    f = datasets.apply_url_priority_and_blacklist
    assert f(pfns) == pfns
    assert f(pfns, url_priority=["c.ch", "b.it"]) == [
        "/local/f.root", "root://c.ch//f", "root://b.it//f", "root://a.de//f"]
    assert f(pfns, url_blacklist=["a.de"]) == [
        "/local/f.root", "root://b.it//f", "root://c.ch//f"]
    with pytest.raises(ValueError):
        f(pfns, url_blacklist=[".de", ".it", ".ch"])
    # Blacklisting is fine if there were no URLs to begin with
    assert f(["/local/f.root"], url_blacklist=["x"]) == ["/local/f.root"]


def test_locate_lfn_local(store):
    lfn = f"cmslfn://store/{MC_DIR}/a.root"
    assert datasets.locate_lfn(lfn, str(store)) == [
        f"{store}/{MC_DIR}/a.root"]
    assert datasets.locate_lfn(f"cmslfn://store/{MC_DIR}/no.root",
                               str(store)) == []


@pytest.mark.xfail(strict=True, reason="Bug: local path only added if "
                   "use_eos_redirector is True (datasets.py locate_lfn)")
def test_locate_lfn_local_without_eos_redirector(store):
    lfn = f"cmslfn://store/{MC_DIR}/a.root"
    assert datasets.locate_lfn(lfn, str(store), use_eos_redirector=False) \
        == [f"{store}/{MC_DIR}/a.root"]


def test_locate_lfn_xrootd(store, fake_xrootd):
    lfn = f"cmslfn://store/{MC_DIR}/a.root"
    pfns = datasets.locate_lfn(lfn, str(store), "redir.cern.ch")
    assert fake_xrootd.url == "root://redir.cern.ch"
    assert pfns == [f"{store}/{MC_DIR}/a.root",
                    f"root://site1:1094//store/{MC_DIR}/a.root",
                    f"root://site2:1094//store/{MC_DIR}/a.root"]
    fake_xrootd.locations = None
    with pytest.raises(OSError):
        datasets.locate_lfn(lfn, None, "redir.cern.ch")


def test_resolve_lfn(fake_xrootd):
    pfns = datasets.resolve_lfn("/store/x.root", None, "redir",
                                url_blacklist=["site1"],
                                url_priority=["site2"])
    assert pfns == ["root://site2:1094//store/x.root"]


def test_dbs_query_mocked(monkeypatch, tmp_path):
    reply = b'[{"logical_file_name": "/store/a.root"}, ' \
        b'{"logical_file_name": "/store/b.root"}]'
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return reply

    def urlopen(url, context=None):
        seen["url"] = url
        return Resp()
    monkeypatch.setattr(datasets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(datasets, "_get_cernca_rootcert", lambda: {})
    monkeypatch.setattr(datasets, "_get_cert_vomsproxy", lambda: {})
    monkeypatch.setattr(datasets.ssl.SSLContext, "load_verify_locations",
                        lambda self, **kw: None)
    monkeypatch.setattr(datasets.ssl.SSLContext, "load_cert_chain",
                        lambda self, **kw: None)
    assert datasets.dataset_to_lfn_dbs(MC_DS) == {"/store/a.root",
                                                  "/store/b.root"}
    assert f"dataset={MC_DS}" in seen["url"]


def test_cert_vomsproxy(monkeypatch, tmp_path):
    monkeypatch.delenv("X509_USER_KEY", raising=False)
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        datasets._get_cert_vomsproxy()
    (tmp_path / "proxy").touch()
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "proxy"))
    assert datasets._get_cert_vomsproxy() == {
        "certfile": str(tmp_path / "proxy")}
    monkeypatch.setenv("X509_USER_KEY", "k")
    monkeypatch.setenv("X509_USER_CERT", "c")
    assert datasets._get_cert_vomsproxy() == {"certfile": "c",
                                              "keyfile": "k"}


# --- Real grid access, same datasets as the ul18_xrootd chain test ---

@pytest.fixture(scope="module")
def dbs_lfns():
    return sorted(datasets.dataset_to_lfn_dbs(MC_DS))


@pytest.mark.needs_grid
def test_grid_dbs(dbs_lfns):
    assert len(dbs_lfns) > 10
    assert all(lfn.startswith("/store/mc/RunIISummer20UL18NanoAODv9/")
               for lfn in dbs_lfns)


@pytest.mark.needs_grid
@pytest.mark.parametrize("domain", ["xrootd-cms.infn.it",
                                    "cms-xrd-global.cern.ch"])
def test_grid_locate(dbs_lfns, domain):
    pfns = datasets.locate_lfn(dbs_lfns[0], xrootddomain=domain)
    assert len(pfns) > 0
    assert all(p.startswith("root://") and p.endswith(dbs_lfns[0])
               for p in pfns)


@pytest.mark.needs_grid
def test_grid_open_file(dbs_lfns):
    import uproot
    pfns = datasets.resolve_lfn(dbs_lfns[0], xrootddomain="xrootd-cms.infn.it")
    for pfn in pfns:
        try:
            with uproot.open(pfn, timeout=60) as f:
                assert f["Events"].num_entries > 0
                return
        except OSError:
            continue
    pytest.fail(f"Could not open any of {pfns}")
