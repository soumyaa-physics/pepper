import time
from dataclasses import dataclass

import coffea.util
import pytest
from coffea.processor.accumulator import set_accumulator
from coffea.processor.executor import FileMeta

import pepper
from pepper import executor as ex


@dataclass
class LocalExecutor(ex.ResumableExecutor):
    """Runs items in-process"""
    fail_on: object = None

    def _submit(self, items, function):
        for item in items:
            if item == self.fail_on:
                # Give the state thread time to pick up previous results
                time.sleep(0.5)
                raise RuntimeError("fail")
            yield function(item)


def square(x):
    return {"sum": x * x, "n": 1}


def make(tmp_path, **kw):
    return LocalExecutor(state_file_name=str(tmp_path / "state.coffea"),
                         status=False, compression=None, **kw)


def test_run_and_save(tmp_path):
    e = make(tmp_path)
    acc, _ = e(list(range(5)), square, None)
    assert acc == {"sum": 30, "n": 5}
    state = coffea.util.load(tmp_path / "state.coffea")
    assert sorted(state["items_done"]) == list(range(5))
    assert state["version"] == ex.STATEFILE_VERSION
    assert not list(tmp_path.glob("pepper_temp*"))


def test_resume_skips_done(tmp_path):
    make(tmp_path)([1, 2], square, None)
    calls = []

    def f(x):
        calls.append(x)
        return square(x)
    e = make(tmp_path)
    e.load_state()
    acc, _ = e([1, 2, 3], f, None)
    assert calls == [3]
    assert acc == {"sum": 14, "n": 3}
    # Nothing left to do
    e = make(tmp_path)
    e.load_state()
    acc, _ = e([1, 2, 3], f, None)
    assert calls == [3] and acc["n"] == 3


def test_failure_keeps_finished_items(tmp_path):
    e = make(tmp_path, fail_on=3, save_interval=0)
    with pytest.raises(RuntimeError):
        e([1, 2, 3, 4], square, None)
    e = make(tmp_path)
    e.load_state()
    assert sorted(e.state["items_done"]) == [1, 2]
    acc, _ = e([1, 2, 3, 4], square, None)
    assert acc == {"sum": 30, "n": 4}


def test_remove_state_at_end(tmp_path):
    make(tmp_path, remove_state_at_end=True)([1], square, None)
    assert not (tmp_path / "state.coffea").exists()


def test_load_state_errors(tmp_path):
    e = make(tmp_path)
    with pytest.raises(FileNotFoundError):
        e.load_state()
    with pytest.raises(ValueError):
        LocalExecutor().load_state()
    pepper.misc.save({"version": -1}, str(tmp_path / "old.coffea"))
    with pytest.raises(ex.StateFileError):
        e.load_state(str(tmp_path / "old.coffea"))


def test_reset_and_copy(tmp_path):
    e = make(tmp_path)
    e([1], square, None)
    e.state["userdata"]["x"] = 1
    cp = e.copy(save_interval=5)
    assert cp.save_interval == 5 and cp.state == e.state
    cp.state["userdata"]["x"] = 2
    assert e.state["userdata"]["x"] == 1
    e.reset_state()
    assert e.state["items_done"] == [] and e.state["accumulator"] is None


def test_userdata_persisted(tmp_path):
    e = make(tmp_path)
    e.state["userdata"]["chunksize"] = 123
    e([1], square, None)
    e2 = make(tmp_path)
    e2.load_state()
    assert e2.state["userdata"]["chunksize"] == 123


# --- Runner helpers ---

def test_resolved_filepaths_fresh():
    f = ex.Runner._resolved_filepaths_are_fresh
    assert f(None, False)
    assert not f(None, 90)
    assert f(time.time() - 86400, 2)
    assert not f(time.time() - 3 * 86400, 2)


def meta(**kw):
    m = {"store_path": None, "xrootddomain": "dom",
         "use_eos_redirector": True, "url_blacklist": None,
         "local_file_blacklist": None}
    m.update(kw)
    return m


def test_locate_lfn_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(pepper.datasets, "locate_lfn",
                        lambda *a: calls.append(a) or ["root://new//f"])
    assert ex.Runner._locate_lfn("/plain.root", meta()) == ["/plain.root"]
    lfn = "cmslfn://store/f.root"
    cached = meta(resolve_files_at_preprocess=True,
                  resolved_filepaths=["root://cached//f"],
                  resolved_filepaths_time=time.time())
    assert ex.Runner._locate_lfn(lfn, cached) == ["root://cached//f"]
    assert calls == []
    cached["resolved_filepaths_time"] = time.time() - 100 * 86400
    assert ex.Runner._locate_lfn(lfn, cached) == ["root://new//f"]
    cached["resolve_files_at_preprocess"] = False
    cached["resolved_filepaths_time"] = time.time()
    assert ex.Runner._locate_lfn(lfn, cached) == ["root://new//f"]
    assert len(calls) == 2


def test_filter_and_prioritize():
    lfn = "cmslfn://store/f.root"
    pfns = ["/local/f.root", "root://a//f", "root://b//f"]
    m = meta(url_blacklist=["a"], url_priority=None)
    assert ex.Runner._filter_and_prioritize(lfn, pfns, m) == \
        ["/local/f.root", "root://b//f"]
    m = meta(local_file_blacklist=["/local/f.root"], url_priority=["b"])
    assert ex.Runner._filter_and_prioritize(lfn, pfns, m) == \
        ["root://b//f", "root://a//f"]


def test_try_open_falls_back(monkeypatch):
    item = type("Item", (), {"filename": "cmslfn://store/f.root"})()
    monkeypatch.setattr(pepper.datasets, "locate_lfn",
                        lambda *a: ["root://bad//f", "root://good//f"])
    tried = []

    def opener(path):
        tried.append(path)
        if "bad" in path:
            raise OSError("broken")
        return "ok"
    res, raw = ex.Runner._try_open_and_process(
        item, meta(check_can_open_local=False), opener)
    assert res == "ok" and tried == ["root://bad//f", "root://good//f"]
    with pytest.raises(OSError):
        ex.Runner._try_open_and_process(
            item, meta(check_can_open_local=False,
                       url_blacklist=["good"]),
            lambda p: (_ for _ in ()).throw(OSError()))


def test_can_open_local(tmp_path):
    f = tmp_path / "f"
    f.write_text("abcd")
    assert ex._can_open_local_within_timeout(str(f), timeout=5)
    assert not ex._can_open_local_within_timeout(str(tmp_path / "x"), 5)
    with pytest.raises(OSError):
        ex._raise_if_cant_open_local(str(tmp_path / "x"), True, 5)
    ex._raise_if_cant_open_local("root://x//f", True, 5)
    ex._raise_if_cant_open_local(str(tmp_path / "x"), False, 5)


def test_purge_stale_resolved_filepaths():
    now = time.time()

    def fm(ds, fn, t):
        m = FileMeta(ds, fn, "Events", {"resolved_filepaths_time": t})
        return m
    fresh = fm("a", "f1", now)
    stale = fm("a", "f2", now - 100 * 86400)
    other = fm("b", "f3", now - 100 * 86400)
    pre = type("Pre", (), {})()
    pre.state = {"accumulator": set_accumulator([fresh, stale, other]),
                 "items_done": [fresh, stale, other]}
    ex.purge_stale_resolved_filepaths(pre, {"a": ["f1", "f2"]}, "Events", 90)
    assert set(pre.state["accumulator"]) == {fresh, other}
    assert pre.state["items_done"] == [fresh, other]
    # Empty state is fine
    pre.state = {"accumulator": None, "items_done": []}
    ex.purge_stale_resolved_filepaths(pre, {"a": ["f1"]}, "Events", 90)
