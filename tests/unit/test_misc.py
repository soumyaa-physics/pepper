import awkward as ak
import coffea.util
import hist
import numpy as np
import pytest

from pepper import misc


def test_runs_and_years():
    assert misc.get_run_for_year("ul2018") == misc.LHCRun.Run2
    assert misc.get_run_for_year("2024") == misc.LHCRun.Run3
    with pytest.raises(NotImplementedError):
        misc.get_run_for_year("1999")
    assert "2018" in misc.get_years_for_run("Run2")
    assert misc.get_years_for_run(misc.LHCRun.Run3)[-1] == "2025"
    with pytest.raises(NotImplementedError):
        misc.get_years_for_run("Run9")


TRIGGERS = {"MuonEG": ["HLT_Mu23_Ele12_v"], "DoubleMuon": ["HLT_Mu17_Mu8"],
            "SingleMuon": ["HLT_IsoMu24", "HLT_Mu17_Mu8"],
            "SingleMuon_B": ["HLT_IsoMu27"]}
ORDER = ["MuonEG", "DoubleMuon", "SingleMuon"]


def test_trigger_paths_mc():
    pos, neg = misc.get_trigger_paths_for("tt", True, TRIGGERS, ORDER)
    assert pos == ["Mu23_Ele12", "Mu17_Mu8", "IsoMu24"]
    assert neg == []


def test_trigger_paths_data_order():
    pos, neg = misc.get_trigger_paths_for("SingleMuon", False, TRIGGERS,
                                          ORDER, normalize=False)
    assert pos == ["HLT_IsoMu24", "HLT_Mu17_Mu8"]
    assert neg == ["HLT_Mu23_Ele12_v", "HLT_Mu17_Mu8"]
    pos, neg = misc.get_trigger_paths_for("MuonEG", False, TRIGGERS, ORDER)
    assert neg == []
    with pytest.raises(ValueError):
        misc.get_trigger_paths_for("EGamma", False, TRIGGERS, ORDER)


def test_trigger_paths_era():
    order = {"B": ["MuonEG", "SingleMuon_B"], "other": ORDER}
    pos, neg = misc.get_trigger_paths_for("SingleMuon", False, TRIGGERS,
                                          order, era="B")
    assert pos == ["IsoMu27"] and neg == ["Mu23_Ele12"]
    pos, _ = misc.get_trigger_paths_for("SingleMuon", False, TRIGGERS,
                                        order, era="C")
    assert pos == ["IsoMu24", "Mu17_Mu8"]


def test_normalize_trigger_path():
    assert misc.normalize_trigger_path("HLT_X_v") == "X"
    assert misc.normalize_trigger_path("L1_Y") == "Y"


def test_lhe_scale_idxs():
    assert misc.get_lhe_scale_idxs(9) == [7, 1, 5, 3]
    with pytest.raises(RuntimeError):
        misc.get_lhe_scale_idxs(3)


def test_eos_path_to_url():
    assert misc.eos_path_to_url("/eos/user/a/b/f.root") == \
        "root://eosuser.cern.ch//eos/user/a/b/f.root"
    assert misc.eos_path_to_url("/eos/home-a/b/f.root") == \
        "root://eosuser.cern.ch//eos/user/a/b/f.root"
    assert misc.eos_path_to_url("/eos/cms/store/mc/f.root") == \
        "root://eoscms.cern.ch//store/mc/f.root"
    with pytest.raises(ValueError):
        misc.eos_path_to_url("/eos/atlas/x")


def test_akremask_and_akismasked():
    res = misc.akremask(ak.Array([1, 2]), np.array([False, True, True]))
    assert res.tolist() == [None, 1, 2]
    assert misc.akismasked(res)
    assert not misc.akismasked(ak.Array([[1], [2]]))
    assert misc.akismasked(ak.Array([[1, None]]))
    assert misc.akremask(ak.Array([]), np.array([False])).tolist() == [None]
    with pytest.raises(ValueError):
        misc.akremask(ak.Array([1]), np.array([True, True]))


def test_onedimeval():
    arr = ak.Array([[1., 2.], [], [3.]])
    res = misc.onedimeval(lambda x, s: x * 2 if s == "a" else x, arr, "a")
    assert res.tolist() == [[2., 4.], [], [6.]]


def test_chunked_calls():
    calls = []

    @misc.chunked_calls("x", returns_multiple=True, chunksize=3)
    def f(x, y, scale=1):
        calls.append(len(x))
        return x * scale, y + x
    x = np.arange(10.)
    a, b = f(x, np.ones(10), scale=2)
    np.testing.assert_array_equal(a, x * 2)
    np.testing.assert_array_equal(b, x + 1)
    assert calls == [3, 3, 3, 1]
    calls.clear()
    f(ak.Array(x), np.ones(10), chunksize=5, num_threads=2)
    assert sorted(calls) == [5, 5]


def test_hist_cat_values_and_split():
    h = hist.Hist(hist.axis.StrCategory(["a", "b"], name="c1"),
                  hist.axis.StrCategory(["x"], name="c2"),
                  hist.axis.Regular(2, 0, 2, name="x"))
    h.fill(c1="a", c2="x", x=[0.5])
    vals = misc.get_hist_cat_values(h)
    assert set(vals) == {("a", "x"), ("b", "x")}
    assert vals[("a", "x")].tolist() == [1., 0.]
    assert set(misc.hist_split_strcat(h)) == {("a", "x"), ("b", "x")}
    plain = hist.Hist(hist.axis.Regular(2, 0, 2))
    assert list(misc.get_hist_cat_values(plain)) == [()]


def test_hist_divide_no_flow():
    def h(vals):
        out = hist.Hist(hist.axis.Regular(3, 0, 3, flow=False),
                        storage="Weight")
        out[...] = np.stack([vals, vals], axis=-1)
        return out
    r = misc.hist_divide(h(np.array([2., 0., 3.])), h(np.array([1., 0., 6.])))
    np.testing.assert_allclose(r.values(), [2., 0., 0.5])
    assert np.isnan(r.variances()[1])
    np.testing.assert_allclose(r.variances()[0], (2 * 1 + 1 * 4) / 1)
    assert r.label == "Ratio"


@pytest.mark.xfail(strict=True, reason="Bug: variances with flow=True mixed "
                   "with values without flow in hist_divide")
def test_hist_divide_with_flow():
    num = hist.Hist(hist.axis.Regular(3, 0, 3), storage="Weight")
    num.fill([0.5, 1.5])
    misc.hist_divide(num, num.copy())


def test_save_load(tmp_path):
    obj = {"a": hist.Hist(hist.axis.Regular(2, 0, 1)), "b": [1, 2]}
    misc.save(obj, str(tmp_path / "x.coffea"))
    loaded = coffea.util.load(str(tmp_path / "x.coffea"))
    assert loaded["b"] == [1, 2]
    assert isinstance(loaded["a"], hist.Hist)


def test_get_event_files(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "a.h5").touch()
    (tmp_path / "ds" / "b.root").touch()
    assert misc.get_event_files(str(tmp_path), ".h5", ["ds"]) == {
        "ds": [str(tmp_path / "ds" / "a.h5")]}
