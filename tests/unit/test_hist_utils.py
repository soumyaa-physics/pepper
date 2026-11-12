import hist
import numpy as np
import pytest

from pepper import hist_utils


def make_hist(datasets=("a", "b", "c"), syst=False):
    axes = [hist.axis.StrCategory(list(datasets), name="dataset")]
    if syst:
        axes.append(hist.axis.StrCategory(["nominal", "up"], name="sys"))
    axes.append(hist.axis.Regular(4, 0, 4, name="x"))
    h = hist.Hist(*axes, storage="Weight")
    for i, ds in enumerate(datasets):
        kw = {"sys": "nominal"} if syst else {}
        h.fill(dataset=ds, x=[0.5, 1.5, 1.5, 3.5], weight=i + 1, **kw)
    return h


def test_group_histogram():
    h = make_hist()
    g = hist_utils.group_histogram(
        h, "dataset", "process", {"p1": ["a", "b"], "p2": ["c", "missing"],
                                  "p3": ["missing"]})
    assert list(g.axes["process"]) == ["p1", "p2", "p3"]
    np.testing.assert_allclose(g["p1", :].values(), [3, 6, 0, 3])
    np.testing.assert_allclose(g["p2", :].values(), [3, 6, 0, 3])
    np.testing.assert_allclose(g["p3", :].values(), 0)
    np.testing.assert_allclose(g["p1", :].variances(), [5, 10, 0, 5])


def test_group_histogram_bad_axis():
    with pytest.raises(ValueError):
        hist_utils.group_histogram(make_hist(), "nope", "p", {})


def test_rebin_int():
    h = make_hist()
    r = hist_utils.rebin_histogram(h, "x", 2)
    np.testing.assert_allclose(r["a", :].values(), [3, 1])


def test_rebin_edges():
    h = make_hist()
    r = hist_utils.rebin_histogram(h, "x", [0, 1, 4], new_axis_label="new")
    assert r.axes["x"].label == "new"
    np.testing.assert_allclose(r["b", :].values(), [2, 6])
    np.testing.assert_allclose(r["b", :].variances(), [4, 12])
    assert r.sum().value == pytest.approx(h.sum().value)


def test_rebin_subrange_keeps_flow():
    h = make_hist()
    r = hist_utils.rebin_histogram(h, "x", [1, 2, 3])
    vals = r["a", :].values(flow=True)
    # underflow gets [0,1), overflow gets [3,4) (+ original flows)
    np.testing.assert_allclose(vals, [1, 2, 0, 1])


def test_rebin_incompatible():
    with pytest.raises(ValueError):
        hist_utils.rebin_histogram(make_hist(), "x", [0, 1.5, 4])


def test_replace_missing_systematics():
    h = make_hist(syst=True)
    h.fill(dataset="a", sys="up", x=[0.5], weight=10)
    hist_utils.replace_missing_systematics(h)
    # "a" has its own up variation, kept
    np.testing.assert_allclose(h["a", "up", :].values(), [10, 0, 0, 0])
    # "b" had none, replaced by nominal
    np.testing.assert_allclose(h["b", "up", :].values(),
                               h["b", "nominal", :].values())


def test_replace_missing_systematics_no_sys():
    with pytest.raises(ValueError):
        hist_utils.replace_missing_systematics(make_hist())


def test_scale_histogram():
    h = make_hist()
    hist_utils.scale_histogram(h, "dataset", {"a": 2., "c": 0.5})
    np.testing.assert_allclose(h["a", :].values(), [2, 4, 0, 2])
    np.testing.assert_allclose(h["b", :].values(), [2, 4, 0, 2])
    np.testing.assert_allclose(h["c", :].values(), [1.5, 3, 0, 1.5])
    np.testing.assert_allclose(h["a", :].variances(), [4, 8, 0, 4])
