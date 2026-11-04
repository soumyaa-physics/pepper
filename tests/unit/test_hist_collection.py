import hist
import numpy as np
import pytest

from pepper.hist_collection import HistCollection


@pytest.fixture
def h():
    h = hist.Hist(
        hist.axis.StrCategory(["ds1", "ds2"], name="dataset", label="DS"),
        hist.axis.StrCategory(["ee", "mm"], name="channel"),
        hist.axis.Regular(3, 0, 3, name="x", label="X"),
        storage="Weight", name="h", label="Events")
    h.fill(dataset="ds1", channel="ee", x=[0.5, 1.5], weight=2.)
    h.fill(dataset="ds2", channel="mm", x=[2.5, 5.], weight=1.)
    return h


def assert_same(a, b, names=True):
    if names:
        assert a.axes.name == b.axes.name
    for ax_a, ax_b in zip(a.axes, b.axes):
        if isinstance(ax_a, hist.axis.StrCategory):
            # Order is not preserved for ROOT (stored as set)
            assert set(ax_a) == set(ax_b)
        else:
            np.testing.assert_allclose(ax_a.edges, ax_b.edges)
    cats = [ax.name for ax in a.axes
            if isinstance(ax, hist.axis.StrCategory)]
    for idx in HistCollection.get_cats_present(a):
        sa = a[dict(zip(cats, idx))]
        sb = b[dict(zip(cats, idx))]
        np.testing.assert_allclose(sa.values(flow=True), sb.values(flow=True))
        np.testing.assert_allclose(sa.variances(flow=True),
                                   sb.variances(flow=True))


@pytest.mark.parametrize("fmt,ext", [("hist", "coffea"), ("root", "root")])
def test_save_load_json_roundtrip(tmp_path, h, fmt, ext):
    coll = HistCollection(str(tmp_path), ["cut", "hist"], userdata={"u": 1})
    cats = HistCollection.get_cats_present(h)
    coll.save(("c1", "h"), h, f"c1_h.{ext}", fmt, cats)
    coll.save(("c2", "h"), h, f"c2_h.{ext}", fmt, cats)
    with open(tmp_path / "hists.json", "w") as f:
        coll.save_metadata_json(f)
    with open(tmp_path / "hists.json") as f:
        coll2 = HistCollection.from_json(f)
    assert coll2.userdata == {"u": 1}
    assert set(coll2) == {("c1", "h"), ("c2", "h")}
    loaded = coll2.load(("c1", "h"))
    # ROOT loses dense axis names, see test_root_keeps_dense_axis_names
    assert_same(loaded, h, names=fmt == "hist")
    assert loaded.pepper_cats_present == cats


def test_root_drops_absent_cats(tmp_path, h):
    coll = HistCollection(str(tmp_path), ["cut"])
    present = {("ds1", "ee"), ("ds2", "mm")}
    coll.save(("c",), h, "c.root", "root", present)
    loaded = coll.load(("c",))
    assert loaded.pepper_cats_present == present
    assert loaded.name == "h"
    assert loaded.label == "Events"
    assert_same(loaded, h, names=False)


@pytest.mark.xfail(strict=True, reason="get_hist_info only stores category "
                   "axes; dense axis names become 'xaxis'")
def test_root_keeps_dense_axis_names(tmp_path, h):
    coll = HistCollection(str(tmp_path), ["cut"])
    coll.save(("c",), h, "c.root", "root", HistCollection.get_cats_present(h))
    assert coll.load(("c",)).axes.name == h.axes.name


def test_from_single_hist(tmp_path, h):
    coll = HistCollection(str(tmp_path), ["cut"])
    coll.save(("c",), h, "single.coffea", "hist", set())
    single = HistCollection.from_single_hist(str(tmp_path / "single.coffea"))
    assert_same(single.load(()), h)


def test_save_errors(tmp_path, h):
    coll = HistCollection(str(tmp_path), ["cut", "hist"])
    with pytest.raises(ValueError):
        coll.save(("c",), h, "x", "hist", set())
    with pytest.raises(ValueError):
        coll.save(("c", 1), h, "x", "hist", set())
    with pytest.raises(ValueError):
        coll.save(("c", "h"), h, "x", "csv", set())


def make_coll():
    content = {("c1", "a"): "1", ("c1", "b"): "2", ("c2", "a"): "3"}
    return HistCollection("/p", ["cut", "hist"], content)


def test_getitem_full_and_dict():
    coll = make_coll()
    assert coll[("c1", "b")] == "2"
    assert coll[{"cut": "c2", "hist": "a"}] == "3"
    with pytest.raises(KeyError):
        coll[("c3", "a")]
    with pytest.raises(ValueError):
        coll[{"nope": "x"}]
    with pytest.raises(TypeError):
        coll["c1"]


def test_getitem_list_partial():
    sub = make_coll()[{"cut": ["c1"], "hist": None}]
    assert sub.key_fields == ["cut", "hist"]
    assert set(sub) == {("c1", "a"), ("c1", "b")}


@pytest.mark.xfail(strict=True, reason="Bug: string key parts after a None "
                   "iterate over the empty 'keys' instead of 'testkeys'")
def test_getitem_partial_with_string():
    sub = make_coll()[{"hist": "a"}]
    assert sub.key_fields == ["cut"]
    assert set(sub) == {("c1",), ("c2",)}


def test_add_and_copy():
    a = make_coll()
    b = HistCollection("/p", ["cut", "hist"], {("c9", "z"): "9"})
    c = a + b
    assert len(c) == 4 and len(a) == 3
    a += b
    assert len(a) == 4
    cp = a.copy()
    cp._content.clear()
    assert len(a) == 4
    with pytest.raises(ValueError):
        a + 1
