import awkward as ak
import h5py
import numpy as np
import pytest

from pepper.hdffile import HDF5File

VALUES = {
    "flat": ak.Array(np.arange(100.)),
    "jagged": ak.Array([[1., 2.], [], [3.]] * 20),
    "record": ak.Array({"a": np.arange(50), "b": [[1, 2]] * 50}),
    "option": ak.Array([1, None, 3] * 10),
    "strings": ak.Array(["x", "yy", "zzz"] * 10),
}


@pytest.mark.parametrize("name", VALUES)
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("compression", [
    "default", None, {"compression": "gzip", "compression_opts": 4}])
def test_roundtrip_awkward(tmp_path, name, packed, compression):
    kw = {} if compression == "default" else {"compression": compression}
    path = str(tmp_path / "f.h5")
    with HDF5File(path, "w", packed=packed, **kw) as f:
        f[name] = VALUES[name]
    with HDF5File(path, "r") as f:
        assert f[name].tolist() == VALUES[name].tolist()
        assert str(f[name].type) == str(VALUES[name].type)


def test_sliced_array_packed(tmp_path):
    arr = ak.Array([[1, 2, 3], [4], [5, 6]] * 100)[::7]
    path = str(tmp_path / "f.h5")
    with HDF5File(path, "w") as f:
        f["x"] = arr
    with HDF5File(path, "r") as f:
        assert f["x"].tolist() == arr.tolist()


def test_parameters_kept(tmp_path):
    arr = ak.with_parameter(ak.Array([1, 2, 3]), "foo", "bar")
    path = str(tmp_path / "f.h5")
    with HDF5File(path, "w") as f:
        f["x"] = arr
    with HDF5File(path, "r") as f:
        assert ak.parameters(f["x"]) == {"foo": "bar"}


def test_python_types(tmp_path):
    path = str(tmp_path / "f.h5")
    with HDF5File(path, "w") as f:
        f["np"] = np.arange(5.)
        f["str"] = "hello"
        f["list"] = [1, 2, 3]
        f["tuple"] = (1, 2)
        f["dict"] = {"a": 1, "b": "x"}
        with pytest.raises(ValueError):
            f["bad"] = object()
    with HDF5File(path, "r") as f:
        assert isinstance(f["np"], np.ndarray)
        np.testing.assert_array_equal(f["np"], np.arange(5.))
        assert f["str"] == "hello"
        assert f["list"] == [1, 2, 3]
        assert f["tuple"] == (1, 2)
        assert f["dict"] == {"a": 1, "b": "x"}
        assert set(f.keys()) == {"np", "str", "list", "tuple", "dict"}


def test_delete_and_h5py_object(tmp_path):
    path = str(tmp_path / "f.h5")
    with h5py.File(path, "w") as raw:
        f = HDF5File(raw)
        f["a"] = np.arange(3)
        f["b"] = np.arange(3)
        del f["a"]
        assert list(f.keys()) == ["b"]
        f.close()


def test_str_needs_mode(tmp_path):
    with pytest.raises(ValueError):
        HDF5File(str(tmp_path / "f.h5"))


@pytest.mark.xfail(strict=True, reason="Bug: __len__ has spurious 'key' arg")
def test_len(tmp_path):
    with HDF5File(str(tmp_path / "f.h5"), "w") as f:
        f["a"] = np.arange(3)
        assert len(f) == 1


@pytest.mark.xfail(strict=True, reason="Bug: __iter__ yields values, "
                   "Mapping protocol requires keys")
def test_iter_keys(tmp_path):
    with HDF5File(str(tmp_path / "f.h5"), "w") as f:
        f["a"] = np.arange(3)
        assert list(iter(f)) == ["a"]
