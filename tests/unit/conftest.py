import os
import subprocess
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TESTREPO = REPO / "tests" / "integration"
UL18_NANO = TESTREPO / "nanoaods_v9_ul18"
SCRIPTS = REPO / "scripts"


def _proxy_valid():
    proxy = os.environ.get("X509_USER_PROXY")
    if proxy is None or not os.path.isfile(proxy):
        return False
    try:
        res = subprocess.run(
            ["voms-proxy-info", "-exists", "-valid", "0:10",
             "-file", proxy], capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # No voms tools: trust the file
        return True
    return res.returncode == 0


def pytest_collection_modifyitems(config, items):
    skips = {
        "needs_grid": (lambda: _proxy_valid(),
                       "no valid grid proxy in X509_USER_PROXY"),
        "needs_cvmfs": (lambda: os.path.isdir("/cvmfs/cms-griddata.cern.ch"),
                        "/cvmfs/cms-griddata.cern.ch not available"),
        "needs_data": (lambda: any(UL18_NANO.glob("*.root")),
                       "test submodule data not checked out"),
    }
    cache = {}
    for item in items:
        for name, (check, reason) in skips.items():
            if name in item.keywords:
                if name not in cache:
                    cache[name] = check()
                if not cache[name]:
                    item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def events():
    """Small event array: 6 events with jagged muons/jets and channel flags"""
    muon_pt = [[50., 20.], [30.], [], [70., 40., 10.], [25.], [60., 35.]]
    jet_pt = [[100., 40.], [80.], [30., 20.], [], [200., 50., 30.], [45.]]
    return ak.Array({
        "Muon": ak.zip({
            "pt": muon_pt,
            "eta": [[0.1 * i for i, _ in enumerate(m)] for m in muon_pt],
        }),
        "Jet": ak.zip({
            "pt": jet_pt,
            "btag": [[i % 2 == 0 for i, _ in enumerate(j)] for j in jet_pt],
        }),
        "MET": ak.zip({"pt": [10., 20., 30., 40., 50., 60.]}),
        "is_ee": np.array([True, False, False, True, False, False]),
        "is_mm": np.array([False, True, False, False, True, True]),
        "is_em": np.array([False, False, True, False, False, False]),
    })


@pytest.fixture
def hist_config():
    """Factory for minimal HistDefinition configs"""
    def make(**kwargs):
        conf = {
            "bins": [{"name": "pt", "label": "pt", "n_or_arr": 10,
                      "lo": 0, "hi": 100}],
            "fill": {"pt": ["MET", "pt"]},
        }
        conf.update(kwargs)
        return conf
    return make
