import json

import awkward as ak
import correctionlib
import correctionlib.schemav2 as cs
import hist
import numpy as np
import pytest
import uproot

from pepper import scale_factors as sfs
from conftest import REPO, TESTREPO

PT_EDGES = [0., 50., 100., 1000.]


def binned_sf(name, sysname, values, inputs=("pt",)):
    """pt-binned correction with a categorical systematic input"""
    content = []
    for key, vals in values.items():
        content.append(cs.CategoryItem(key=key, value=cs.Binning(
            nodetype="binning", input="pt", edges=PT_EDGES, content=vals,
            flow="clamp")))
    return cs.Correction(
        name=name, version=1,
        inputs=[cs.Variable(name=sysname, type="string")]
        + [cs.Variable(name=i, type="real") for i in inputs],
        output=cs.Variable(name="sf", type="real"),
        data=cs.Category(nodetype="category", input=sysname,
                         content=content))


def multiply(name, inputs, factor):
    return cs.Correction(
        name=name, version=1,
        inputs=[cs.Variable(name=i, type="real") for i in inputs],
        output=cs.Variable(name="f", type="real"),
        data=cs.Formula(nodetype="formula", expression=str(factor),
                        parser="TFormula", variables=[inputs[0]]))


def jet_id(name):
    names = ["eta", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF",
             "chMultiplicity", "neMultiplicity", "multiplicity"]
    return cs.Correction(
        name=name, version=1,
        inputs=[cs.Variable(name=n, type="real") for n in names],
        output=cs.Variable(name="pass", type="real"),
        data=cs.Binning(nodetype="binning", input="neHEF",
                        edges=[0., 0.9, 1.], content=[1., 0.], flow="clamp"))


@pytest.fixture(scope="module")
def corrfile(tmp_path_factory):
    vals = {"nom": [1.0, 1.1, 1.2], "up": [1.1, 1.2, 1.3],
            "down": [0.9, 1.0, 1.1]}
    cset = cs.CorrectionSet(schema_version=2, corrections=[
        binned_sf("sf_valtype", "ValType", vals),
        binned_sf("sf_sys", "systematic", vals),
        binned_sf("sf_wp", "scale_factors", {
            k: [x * 2 for x in v] for k, v in vals.items()}),
        binned_sf("sf_eta", "ValType", vals, inputs=("eta", "pt")),
        binned_sf("sf_wpcat", "wp", vals),
        multiply("L1", ["JetPt", "JetEta"], 2.),
        multiply("L2", ["JetPt", "JetEta"], 3.),
        jet_id("AK4PUPPI_Tight"),
        jet_id("AK4PUPPI_TightLeptonVeto"),
    ])
    path = tmp_path_factory.mktemp("corr") / "corr.json"
    path.write_text(cset.model_dump_json(exclude_unset=True))
    return str(path)


SYSNAMING = {"central": "nom", "up": "up", "down": "down"}
PT = ak.Array([[10., 60.], [], [200.]])


# --- CorrLibSFs ---

def test_corrlib_variations(corrfile):
    sf = sfs.CorrLibSFs(SYSNAMING, corrfile, "sf_valtype", {})
    assert sf(pt=PT).tolist() == [[1.0, 1.1], [], [1.2]]
    assert sf("up", pt=PT).tolist() == [[1.1, 1.2], [], [1.3]]
    assert sf("down", pt=PT).tolist() == [[0.9, 1.0], [], [1.1]]
    with pytest.raises(ValueError):
        sf("sideways", pt=PT)
    with pytest.raises(ValueError):
        sf(eta=PT)
    assert sf.dimlabels == ["pt"]
    assert "sf_valtype" in repr(sf)


@pytest.mark.parametrize("name", ["sf_sys", "sf_wp"])
def test_corrlib_sys_input_names(corrfile, name):
    sf = sfs.CorrLibSFs(SYSNAMING, corrfile, name, {})
    scale = 2. if name == "sf_wp" else 1.
    np.testing.assert_allclose(ak.flatten(sf("up", pt=PT)),
                               np.array([1.1, 1.2, 1.3]) * scale)


@pytest.mark.xfail(strict=True, reason="dimlabels lists auto-filled "
                   "'systematic'/'weights' inputs")
def test_corrlib_dimlabels_systematic(corrfile):
    assert sfs.CorrLibSFs(SYSNAMING, corrfile, "sf_sys", {}).dimlabels == \
        ["pt"]


def test_corrlib_flat(corrfile):
    sf = sfs.CorrLibSFs(SYSNAMING, corrfile, "sf_valtype", {})
    np.testing.assert_allclose(sf(pt=ak.Array([10., 60.])), [1.0, 1.1])


def test_corrlib_add_args_fixed(corrfile):
    sf = sfs.CorrLibSFs(SYSNAMING, corrfile, "sf_wpcat", {"wp": "up"})
    assert sf.dimlabels == ["pt"]
    assert sf(pt=PT).tolist() == [[1.1, 1.2], [], [1.3]]


def test_corrlib_add_args_ranges(corrfile):
    """Mapping in add_args: different correction key per pt range"""
    sf = sfs.CorrLibSFs(SYSNAMING, corrfile, "sf_wpcat",
                        {"wp": {"up": (0, 50), "down": (50, 1e9)}})
    assert sf(pt=PT).tolist() == [[1.1, 1.0], [], [1.1]]


def test_corrlib_from_correctionset(corrfile):
    cset = correctionlib.CorrectionSet.from_file(corrfile)
    sf = sfs.CorrLibSFs(SYSNAMING, cset, "sf_valtype", {})
    assert sf(pt=PT).tolist() == [[1.0, 1.1], [], [1.2]]


def test_corrlib_errors(corrfile):
    with pytest.raises(Exception):
        sfs.CorrLibSFs(SYSNAMING, corrfile, "missing", {})
    with pytest.raises(ValueError):
        sfs.CorrLibSFs(SYSNAMING, 42, "sf_valtype", {})


def test_correctionlib_adapter(corrfile):
    adapter = sfs.CorrectionlibAdapter(corrfile, "sf_eta")
    # Scalars and jagged arrays are broadcast
    res = adapter(eta=ak.Array([0.5, 1.0, 1.5]), pt=PT)
    assert res.tolist() == [[1.0, 1.1], [], [1.2]]
    with pytest.raises(ValueError):
        sfs.CorrectionlibAdapter(corrfile, ["sf_eta"])


def test_compound_adapter(corrfile):
    comp = sfs.CorrectionlibCompoundAdapter(corrfile, ["L1", "L2"])
    pt = ak.Array([[10., 20.], [30.]])
    assert comp(JetPt=pt, JetEta=pt * 0).tolist() == [[6., 6.], [6.]]
    assert sfs.CorrectionlibCompoundAdapter(corrfile, "")(JetPt=pt) == 1.
    with pytest.raises(ValueError):
        comp(JetEta=pt)
    with pytest.raises(ValueError):
        sfs.CorrectionlibCompoundAdapter(corrfile, ["nope"])


def test_jet_id_producer(corrfile):
    jets = ak.Array([[{"eta": 0., "chHEF": .1, "neHEF": .5, "chEmEF": .1,
                       "neEmEF": .1, "muEF": 0., "chMultiplicity": 3,
                       "neMultiplicity": 2},
                      {"eta": 0., "chHEF": .1, "neHEF": .95, "chEmEF": .1,
                       "neEmEF": .1, "muEF": 0., "chMultiplicity": 3,
                       "neMultiplicity": 2}], []])
    tight, lepveto = sfs.JetIdProducer(corrfile, "AK4PUPPI").evaluate(jets)
    assert tight.tolist() == [[True, False], []]
    assert lepveto.tolist() == [[True, False], []]


@pytest.mark.needs_cvmfs
@pytest.mark.needs_data
def test_real_2024_corrections():
    """Smoke test the POG files referenced by the 2024 test config"""
    import hjson
    with open(TESTREPO / "configs/test_config_2024.json") as f:
        conf = hjson.load(f)
    paths = [v for v in _walk(conf) if isinstance(v, str)
             and v.startswith("/cvmfs") and v.endswith(".json.gz")]
    assert paths
    for path in paths:
        assert len(list(correctionlib.CorrectionSet.from_file(path))) > 0


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


# --- Histogram based ---

def sf_hist2d():
    h = hist.Hist(hist.axis.Variable([0, 50, 100], name="pt"),
                  hist.axis.Variable([0, 1.5, 2.5], name="abseta"),
                  storage="Weight")
    h[...] = np.stack([np.array([[1.0, 1.1], [1.2, 1.3]]),
                       np.full((2, 2), 0.01)], axis=-1)
    return h


@pytest.fixture
def root_sf(tmp_path):
    with uproot.recreate(tmp_path / "sf.root") as f:
        f["sf"] = sf_hist2d()
    return uproot.open(tmp_path / "sf.root")


def test_scalefactors_from_hist(root_sf):
    sf = sfs.ScaleFactors.from_hist(root_sf["sf"], ["pt", "abseta"])
    assert list(sf.dimlabels) == ["pt", "abseta"]
    pt = ak.Array([[10., 70.], [], [500.]])
    eta = ak.Array([[0.5, 2.], [], [0.5]])
    assert sf(pt=pt, abseta=eta).tolist() == [[1.0, 1.3], [], [1.0]]
    up = sf(pt=pt, abseta=eta, variation="up")
    np.testing.assert_allclose(ak.flatten(up), [1.1, 1.4, 1.0])
    np.testing.assert_allclose(sf(pt=np.array([60.]), abseta=np.array([0.])),
                               [1.2])
    with pytest.raises(ValueError):
        sf(pt=pt)
    with pytest.raises(ValueError):
        sf(pt=pt, abseta=eta, variation="x")
    with pytest.raises(ValueError):
        sfs.ScaleFactors.from_hist(root_sf["sf"], ["pt"])


def test_scalefactors_validation():
    with pytest.raises(ValueError):
        sfs.ScaleFactors(np.ones(2), None, None, {"variation": [0, 1, 2]})
    with pytest.raises(ValueError):
        sfs.ScaleFactors(np.ones(2), None, None, {"x": [0, 1]})
    with pytest.raises(ValueError):
        sfs.ScaleFactors(np.ones(2), np.ones(3), np.ones(2),
                         {"x": [0, 1, 2]})


def test_muon_scale_factor():
    def make(v):
        return sfs.ScaleFactors(np.array([1., v, 1.]), np.array([1., v + 1,
                                1.]), np.array([1., v - 1, 1.]),
                                {"pt": [-np.inf, 0, 100, np.inf]})
    msf = sfs.MuonScaleFactor(make(2.), make(20.), make(200.))
    pt = np.array([50.])
    assert msf(pt=pt)[0] == 2.
    assert msf("stat up", pt=pt)[0] == 21.
    assert msf("syst down", pt=pt)[0] == 199.
    with pytest.raises(ValueError):
        msf("bad", pt=pt)
    assert list(msf.dimlabels) == ["pt"]


def test_pileup_weighter(tmp_path):
    with uproot.recreate(tmp_path / "pu.root") as f:
        for suffix, val in [("", 1.), ("_up", 2.), ("_down", 0.5)]:
            h = hist.Hist(hist.axis.Regular(10, 0, 100, name="ntrueint"),
                          storage="Weight")
            h[...] = np.stack([np.full(10, val), np.zeros(10)], axis=-1)
            f["all_datasets" + suffix] = h
    with uproot.open(tmp_path / "pu.root") as f:
        pu = sfs.PileupWeighter(f)
    n = np.array([5., 50.])
    np.testing.assert_allclose(pu("any", n), 1.)
    np.testing.assert_allclose(pu("any", n, "up"), 2.)
    np.testing.assert_allclose(pu("any", n, "down"), 0.5)
    with pytest.raises(ValueError):
        pu("any", n, "x")


def test_pileup_weighter_missing(tmp_path):
    with uproot.recreate(tmp_path / "pu.root") as f:
        f["ds"] = hist.Hist(hist.axis.Regular(1, 0, 1, name="n"),
                            storage="Weight")
    with uproot.open(tmp_path / "pu.root") as f, pytest.raises(ValueError):
        sfs.PileupWeighter(f)


def test_top_pt_weighter():
    w = sfs.TopPtWeigter("datanlo", a=0.1, b=-0.001)
    pt = np.array([0., 100.])
    np.testing.assert_allclose(w(pt, pt), np.exp(0.1 - 0.001 * pt))
    w = sfs.TopPtWeigter("theory", scale=2., a=1., b=0., c=0., d=0.)
    np.testing.assert_allclose(w(pt, pt), 2.)
    with pytest.raises(ValueError):
        sfs.TopPtWeigter("bad")


def test_btag_wps():
    wp_dict = {"deepjet": {"name": "DeepJet",
                           "wpnames": {"loose": "l", "medium": "m",
                                       "tight": "t"},
                           "wps": {"2018": [0.05, 0.27, 0.73],
                                   "2024": [0.05, 0.3]}}}
    wps = sfs.BTagWPs(wp_dict, "deepjet", "2018")
    assert wps("Medium") == ("M", 1, 0.27)
    assert wps.name == "DeepJet"
    assert list(sfs.BTagWPs(wp_dict, "deepjet", "2024").wpnames) == \
        ["loose", "medium"]
    with pytest.raises(ValueError):
        sfs.BTagWPs(wp_dict, "deepjet", "2024")("tight")


# --- BTagWeighter ---

BTAG_WP_DICT = {"deepjet": {
    "name": "deepJet", "wpnames": {"loose": "l", "medium": "m", "tight": "t"},
    "wps": {"ul2018": [0.1, 0.5, 0.9], "2024": [0.1, 0.5, 0.9]}}}
BTAG_SOURCES = ["jes", "pileup", "statistic", "type3"]
# Per flavour: (sf central, up, down)
BTAG_SF = {0: (0.9, 1.0, 0.8), 4: (1.1, 1.3, 0.9), 5: (1.2, 1.3, 1.1)}
BTAG_EFF = {0: 0.1, 4: 0.3, 5: 0.7}
# Event 0: tagged b + untagged light, event 1: untagged c, event 2: none
JF = ak.Array([[5, 0], [4], []])
ETA = ak.Array([[0.5, -1.], [2.], []])
BPT = ak.Array([[60., 40.], [150.], []])
DISCR = ak.Array([[0.8, 0.2], [0.3], []])


def btag_json_correction(name, flavors, nsys=True):
    def by_flav(i):
        return cs.Category(nodetype="category", input="flavor", content=[
            cs.CategoryItem(key=f, value=BTAG_SF[f][i]) for f in flavors])
    systs = {"central": 0, "up": 1, "down": 2}
    if nsys:
        for s in BTAG_SOURCES + ["correlated", "uncorrelated"]:
            systs[f"up_{s}"] = 1
            systs[f"down_{s}"] = 2
    return cs.Correction(
        name=name, version=1,
        inputs=[cs.Variable(name="systematic", type="string"),
                cs.Variable(name="working_point", type="string"),
                cs.Variable(name="flavor", type="int"),
                cs.Variable(name="abseta", type="real"),
                cs.Variable(name="pt", type="real")],
        output=cs.Variable(name="weight", type="real"),
        data=cs.Category(nodetype="category", input="systematic", content=[
            cs.CategoryItem(key=s, value=cs.Category(
                nodetype="category", input="working_point", content=[
                    cs.CategoryItem(key=wp, value=by_flav(i))
                    for wp in "LMT"]))
            for s, i in systs.items()]))


@pytest.fixture(scope="module")
def btag_eff(tmp_path_factory):
    h = hist.Hist(hist.axis.Variable([0, 4, 5, 6], name="flav"),
                  hist.axis.Variable([20., 1000.], name="pt"),
                  hist.axis.Variable([0., 2.5], name="abseta"))
    h.view()[:] = np.array([BTAG_EFF[f] for f in (0, 4, 5)])[:, None, None]
    path = tmp_path_factory.mktemp("btag") / "eff.root"
    with uproot.recreate(path) as f:
        f["central"] = h
    return str(path)


@pytest.fixture(scope="module")
def btag_json(tmp_path_factory):
    def make(light_name):
        cset = cs.CorrectionSet(schema_version=2, corrections=[
            btag_json_correction("deepJet_mujets", [4, 5]),
            btag_json_correction(light_name, [0], nsys=False)])
        path = tmp_path_factory.mktemp("btag") / f"{light_name}.json"
        path.write_text(cset.model_dump_json(exclude_unset=True))
        return str(path)
    return {"ul2018": make("deepJet_incl"), "2024": make("deepJet_light")}


@pytest.fixture(scope="module")
def btag_csv(tmp_path_factory):
    btvflav = {5: 0, 4: 1, 0: 2}
    lines = ["DeepJet;OperatingPoint, measurementType, sysType, jetFlavor, "
             "etaMin, etaMax, ptMin, ptMax, discrMin, discrMax, formula"]
    for wp in range(3):
        for flav, (c, u, d) in BTAG_SF.items():
            mtype = "incl" if flav == 0 else "mujets"
            for sys_, val in (("central", c), ("up", u), ("down", d)):
                lines.append(f"{wp}, {mtype}, {sys_}, {btvflav[flav]}, 0, "
                             f"2.5, 20, 1000, 0, 1, \"{val}+0*x\"")
    path = tmp_path_factory.mktemp("btag") / "sf.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def expected_btag_weight(wp_val, sf_override={}):
    out = []
    for jfs, ds in zip(JF.tolist(), DISCR.tolist()):
        p_mc = p_data = 1.
        for f, d in zip(jfs, ds):
            eff = BTAG_EFF[f]
            sf = sf_override.get(f, BTAG_SF[f][0])
            tagged = d > wp_val
            p_mc *= eff if tagged else 1 - eff
            p_data *= sf * eff if tagged else 1 - sf * eff
        out.append(p_data / p_mc)
    return np.array(out)


def make_btag_weighter(sf, eff, year="ul2018", **kwargs):
    wps = sfs.BTagWPs(BTAG_WP_DICT, "deepjet", year)
    return sfs.BTagWeighter(sf, eff, "deepjet", year, wps, **kwargs)


@pytest.mark.parametrize("year", ["ul2018", "2024"])
def test_btag_weighter_json(btag_json, btag_eff, year):
    w = make_btag_weighter(btag_json[year], btag_eff, year)
    assert w.filetype == "json"
    assert sorted(w.sources) == BTAG_SOURCES
    assert "central" in w.available_efficiencies
    np.testing.assert_allclose(
        w("medium", JF, ETA, BPT, DISCR), expected_btag_weight(0.5))
    # Tight: the b jet is no longer tagged
    np.testing.assert_allclose(
        w("tight", JF, ETA, BPT, DISCR), expected_btag_weight(0.9))


@pytest.mark.parametrize("variation,idx,flavors", [
    ("heavy up", 1, (4, 5)), ("heavy down", 2, (4, 5)),
    ("light up", 1, (0,)), ("light down", 2, (0,))])
def test_btag_weighter_json_variations(btag_json, btag_eff, variation, idx,
                                       flavors):
    w = make_btag_weighter(btag_json["ul2018"], btag_eff)
    res = w("medium", JF, ETA, BPT, DISCR, variation=variation)
    override = {f: BTAG_SF[f][idx] for f in flavors}
    np.testing.assert_allclose(res, expected_btag_weight(0.5, override))
    assert res[2] == 1.


def test_btag_weighter_json_errors(btag_json, btag_eff):
    w = make_btag_weighter(btag_json["ul2018"], btag_eff)
    with pytest.raises(ValueError):
        w("medium", JF, ETA, BPT, DISCR, variation="other up")
    with pytest.raises(ValueError):
        w("superloose", JF, ETA, BPT, DISCR)
    with pytest.raises(NotImplementedError):
        make_btag_weighter(btag_json["ul2018"], btag_eff,
                           method="iterativefit")
    with pytest.raises(ValueError):
        make_btag_weighter(btag_json["ul2018"], btag_eff, method="bad")


def test_btag_weighter_csv(btag_csv, btag_eff):
    w = make_btag_weighter(btag_csv, btag_eff)
    assert w.filetype == "csv"
    assert sorted(w.sources) == BTAG_SOURCES
    np.testing.assert_allclose(
        w("medium", JF, ETA, BPT, DISCR), expected_btag_weight(0.5))
    np.testing.assert_allclose(
        w("medium", JF, ETA, BPT, DISCR, variation="heavy up"),
        expected_btag_weight(0.5, {4: 1.3, 5: 1.3}))
    np.testing.assert_allclose(
        w("medium", JF, ETA, BPT, DISCR, variation="light down"),
        expected_btag_weight(0.5, {0: 0.8}))


# Real POG files: check that all variations used by ProcessorBasicPhysics
# evaluate
BTV = "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/"
BTAG_RUN2 = BTV + "Run2-2018-UL-NanoAODv9/latest/btagging.json.gz"
BTAG_RUN3 = (BTV + "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/"
             "latest/btagging.json.gz")
REAL_JF = ak.Array([[5, 4, 0], [0, 5], [], [4]])
REAL_ETA = ak.Array([[0.3, -1.2, 2.1], [-0.4, 1.8], [], [0.9]])
REAL_PT = ak.Array([[45., 80., 30.], [250., 35.], [], [120.]])
REAL_DISCR = ak.Array([[0.95, 0.4, 0.01], [0.6, 0.2], [], [0.05]])


def real_btag_weighter(path, tagger, year, btag_eff):
    with open(REPO / "example/btag_wps.json") as f:
        wps = sfs.BTagWPs(json.load(f), tagger, year)
    return sfs.BTagWeighter(path, btag_eff, tagger, year, wps)


def eval_btag_variations(w, splits):
    central = w("medium", REAL_JF, REAL_ETA, REAL_PT, REAL_DISCR)
    assert np.all(np.isfinite(central)) and np.all(central > 0)
    assert central[2] == 1. and not np.allclose(central, 1.)
    for split in splits:
        for kind in ("heavy", "light"):
            for d in ("up", "down"):
                var = w("medium", REAL_JF, REAL_ETA, REAL_PT, REAL_DISCR,
                        variation=f"{kind} {d}{split}")
                assert np.all(np.isfinite(var)) and np.all(var > 0), \
                    f"{kind} {d}{split}"


SCHEME_SPLITS = ["", "_correlated", "_uncorrelated"]


@pytest.mark.needs_cvmfs
def test_real_btag_sf_run2(btag_eff):
    w = real_btag_weighter(BTAG_RUN2, "deepjet", "ul2018", btag_eff)
    eval_btag_variations(w, SCHEME_SPLITS)
    for s in w.sources:
        for d in ("up", "down"):
            w("medium", REAL_JF, REAL_ETA, REAL_PT, REAL_DISCR,
              variation=f"heavy {d}_{s}")


@pytest.mark.needs_cvmfs
def test_real_btag_sf_run3(btag_eff):
    w = real_btag_weighter(BTAG_RUN3, "upart", "2024", btag_eff)
    eval_btag_variations(w, SCHEME_SPLITS)


@pytest.mark.needs_cvmfs
@pytest.mark.xfail(strict=True, reason="Bug: hardcoded mujets sources "
                   "include 'jes', missing in the Run 3 UParT file")
def test_real_btag_sf_run3_sources(btag_eff):
    w = real_btag_weighter(BTAG_RUN3, "upart", "2024", btag_eff)
    for s in w.sources:
        w("medium", REAL_JF, REAL_ETA, REAL_PT, REAL_DISCR,
          variation=f"heavy up_{s}")


# Real POG files loaded through ConfigBasicPhysics, called like
# ProcessorBasicPhysics does
GRID = "/cvmfs/cms-griddata.cern.ch/cat/metadata/"
RUN2 = "Run2-2018-UL-NanoAODv9/latest/"
RUN3 = "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/"


def jme_config(path, version, jr, jettype):
    def levels(dtype, lvls):
        return [f"{version}_{dtype}_{lvl}_{jettype}" for lvl in lvls]
    return {
        "path": path,
        "jet_correction_mc": levels(
            "MC", ["L1FastJet", "L2Relative", "L3Absolute"]),
        "jet_correction_data": levels(
            "DATA", ["L1FastJet", "L2Relative", "L3Absolute",
                     "L2L3Residual"]),
        "jet_uncertainty_template": f"{version}_MC_[UNC]_{jettype}",
        "jet_uncertainty": ["Total", "Regrouped_Total"],
        "jet_resolution": f"{jr}_MC_PtResolution_{jettype}",
        "jet_ressf": f"{jr}_MC_ScaleFactor_{jettype}",
        "jet_ressf_uncertainty": f"{jr}_MC_SFUncertainty_{jettype}",
    }


REAL_CONFIGS = {
    "run2": {
        "year": "ul2018",
        "jme_correctionlib_corrections": jme_config(
            GRID + "JME/" + RUN2 + "jet_jerc.json.gz", "Summer19UL18_V5",
            "Summer19UL18_JRV3", "AK4PFchs"),
        "pileup_reweighting": [GRID + "LUM/" + RUN2 + "puWeights.json.gz",
                               "Collisions18_UltraLegacy_goldenJSON"],
        "muon_sf": [[GRID + "MUO/" + RUN2 + "muon_Z.json.gz",
                     "NUM_TightID_DEN_TrackerMuons", {}],
                    [GRID + "MUO/" + RUN2 + "muon_Z.json.gz",
                     "NUM_TightRelIso_DEN_TightIDandIPCut", {}]],
    },
    "run3": {
        "year": "2024",
        "jme_correctionlib_corrections": jme_config(
            GRID + "JME/" + RUN3 + "jet_jerc.json.gz", "Summer24Prompt24_V5",
            "Summer24Prompt24_JRV2", "AK4PFPuppi"),
        "pileup_reweighting": [
            GRID + "LUM/" + RUN3 + "puWeights_BCDEFGHI.json.gz",
            "Collisions24_BCDEFGHI_goldenJSON"],
        "muon_sf": [[GRID + "MUO/" + RUN3 + "muon_Z.json.gz",
                     "NUM_TightID_DEN_TrackerMuons", {}],
                    [GRID + "MUO/" + RUN3 + "muon_Z.json.gz",
                     "NUM_TightPFIso_DEN_TightID", {}]],
        "jet_ids": [GRID + "JME/" + RUN3 + "jetid.json.gz", "AK4PUPPI"],
    },
}
RUN_NUMBERS = {"run2": 320000, "run3": 380000}


@pytest.fixture(scope="module", params=["run2", "run3"])
def real_config(request, tmp_path_factory):
    from pepper import ConfigBasicPhysics
    path = tmp_path_factory.mktemp("conf") / f"{request.param}.json"
    path.write_text(json.dumps(REAL_CONFIGS[request.param]))
    return request.param, ConfigBasicPhysics(str(path))


JET_PT = ak.Array([[150., 45.], [], [30., 25., 300.]])
JET_ETA = ak.Array([[0.3, -2.1], [], [1.4, 3.2, -0.8]])
JET_PHI = ak.Array([[0.1, -2.], [], [3., 1., -1.]])
JET_AREA = ak.full_like(JET_PT, 0.5)
RHO = ak.Array([20., 15., 35.])


def assert_jagged_finite(arr, lo, hi):
    flat = ak.to_numpy(ak.flatten(arr))
    assert len(flat) == 5
    assert np.all(np.isfinite(flat)) and np.all((flat > lo) & (flat < hi))


@pytest.mark.needs_cvmfs
def test_real_jec(real_config):
    run, conf = real_config
    jme = conf["jme_correctionlib_corrections"]
    runs = np.full(3, RUN_NUMBERS[run])
    kwargs = dict(JetPt=JET_PT, JetEta=JET_ETA, JetPhi=JET_PHI, JetA=JET_AREA,
                  Rho=RHO, run=runs)
    mc = jme["jet_correction_mc"](**kwargs)
    data = jme["jet_correction_data"](**kwargs)
    assert_jagged_finite(mc, 0.2, 3.)
    assert_jagged_finite(data, 0.2, 3.)
    assert ak.all(ak.num(mc) == ak.num(JET_PT))
    assert not np.allclose(ak.flatten(mc), ak.flatten(data))


@pytest.mark.needs_cvmfs
def test_real_jec_uncertainty(real_config):
    _, conf = real_config
    juncs = conf["jme_correctionlib_corrections"]["jet_uncertainty"]
    assert set(juncs) == {"Total", "Regrouped_Total"}
    for junc in juncs.values():
        assert_jagged_finite(junc(JetPt=JET_PT, JetEta=JET_ETA), 0., 0.3)


@pytest.mark.needs_cvmfs
def test_real_jer(real_config):
    _, conf = real_config
    jme = conf["jme_correctionlib_corrections"]
    res = jme["jet_resolution"](JetPt=JET_PT, JetEta=JET_ETA, Rho=RHO)
    assert_jagged_finite(res, 0., 1.)
    sf = jme["jet_ressf"](JetPt=JET_PT, JetEta=JET_ETA)
    unc = jme["jet_ressf_uncertainty"](JetPt=JET_PT, JetEta=JET_ETA)
    assert_jagged_finite(sf, 0.5, 2.)
    assert_jagged_finite(unc, 0., 0.5)


JER_OLD = (GRID + "JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/"
           "2025-12-02/jet_jerc.json.gz")


@pytest.mark.needs_cvmfs
@pytest.mark.parametrize("fmt", ["sf_systematic_input", "sf_uncertainty"])
def test_real_compute_jer_factor(fmt):
    """JER variations in ProcessorBasicPhysics.compute_jer_factor for both
    JER SF formats (systematic input vs separate SFUncertainty)"""
    from types import SimpleNamespace
    from pepper import ProcessorBasicPhysics
    if fmt == "sf_systematic_input":
        path, jr = JER_OLD, "Summer23BPixPrompt23_RunD_JRV1"
    else:
        path, jr = GRID + "JME/" + RUN3 + "jet_jerc.json.gz", \
            "Summer24Prompt24_JRV2"
    jme = {"jet_resolution": sfs.CorrectionlibAdapter(
               path, f"{jr}_MC_PtResolution_AK4PFPuppi"),
           "jet_ressf": sfs.CorrectionlibAdapter(
               path, f"{jr}_MC_ScaleFactor_AK4PFPuppi")}
    if fmt == "sf_uncertainty":
        jme["jet_ressf_uncertainty"] = sfs.CorrectionlibAdapter(
            path, f"{jr}_MC_SFUncertainty_AK4PFPuppi")
    proc = SimpleNamespace(config={"jme_correctionlib_corrections": jme})
    data = ak.zip({"Jet": ak.zip({"pt": JET_PT, "eta": JET_ETA}),
                   "fixedGridRhoFastjetAll": RHO}, depth_limit=1)
    dev = {}
    for v in ("central", "up", "down"):
        factor = ProcessorBasicPhysics.compute_jer_factor(
            proc, data, np.random.default_rng(1), "Jet", v, hybrid=False)
        dev[v] = abs(ak.to_numpy(ak.flatten(factor)) - 1)
    # Same random numbers: smearing grows with the SF
    assert np.all(dev["up"] > dev["central"])
    assert np.all(dev["central"] > dev["down"])


@pytest.mark.needs_cvmfs
def test_real_pileup(real_config):
    _, conf = real_config
    weighter = conf["pileup_reweighting"]
    ntrueint = ak.Array([5., 20., 35., 60.])
    w = {v: weighter(dsname="TT", NumTrueInteractions=ntrueint, variation=v)
         for v in ("central", "up", "down")}
    assert np.all(np.isfinite(w["central"])) and np.all(w["central"] >= 0)
    assert not np.allclose(w["central"], 1.)
    assert not np.allclose(w["up"], w["down"])


@pytest.mark.needs_cvmfs
def test_real_muon_sf(real_config):
    _, conf = real_config
    muons = ak.zip({"pt": ak.Array([[30., 80.], [], [25.]]),
                    "eta": ak.Array([[0.5, -2.2], [], [1.3]])})
    for sffunc in conf["muon_sf"]:
        params = {k: abs(muons.eta) if k == "abseta" else muons[k]
                  for k in sffunc.dimlabels}
        sf = {v: ak.prod(sffunc(**params, variation=v), axis=1)
              for v in ("central", "up", "down")}
        assert np.all(np.isfinite(sf["central"]))
        assert np.all((sf["central"] > 0.8) & (sf["central"] < 1.2))
        assert sf["central"][1] == 1.
        assert np.all(sf["up"] >= sf["central"])
        assert np.all(sf["down"] <= sf["central"])
        assert np.any(sf["up"] > sf["down"])


@pytest.mark.needs_cvmfs
def test_real_jet_id(tmp_path):
    from pepper import ConfigBasicPhysics
    path = tmp_path / "conf.json"
    path.write_text(json.dumps(REAL_CONFIGS["run3"]))
    producer = ConfigBasicPhysics(str(path))["jet_ids"]
    # good, high neutral hadron fraction | high muon fraction, good
    jets = ak.zip({
        "eta": [[0.5, 0.5], [1.0, -2.0]],
        "chHEF": [[0.5, 0.0], [0.5, 0.5]],
        "neHEF": [[0.2, 0.99], [0.2, 0.2]],
        "chEmEF": [[0.1, 0.0], [0.1, 0.1]],
        "neEmEF": [[0.1, 0.01], [0.1, 0.1]],
        "muEF": [[0.0, 0.0], [0.9, 0.0]],
        "chMultiplicity": [[10, 0], [10, 10]],
        "neMultiplicity": [[5, 1], [5, 5]],
    })
    tight, tight_lep_veto = producer.evaluate(jets)
    assert tight.tolist() == [[True, False], [True, True]]
    assert tight_lep_veto.tolist() == [[True, False], [False, True]]
