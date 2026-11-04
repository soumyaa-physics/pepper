import os

import awkward as ak
import numpy as np

import pepper


class Processor(pepper.Processor):
    """Minimal processor for end-to-end tests"""

    def process_selection(self, selector, dsname, is_mc, filler):
        if os.environ.get("MINIPROC_FAIL") == dsname:
            raise RuntimeError("Requested failure")
        selector.add_cut("Trigger", lambda d: d.HLT[
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL"])
        selector.set_column(
            "Muon", lambda d: d.Muon[(d.Muon.pt > 20) & d.Muon.tightId])
        selector.add_cut("HasMuon", lambda d: ak.num(d.Muon) > 0)
        selector.set_column("is_1mu", lambda d: ak.num(d.Muon) == 1)
        selector.set_column("is_2mu", lambda d: ak.num(d.Muon) >= 2)
        selector.set_cat("nmu", {"is_1mu", "is_2mu"})
        if is_mc:
            n = selector.num
            selector.set_systematic(
                "toy", np.full(n, 1.1), np.full(n, 0.9))
        selector.add_cut(
            "MET", lambda d: d.MET.pt > 30,
            categories={"nmu": ["is_1mu", "is_2mu"]})
