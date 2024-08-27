#!/usr/bin/env python3

import numpy as np
import uproot
import hjson

import pepper
from pepper.scale_factors import ScaleFactors, TauScaleFactors


class ConfigSTau(pepper.ConfigBasicPhysics):
    def __init__(self, path_or_file, textparser=hjson.load, cwd="."):
        """Initialize the configuration.

        Arguments:
        path_or_file -- Either a path to the file containing the configuration
                        or a file-like object of it
        textparser -- Callable to be used to parse the text contained in
                      path_or_file
        cwd -- A path to use as the working directory for relative paths in the
               config. The actual working directory of the process might change
        """
        super().__init__(path_or_file, textparser, cwd)

        self.behaviors.update(
            {
                "jet_fake_rate": self._get_jet_fake_rate,
                "muon_sf_trigger": self._get_singlemuon_trg_sfs,
                "DY_ZptLO_weights": self._get_dy_lo_sfs,
                "DY_jet_reweight": self._get_dy_jet_reweight,
                "W_jet_reweight": self._get_w_jet_reweight,
                "MET_trigger_sfs": self._get_trigger_sfs,
                "tauid_sfs": self._get_tauid_sfs
            }
        )
    
    def _get_jet_fake_rate(self, value):
        if not isinstance(value, list) or len(value) < 2:
            raise pepper.config.ConfigError(
                "scale factors needs to be list of 2-element-lists in "
                "form of [rootfile, histname]")
        with uproot.open(self._get_path(value[0])) as f:
            hist = f[value[1]]
        return ScaleFactors.from_hist(hist, value[2], setoverflow=False)
    
    def _get_singlemuon_trg_sfs(self, value):
        return self._get_muonscalefactor(value)

    def _get_dy_lo_sfs(self, value):
        if not isinstance(value, list) or len(value) < 2:
            raise pepper.config.ConfigError(
                "scale factors needs to be list of 2-element-lists in "
                "form of [rootfile, histname]")
        with uproot.open(self._get_path(value[0])) as f:
            hist = f[value[1]]
        return ScaleFactors.from_hist(hist, value[2], setoverflow=True)
    
    def _get_trigger_sfs(self, value):
        if not isinstance(value, list) or len(value) < 2:
            raise pepper.config.ConfigError(
                "scale factors needs to be list of 2-element-lists in "
                "form of [rootfile, histname]")
        with uproot.open(self._get_path(value[0])) as f:
            hist = f[value[1]]
        return ScaleFactors.from_hist(hist, value[2], setoverflow=True)
    
    def _get_dy_jet_reweight(self, value):
        return np.array(value[0], dtype=np.float64)
    
    def _get_w_jet_reweight(self, value):
        return np.array(value[0], dtype=np.float64)
    
    def _get_tauid_sfs(self, value):
        if not isinstance(value, list) or len(value) != 5:
            raise pepper.config.ConfigError(
                "tau id scale factors needs to be list of 5-element-lists in "
                "form of [rootfile, id, wpVSjet, wpVSe, type]")
        return TauScaleFactors(self._get_path(value[0]), *value[1:])