#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package plotter
Plotter for searchingfornues TTree

This module produces plot from the TTree produced by the
searchingfornues framework (https://github.com/ubneutrinos/searchingfornues)

Example:
    my_plotter = plotter.Plotter(samples, weights)
    fig, ax1, ax2 = my_plotter.plot_variable(
        "reco_e",
        query="selected == 1"
        kind="event_category",
        title="$E_{deposited}$ [GeV]",
        bins=20,
        range=(0, 2)
    )

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
"""

import math
import warnings
import bisect

from collections import defaultdict
from collections.abc import Iterable
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


category_labels = {
    1: r"$\nu_e$ CC",
    10: r"$\nu_e$ CC0$\pi$0p",
    11: r"$\nu_e$ CC0$\pi$Np",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out. fid. vol.",
    6: r"other",
    0: r"No slice"
}

pdg_labels = {
    2212: r"$p$",
    13: r"$\mu$",
    11: r"$e$",
    111: r"$\pi^0$",
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi^{\pm}$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic"
}

int_labels = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    3: "Coherent",
    4: "Coherent Elastic",
    5: "Electron scatt.",
    6: "IMDAnnihilation",
    7: r"Inverse $\beta$ decay",
    8: "Glashow resonance",
    9: "AMNuGamma",
    10: "MEC",
    11: "Diffractive",
    12: "EM",
    13: "Weak Mix"
}


int_colors = {
    0: "bisque",
    1: "darkorange",
    2: "goldenrod",
    3: "lightcoral",
    4: "forestgreen",
    5: "turquoise",
    6: "teal",
    7: "deepskyblue",
    8: "steelblue",
    9: "royalblue",
    10: "crimson",
    11: "mediumorchid",
    12: "magenta",
    13: "pink",
    111: "black"
}


category_colors = {
    4: "xkcd:light red",
    5: "xkcd:brick",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    3: "xkcd:cobalt",
    31: "xkcd:sky blue",
    1: "xkcd:green",
    10: "xkcd:mint green",
    11: "xkcd:lime green",
    111: "xkcd:goldenrod",
    6: "xkcd:grey",
    0: "xkcd:black"
}

pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "#e31a1c",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "#cab2d6",
}


class Plotter:
    """Main plotter class

    Args:
        samples (dict): Dictionary of pandas dataframes.
            mc`, `nue`, `data`, and `ext` are required. `lee` and `dirt` are optional.
        weights (dict): Dictionary of global dataframes weights.
            One for each entry in the samples dict.
        pot (int): Number of protons-on-target. Defaults is 4.5e19.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    def __init__(self, samples, weights, pot=4.5e19,):
        self.weights = weights
        self.samples = samples
        self.pot = pot
        self.significance = 0
        self.significance_likelihood = 0

        if "dirt" not in samples:
            warnings.warn("Missing dirt sample")

        necessary = ["category", "shr_pfp_id_v", "selected",  # "trk_pfp_id",
                     "backtracked_pdg", "nu_pdg", "ccnc", "trk_bkt_pdg", "shr_bkt_pdg"]

        missing = np.setdiff1d(necessary, samples["mc"].columns)

        if missing.size > 0:
            raise ValueError(
                "Missing necessary columns in the DataFrame: %s" % missing)

    @staticmethod
    def _sigma_calc_likelihood(sig, bkg, err_bkg, scale_factor=1):
        """It calculates the significance with the profile likelihood ratio
        assuming an uncertainity on the background entries.
        Taken from http://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
        """
        b = bkg * scale_factor
        if not isinstance(err_bkg, Iterable):
            e = np.array([err_bkg]) * scale_factor
        else:
            e = err_bkg * scale_factor

        s = sig * scale_factor

        p1 = (s+b)*np.log((s+b)*(b+e**2)/(b**2+(s+b)*e**2))

        p2 = -s
        if sum(e) > 0:
            p2 = -b**2/(e**2)*np.log(1+e**2*s/(b*(b+e**2)))
        z = 2*(p1+p2)

        return math.sqrt(sum(z))

    @staticmethod
    def _sigma_calc_matrix(signal, background, scale_factor=1, cov=None):
        """It calculates the significance as the square root of the Δχ2 score

        Args:
            signal (np.array): array of signal histogram
            background (np.array): array of background histogram
            scale_factor (float, optional): signal and background scaling factor.
                Default is 1

        Returns:
            Square root of S•B^(-1)•S^T
        """

        bkg_array = background * scale_factor
        empty_elements = np.where(bkg_array == 0)[0]
        sig_array = signal * scale_factor
        cov = cov * scale_factor * scale_factor
        sig_array = np.delete(sig_array, empty_elements)
        bkg_array = np.delete(bkg_array, empty_elements)
        cov[np.diag_indices_from(cov)] += bkg_array
        emtxinv = np.linalg.inv(cov)
        chisq = float(sig_array.dot(emtxinv).dot(sig_array.T))

        return np.sqrt(chisq)

    @staticmethod
    def _ratio_err(num, den, num_err, den_err):
        n, d, n_e, d_e = num, den, num_err, den_err
        n[n == 0] = 0.00001
        d[d == 0] = 0.00001
        return np.array([
            n[i] / d[i] * math.sqrt((n_e[i] / n[i])**2 + (d_e[i] / d[i])**2)
            for i, k in enumerate(num)
        ])

    @staticmethod
    def _is_fiducial(x, y, z):
        try:
            x_1 = x[:, 0] > 10
            x_2 = x[:, 1] > 10
            y_1 = y[:, 0] > 15
            y_2 = y[:, 1] > 15
            z_1 = z[:, 0] > 10
            z_2 = z[:, 1] > 50

            return x_1 & x_2 & y_1 & y_2 & z_1 & z_2
        except IndexError:
            return True

    @staticmethod
    def _chisquare(data, mc, err_data, err_mc):
        num = (data - mc)**2
        den = err_mc**2
        if np.count_nonzero(data):
            return sum(num / den) / np.count_nonzero(data)
        return np.inf

    def _select_showers(self, variable, variable_name, sample, query="selected==1", score=0.5, extra_cut=None):
        variable = variable.ravel()

        if variable.size > 0:
            if isinstance(variable[0], np.ndarray):
                variable = np.hstack(variable)
                if "shr" in variable_name and variable_name != "shr_score_v":
                    shr_score = np.hstack(self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut))
                    shr_score_id = shr_score < score
                    variable = variable[shr_score_id]
                elif "trk" in variable_name and variable_name != "trk_score_v":
                    trk_score = np.hstack(self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut))
                    trk_score_id = trk_score >= score
                    variable = variable[trk_score_id]

        return variable

    def _selection(self, variable, sample, query="selected==1", extra_cut=None):
        sel_query = query

        if extra_cut is not None:
            sel_query += "& %s" % extra_cut

        return sample.query(sel_query).eval(variable).ravel()

    def _categorize_entries_pdg(self, sample, variable, query="selected==1", extra_cut=None):

        if "trk" in variable:
            pfp_id_variable = "trk_pfp_id"
            score_v = self._selection("trk_score_v", sample, query=query, extra_cut=extra_cut)
        else:
            pfp_id_variable = "shr_pfp_id_v"
            score_v = self._selection("shr_score_v", sample, query=query, extra_cut=extra_cut)


        pfp_id = self._selection(
            pfp_id_variable, sample, query=query, extra_cut=extra_cut)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)
        pfp_id = np.subtract(pfp_id, 1)
        backtracked_pdg = np.abs(self._selection(
            "backtracked_pdg", sample, query=query, extra_cut=extra_cut))

        plotted_variable = self._select_showers(
            plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        if "trk" in variable:
            pfp_id = np.array([pf_id[score > 0.5] for pf_id, score in zip(pfp_id, score_v)])
        else:
            pfp_id = np.array([pf_id[score <= 0.5] for pf_id, score in zip(pfp_id, score_v)])

        pfp_pdg = np.array([pdg[pf_id]
                            for pdg, pf_id in zip(backtracked_pdg, pfp_id)])
        pfp_pdg = np.hstack(pfp_pdg)
        pfp_pdg = abs(pfp_pdg)

        return pfp_pdg, plotted_variable

    def _categorize_entries_single_pdg(self, sample, variable, query="selection==1", extra_cut=None):
        if "trk" in variable:
            bkt_variable = "trk_bkt_pdg"
        else:
            bkt_variable = "shr_bkt_pdg"

        backtracked_pdg = np.abs(self._selection(
            bkt_variable, sample, query=query, extra_cut=extra_cut))
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)

        return backtracked_pdg, plotted_variable

    def _categorize_entries(self, sample, variable, query="selected==1", extra_cut=None):
        category = self._selection(
            "category", sample, query=query, extra_cut=extra_cut)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)


        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable

    def _categorize_entries_int(self, sample, variable, query="selected==1", extra_cut=None):
        category = self._selection(
            "interaction", sample, query=query, extra_cut=extra_cut)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)


        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable



    @staticmethod
    def _variable_bin_scaling(bins, bin_width, variable):
        idx = bisect.bisect_left(bins, variable)
        if len(bins) > idx:
            return bin_width/(bins[idx]-bins[idx-1])
        return 0

    def _get_genie_weight(self, sample, variable, query="selected==1", extra_cut=None):

        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)
        genie_weights = self._selection(
            "weightSplineTimesTune", sample, query=query, extra_cut=extra_cut)
        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut)
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut)
                genie_weights = np.array([
                    np.array([c] * len(v[s > 0.5])) for c, v, s in zip(genie_weights, plotted_variable, score)
                ])
                genie_weights = np.hstack(genie_weights)
        return genie_weights

    def _get_variable(self, variable, query):
        if ( ("cc" in self.samples) and ("nc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"
        elif ( ("nc" in self.samples) and not ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 1 & category != 5)"
        elif ( not ("nc" in self.samples) and ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 0 & category != 5)"
        else:
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0)"
            
        # if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
        #     query += "& %s <= %g & %s >= %g" % (
        #         variable, plot_options["range"][1], variable, plot_options["range"][0])

        mc_plotted_variable = self._selection(
            variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_weight = [self.weights["mc"]] * len(mc_plotted_variable)

        nue_plotted_variable = self._selection(
            variable, self.samples["nue"], query=query)
        nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["nue"], query=query)
        nue_weight = [self.weights["nue"]] * len(nue_plotted_variable)

        ext_plotted_variable = self._selection(
            variable, self.samples["ext"], query=query)
        ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)
        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)

        dirt_weight = []
        dirt_plotted_variable = []
        if "dirt" in self.samples:
            dirt_plotted_variable = self._selection(
                variable, self.samples["dirt"], query=query)
            dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["dirt"], query=query)
            dirt_weight = [self.weights["dirt"]] * len(dirt_plotted_variable)

        nc_weight = []
        nc_plotted_variable = []
        if "nc" in self.samples:
            nc_plotted_variable = self._selection(
                variable, self.samples["nc"], query=query)
            nc_plotted_variable = self._select_showers(
                nc_plotted_variable, variable, self.samples["nc"], query=query)
            nc_weight = [self.weights["nc"]] * len(nc_plotted_variable)

        cc_weight = []
        cc_plotted_variable = []
        if "cc" in self.samples:
            cc_plotted_variable = self._selection(
                variable, self.samples["cc"], query=query)
            cc_plotted_variable = self._select_showers(
                cc_plotted_variable, variable, self.samples["cc"], query=query)
            cc_weight = [self.weights["cc"]] * len(cc_plotted_variable)

        lee_weight = []
        lee_plotted_variable = []
        if "lee" in self.samples:
            lee_plotted_variable = self._selection(
                variable, self.samples["lee"], query=query)
            lee_plotted_variable = self._select_showers(
                lee_plotted_variable, variable, self.samples["lee"], query=query)
            lee_weight = self.samples["lee"].query(
                query)["leeweight"] * self.weights["lee"]

        total_weight = np.concatenate((mc_weight, nue_weight, ext_weight, dirt_weight, nc_weight, cc_weight, lee_weight))
        total_variable = np.concatenate((mc_plotted_variable, nue_plotted_variable, ext_plotted_variable, dirt_plotted_variable, nc_plotted_variable, cc_plotted_variable, lee_plotted_variable))
        return total_variable, total_weight


    def plot_2d(self, variable1_name, variable2_name, query="selected==1", **plot_options):
        variable1, weight1 = self._get_variable(variable1_name, query)
        variable2, weight2 = self._get_variable(variable2_name, query)

        heatmap, xedges, yedges = np.histogram2d(variable1, variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=weight1)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, axes  = plt.subplots(1,3, figsize=(12,3))

        axes[0].imshow(heatmap.T, extent=extent, origin='lower', aspect="auto")

        data_variable1 = self._selection(variable1_name, self.samples["data"], query=query)
        data_variable1 = self._select_showers(data_variable1, variable1_name, self.samples["data"], query=query)

        data_variable2 = self._selection(
            variable2_name, self.samples["data"], query=query)
        data_variable2 = self._select_showers(
            data_variable2, variable2_name, self.samples["data"], query=query)

        heatmap_data, xedges, yedges = np.histogram2d(data_variable1, data_variable2, range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [
                                                      plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                      bins=[plot_options["bins_x"],
                                                      plot_options["bins_y"]])

        axes[1].imshow(heatmap_data.T, extent=extent, origin='lower', aspect="auto")

        ratio = heatmap_data/heatmap
        im_ratio = axes[2].imshow(ratio.T, extent=extent, origin='lower', aspect='auto', vmin=0, vmax=2, cmap="coolwarm")
        fig.colorbar(im_ratio)

        axes[0].title.set_text('MC+EXT')
        axes[1].title.set_text('Data')
        axes[2].title.set_text('Data/(MC+EXT)')
        if "title" in plot_options:
            axes[0].set_xlabel(plot_options["title"].split(";")[0])
            axes[0].set_ylabel(plot_options["title"].split(";")[1])
            axes[1].set_xlabel(plot_options["title"].split(";")[0])
            axes[2].set_xlabel(plot_options["title"].split(";")[0])
        else:
            axes[0].set_xlabel(variable1_name)
            axes[0].set_ylabel(variable2_name)
            axes[1].set_xlabel(variable1_name)
            axes[2].set_xlabel(variable1_name)

    def plot_variable(self, variable, query="selected==1", title="", kind="event_category", draw_sys=False, **plot_options):
        """It plots the variable from the TTree, after applying an eventual query

        Args:
            variable (str): name of the variable.
            query (str): pandas query. Default is ``selected``.
            title (str, optional): title of the plot. Default is ``variable``.
            kind (str, optional): Categorization of the plot.
                Accepted values are ``event_category``, ``particle_pdg``, and ``sample``
                Default is ``event_category``.
            **plot_options: Additional options for matplotlib plot (e.g. range and bins).

        Returns:
            Figure, top subplot, and bottom subplot (ratio)

        """
        if not title:
            title = variable
        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])

        if kind == "event_category":
            categorization = self._categorize_entries
            cat_labels = category_labels
        elif kind == "particle_pdg":
            var = self.samples["mc"].query(query).eval(variable)
            if var.dtype == np.float32:
                categorization = self._categorize_entries_single_pdg
            else:
                categorization = self._categorize_entries_pdg
            cat_labels = pdg_labels
        elif kind == "interaction":
            categorization = self._categorize_entries_int
            cat_labels = int_labels
        elif kind == "sample":
            return self._plot_variable_samples(variable, query, title, **plot_options)
        else:
            raise ValueError(
                "Unrecognized categorization, valid options are 'sample', 'event_category', and 'particle_pdg'")

        if ( ("cc" in self.samples) and ("nc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"
        elif ( ("nc" in self.samples) and not ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 1 & category != 5)"
        elif ( not ("nc" in self.samples) and ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 0 & category != 5)"
        else:
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0)"
        
        category, mc_plotted_variable = categorization(
            self.samples["mc"], variable, query=query, extra_cut=nu_pdg)

        var_dict = defaultdict(list)
        weight_dict = defaultdict(list)
        mc_genie_weights = self._get_genie_weight(
            self.samples["mc"], variable, query=query, extra_cut=nu_pdg)

        for c, v, w in zip(category, mc_plotted_variable, mc_genie_weights):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["mc"] * w)

        nue_genie_weights = self._get_genie_weight(
            self.samples["nue"], variable, query=query)

        category, nue_plotted_variable = categorization(
            self.samples["nue"], variable, query=query)

        for c, v, w in zip(category, nue_plotted_variable, nue_genie_weights):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["nue"] * w)

        if "nc" in self.samples:
            nc_genie_weights = self._get_genie_weight(
                    self.samples["nc"], variable, query=query)
            category, nc_plotted_variable = categorization(
                self.samples["nc"], variable, query=query)

            for c, v, w in zip(category, nc_plotted_variable, nc_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["nc"] * w)

        if "cc" in self.samples:
            cc_genie_weights = self._get_genie_weight(
                    self.samples["cc"], variable, query=query)
            category, cc_plotted_variable = categorization(
                self.samples["cc"], variable, query=query)

            for c, v, w in zip(category, cc_plotted_variable, cc_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["cc"] * w)


        if "dirt" in self.samples:
            dirt_genie_weights = self._get_genie_weight(
                self.samples["dirt"], variable, query=query)
            category, dirt_plotted_variable = categorization(
                self.samples["dirt"], variable, query=query)

            for c, v, w in zip(category, dirt_plotted_variable, dirt_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["dirt"] * w)

        if "lee" in self.samples:
            category, lee_plotted_variable = categorization(
                self.samples["lee"], variable, query=query)
            leeweight = self.samples["lee"].query(query)["leeweight"] * self._selection("weightSplineTimesTune", self.samples["lee"], query=query)

            for c, v, w in zip(category, lee_plotted_variable, leeweight):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["lee"] * w)

            lee_hist, lee_bins = np.histogram(
                var_dict[111],
                bins=plot_options["bins"],
                range=plot_options["range"],
                weights=weight_dict[111])

        ext_plotted_variable = self._selection(
            variable, self.samples["ext"], query=query)
        ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)

        data_plotted_variable = self._selection(
            variable, self.samples["data"], query=query)
        data_plotted_variable = self._select_showers(data_plotted_variable, variable,
                                                     self.samples["data"], query=query)

        fig = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        total = sum(sum(weight_dict[c]) for c in var_dict)
        total += sum([self.weights["ext"]] * len(ext_plotted_variable))
        labels = [
            "%s: %.1f" % (cat_labels[c], sum(weight_dict[c])) \
            if sum(weight_dict[c]) else ""
            for c in var_dict.keys()
        ]

        if kind == "event_category":
            plot_options["color"] = [category_colors[c]
                                     for c in var_dict.keys()]
        elif kind == "particle_pdg":
            plot_options["color"] = [pdg_colors[c]
                                     for c in var_dict.keys()]
        else:
            plot_options["color"] = [int_colors[c]
                                     for c in var_dict.keys()]
        stacked = ax1.hist(
            var_dict.values(),
            weights=list(weight_dict.values()),
            stacked=True,
            label=labels,
            **plot_options)

        total_array = np.concatenate(list(var_dict.values()))
        total_weight = np.concatenate(list(weight_dict.values()))

        #print(stacked)
        #print(labels)

        plot_options.pop('color', None)

        total_hist, total_bins = np.histogram(
            total_array, weights=total_weight,  **plot_options)

        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)
        n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            weights=ext_weight,
            bottom=total_hist,
            label="EXT: %.1f" % sum(ext_weight) if sum(ext_weight) else "",
            hatch="//",
            color="white",
            **plot_options)

        total_array = np.concatenate([total_array, ext_plotted_variable])
        total_weight = np.concatenate([total_weight, ext_weight])

        n_tot, bin_edges, patches = ax1.hist(
            total_array,
            weights=total_weight,
            histtype="step",
            edgecolor="black",
            **plot_options)

        bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mc_uncertainties, bins = np.histogram(
            mc_plotted_variable, **plot_options)
        err_mc = np.array(
            [n * self.weights["mc"] * self.weights["mc"] for n in mc_uncertainties])

        nue_uncertainties, bins = np.histogram(
            nue_plotted_variable, **plot_options)
        err_nue = np.array(
            [n * self.weights["nue"] * self.weights["nue"] for n in nue_uncertainties])

        err_dirt = np.array([0 for n in mc_uncertainties])
        if "dirt" in self.samples:
            dirt_uncertainties, bins = np.histogram(
                dirt_plotted_variable, **plot_options)
            err_dirt = np.array(
                [n * self.weights["dirt"] * self.weights["dirt"] for n in dirt_uncertainties])

        err_lee = np.array([0 for n in mc_uncertainties])
        if "lee" in self.samples:
            if isinstance(plot_options["bins"], Iterable):
                lee_bins = plot_options["bins"]
            else:
                bin_size = (plot_options["range"][1] - plot_options["range"][0])/plot_options["bins"]
                lee_bins = [plot_options["range"][0]+n*bin_size for n in range(plot_options["bins"]+1)]

            if variable[-2:] != "_v":
                binned_lee = pd.cut(self.samples["lee"].query(query).eval(variable), lee_bins)
                err_lee = self.samples["lee"].query(query).groupby(binned_lee)['leeweight'].agg(
                    "sum").values * self.weights["lee"] * self.weights["lee"]

        err_nc = np.array([0 for n in mc_uncertainties])
        if "nc" in self.samples:
            nc_uncertainties, bins = np.histogram(
                nc_plotted_variable, **plot_options)
            err_nc = np.array(
                [n * self.weights["nc"] * self.weights["nc"] for n in nc_uncertainties])

        err_cc = np.array([0 for n in mc_uncertainties])
        if "cc" in self.samples:
            cc_uncertainties, bins = np.histogram(
                cc_plotted_variable, **plot_options)
            err_cc = np.array(
                [n * self.weights["cc"] * self.weights["cc"] for n in nc_uncertainties])

        err_ext = np.array(
            [n * self.weights["ext"] * self.weights["ext"] for n in n_ext])

        exp_err = np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_nc + err_cc)

        bin_size = [(bin_edges[i + 1] - bin_edges[i]) / 2
                    for i in range(len(bin_edges) - 1)]

        cov = np.zeros([len(exp_err), len(exp_err)])

        if draw_sys:
            #cov = self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTune")
            cov = self.sys_err("weightsGenie", variable, query, plot_options["range"], plot_options["bins"], "weightSpline") + \
                  self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTune") #+ \
                  #self.sys_err("weightsReint", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTune")
            exp_err = np.sqrt(np.diag(cov) + exp_err*exp_err)

        cov[np.diag_indices_from(cov)] += (err_mc + err_ext + err_nue + err_dirt + err_nc + err_cc)

        if "lee" in self.samples:
            if kind == "event_category":
                try:
                    self.significance = self._sigma_calc_matrix(
                        lee_hist, n_tot-lee_hist, scale_factor=1.3e21/self.pot, cov=cov)
                    self.significance_likelihood = self._sigma_calc_likelihood(
                        lee_hist, n_tot-lee_hist, np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_nc + err_cc), scale_factor=1.3e21/self.pot)
                except (np.linalg.LinAlgError, ValueError) as err:
                    print("Error calculating the significance", err)
                    self.significance = -1
                    self.significance_likelihood = -1

        ax1.bar(bincenters, n_tot, facecolor='none',
                edgecolor='none', width=0, yerr=exp_err)

        n_data, bins = np.histogram(data_plotted_variable, **plot_options)
        data_err = np.sqrt(n_data)
        if sum(n_data) > 0:
            ax1.errorbar(
                bincenters,
                n_data,
                xerr=bin_size,
                yerr=data_err,
                fmt='ko',
                label="BNB: %i" % len(data_plotted_variable) if len(data_plotted_variable) else "")

        leg = ax1.legend(
            frameon=False, ncol=3, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), fontweight='bold')

        unit = title[title.find("[") +
                     1:title.find("]")] if "[" and "]" in title else ""
        x_range = plot_options["range"][1] - plot_options["range"][0]
        if isinstance(plot_options["bins"], Iterable):
            ax1.set_ylabel("N. Entries")
        else:
            ax1.set_ylabel(
                "N. Entries / %g %s" % (x_range / plot_options["bins"], unit))
        ax1.set_xticks([])
        ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])

        self._draw_ratio(ax2, bins, n_tot, n_data, exp_err, data_err)
        # if sum(n_data) > 0:
        #     ax2.text(
        #         0.88,
        #         0.845,
        #         r'$\chi^2 /$n.d.f. = %.2f' % self._chisquare(n_data, n_tot, data_err, exp_err) +
        #         '\n' +
        #         'K.S. prob. = %.2f' % scipy.stats.ks_2samp(n_data, n_tot)[1],
        #         va='center',
        #         ha='center',
        #         ma='right',
        #         fontsize=12,
        #         transform=ax2.transAxes)

        ax2.set_xlabel(title)
        ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
        fig.tight_layout()
        if title == variable:
            ax1.set_title(query)
        #     fig.suptitle(query)
        # fig.savefig("plots/%s_cat.pdf" % variable.replace("/", "_"))
        return fig, ax1, ax2

    def _plot_variable_samples(self, variable, query, title, **plot_options):

        if ( ("cc" in self.samples) and ("nc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"
        elif ( ("nc" in self.samples) and not ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 1 & category != 5)"
        elif ( not ("nc" in self.samples) and ("cc" in self.samples) ):
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & ccnc == 0 & category != 5)"
        else:
            nu_pdg = "~(nu_pdg == 12 & ccnc == 0)"
        
        if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])

        mc_plotted_variable = self._selection(
            variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_weight = [self.weights["mc"]] * len(mc_plotted_variable)

        nue_plotted_variable = self._selection(
            variable, self.samples["nue"], query=query)
        nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["nue"], query=query)
        nue_weight = [self.weights["nue"]] * len(nue_plotted_variable)

        ext_plotted_variable = self._selection(
            variable, self.samples["ext"], query=query)
        ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)
        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)

        if "dirt" in self.samples:
            dirt_plotted_variable = self._selection(
                variable, self.samples["dirt"], query=query)
            dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["dirt"], query=query)
            dirt_weight = [self.weights["dirt"]] * len(dirt_plotted_variable)

        if "nc" in self.samples:
            nc_plotted_variable = self._selection(
                variable, self.samples["nc"], query=query)
            nc_plotted_variable = self._select_showers(
                nc_plotted_variable, variable, self.samples["nc"], query=query)
            nc_weight = [self.weights["nc"]] * len(nc_plotted_variable)

        if "cc" in self.samples:
            cc_plotted_variable = self._selection(
                variable, self.samples["cc"], query=query)
            cc_plotted_variable = self._select_showers(
                cc_plotted_variable, variable, self.samples["cc"], query=query)
            cc_weight = [self.weights["cc"]] * len(cc_plotted_variable)            

        if "lee" in self.samples:
            lee_plotted_variable = self._selection(
                variable, self.samples["lee"], query=query)
            lee_plotted_variable = self._select_showers(
                lee_plotted_variable, variable, self.samples["lee"], query=query)
            lee_weight = self.samples["lee"].query(query)["leeweight"] * self.weights["lee"]


        data_plotted_variable = self._selection(
            variable, self.samples["data"], query=query)
        data_plotted_variable = self._select_showers(
            data_plotted_variable,
            variable,
            self.samples["data"],
            query=query)

        if "dirt" in self.samples:
            total_variable = np.concatenate(
                [mc_plotted_variable,
                 nue_plotted_variable,
                 ext_plotted_variable,
                 dirt_plotted_variable])
            total_weight = np.concatenate(
                [mc_weight, nue_weight, ext_weight, dirt_weight])
        else:
            total_variable = np.concatenate(
                [mc_plotted_variable, nue_plotted_variable, ext_plotted_variable])
            total_weight = np.concatenate(
                [mc_weight, nue_weight, ext_weight])

        if "lee" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 lee_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, lee_weight])

        if "nc" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 nc_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, nc_weight])

        if "cc" in self.samples:
            total_variable = np.concatenate(
                [total_variable,
                 cc_plotted_variable])
            total_weight = np.concatenate(
                [total_weight, cc_weight])

            
        fig = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        n_mc, mc_bins, patches = ax1.hist(
            mc_plotted_variable,
            weights=mc_weight,
            label="BNB overlay: %.1f entries" % sum(mc_weight),
            **plot_options)

        n_nue, nue_bins, patches = ax1.hist(
            nue_plotted_variable,
            bottom=n_mc,
            weights=nue_weight,
            label=r"$\nu_{e}$ overlay: %.1f entries" % sum(nue_weight),
            **plot_options)

        n_dirt = 0
        if "dirt" in self.samples:
            n_dirt, dirt_bins, patches = ax1.hist(
                dirt_plotted_variable,
                bottom=n_mc + n_nue,
                weights=dirt_weight,
                label=r"Dirt: %.1f entries" % sum(dirt_weight),
                **plot_options)

        n_nc = 0
        if "nc" in self.samples:
            n_nc, nc_bins, patches = ax1.hist(
                nc_plotted_variable,
                bottom=n_mc + n_nue + n_dirt,
                weights=nc_weight,
                label=r"NC$\pi^0$: %.1f entries" % sum(nc_weight),
                **plot_options)

        n_cc = 0
        if "cc" in self.samples:
            n_cc, cc_bins, patches = ax1.hist(
                cc_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_nc,
                weights=nc_weight,
                label=r"CC$\pi^0$: %.1f entries" % sum(cc_weight),
                **plot_options)

        n_lee = 0
        if "lee" in self.samples:
            n_lee, lee_bins, patches = ax1.hist(
                lee_plotted_variable,
                bottom=n_mc + n_nue + n_dirt + n_nc + n_cc,
                weights=lee_weight,
                label=r"MiniBooNE LEE: %.1f entries" % sum(lee_weight),
                **plot_options)

        n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            bottom=n_mc + n_nue + n_dirt + n_lee + n_nc + n_cc,
            weights=ext_weight,
            label="EXT: %.1f entries" % sum(ext_weight),
            hatch="//",
            color="white",
            **plot_options)

        n_tot, tot_bins, patches = ax1.hist(
            total_variable,
            weights=total_weight,
            histtype="step",
            edgecolor="black",
            **plot_options)

        mc_uncertainties, bins = np.histogram(
            mc_plotted_variable, **plot_options)
        nue_uncertainties, bins = np.histogram(
            nue_plotted_variable, **plot_options)
        ext_uncertainties, bins = np.histogram(
            ext_plotted_variable, **plot_options)
        err_mc = np.array([n * self.weights["mc"] * self.weights["mc"] for n in mc_uncertainties])
        err_nue = np.array(
            [n * self.weights["nue"] * self.weights["nue"] for n in nue_uncertainties])
        err_ext = np.array(
            [n * self.weights["ext"] * self.weights["ext"] for n in ext_uncertainties])
        err_dirt = np.array([0 for n in n_mc])
        err_lee = np.array([0 for n in n_mc])

        if "dirt" in self.samples:
            dirt_uncertainties, bins = np.histogram(dirt_plotted_variable, **plot_options)
            err_dirt = np.array(
                [n * self.weights["dirt"] * self.weights["dirt"] for n in dirt_uncertainties])

        err_nc = np.array([0 for n in n_mc])
        if "nc" in self.samples:
            nc_uncertainties, bins = np.histogram(nc_plotted_variable, **plot_options)
            err_nc = np.array(
                [n * self.weights["nc"] * self.weights["nc"] for n in nc_uncertainties])

        err_cc = np.array([0 for n in n_mc])
        if "cc" in self.samples:
            cc_uncertainties, bins = np.histogram(cc_plotted_variable, **plot_options)
            err_cc = np.array(
                [n * self.weights["cc"] * self.weights["cc"] for n in cc_uncertainties])

        if "lee" in self.samples:
            if isinstance(plot_options["bins"], Iterable):
                lee_bins = plot_options["bins"]
            else:
                bin_size = (
                    plot_options["range"][1] - plot_options["range"][0])/plot_options["bins"]
                lee_bins = [plot_options["range"][0]+n *
                            bin_size for n in range(plot_options["bins"]+1)]

            binned_lee = pd.cut(self.samples["lee"].query(
                query).eval(variable), lee_bins)
            err_lee = self.samples["lee"].query(query).groupby(binned_lee)['leeweight'].agg(
                "sum").values * self.weights["lee"] * self.weights["lee"]
        exp_err = np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_lee + err_nc + err_cc)

        bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
        bin_size = [(tot_bins[i + 1] - tot_bins[i]) / 2
                    for i in range(len(tot_bins) - 1)]
        ax1.bar(bincenters, n_tot, width=0, yerr=exp_err)

        n_data, bins = np.histogram(data_plotted_variable, **plot_options)
        data_err = np.sqrt(n_data)
        ax1.errorbar(
            bincenters,
            n_data,
            xerr=bin_size,
            yerr=data_err,
            fmt='ko',
            label="BNB: %i events" % len(data_plotted_variable))

        leg = ax1.legend(
            frameon=False, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), fontweight='bold')

        unit = title[title.find("[") + 1:title.find("]")
                     ] if "[" and "]" in title else ""
        xrange = plot_options["range"][1] - plot_options["range"][0]
        if isinstance(bins, Iterable):
            ax1.set_ylabel("N. Entries")
        else:
            ax1.set_ylabel(
                "N. Entries / %g %s" % (xrange / plot_options["bins"], unit))
        ax1.set_xticks([])
        ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])

        self._draw_ratio(ax2, bins, n_tot, n_data, exp_err, data_err)

        ax2.set_xlabel(title)
        ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
        fig.tight_layout()
        # fig.savefig("plots/%s_samples.pdf" % variable)
        return fig, ax1, ax2

    def _draw_ratio(self, ax, bins, n_tot, n_data, tot_err, data_err):
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        bin_size = [(bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)]
        ratio_error = self._ratio_err(n_data, n_tot, data_err, tot_err)
        ax.errorbar(bincenters, n_data / n_tot,
                    xerr=bin_size, yerr=ratio_error, fmt="ko")

        ratio_error_mc = self._ratio_err(n_tot, n_tot, tot_err, tot_err)
        ratio_error_mc = np.insert(ratio_error_mc, 0, ratio_error_mc[0])
        ax.fill_between(
            bins,
            1 - ratio_error_mc,
            ratio_error_mc + 1,
            step="pre",
            color="grey",
            alpha=0.5)

        ax.set_ylim(0, 2)
        ax.set_ylabel("BNB / (MC+EXT)")
        ax.axhline(1, linestyle="--", color="k")

    def sys_err(self, name, var_name, query, x_range, n_bins, weightVar):

        n_tot = np.empty([50, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["ext", "data", "lee"]:
                continue

            tree = self.samples[t]

            extra_query = ""
            if t == "mc":
                extra_query = "& ~(nu_pdg == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            syst_weights = queried_tree[name]
            spline_fix = queried_tree[weightVar] * self.weights[t]

            s = syst_weights
            df = pd.DataFrame(s.values.tolist())

            n_cv, bins = np.histogram(
                variable,
                range=x_range,
                bins=n_bins,
                weights=spline_fix)
            n_cv_tot += n_cv

            # how many universes?
            Nuniverse = 50 #len(df)
            print ('Nuniverse: %i'%Nuniverse)

            if not df.empty:
                for i in range(Nuniverse):
                    weight = df[i].values
                    weight[np.isnan(weight)] = 1
                    weight[weight > 100] = 1
                    weight[weight < 0] = 1
                    weight[weight == np.inf] = 1

                    n, bins = np.histogram(
                        variable, weights=weight*spline_fix, range=x_range, bins=n_bins)
                    n_tot[i] += n

        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n in n_tot:
            for i in range(len(n_cv)):
                for j in range(len(n_cv)):
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse

        return cov
