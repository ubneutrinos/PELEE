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

import numpy as np

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
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic"
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
        pot (int): Number of protons-on-target. Defaults is 4.3e19.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    def __init__(self, samples, weights, pot=4.3e19):
        self.weights = weights
        self.samples = samples
        self.pot = pot
        self.significance = 0

        if "dirt" not in samples:
            warnings.warn("Missing dirt sample")

        necessary = ["trk_pfp_id", "category", "shr_pfp_id_v", "selected",
                     "backtracked_pdg", "nu_pdg", "ccnc", "trk_bkt_pdg", "shr_bkt_pdg"]

        missing = np.setdiff1d(necessary, samples["mc"].columns)

        if missing.size > 0:
            raise ValueError(
                "Missing necessary columns in the DataFrame: %s" % missing)

    @staticmethod
    def _sigma_calc_matrix(signal, background, scale_factor=1):
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

        sig_array = np.delete(sig_array, empty_elements)
        bkg_array = np.delete(bkg_array, empty_elements)

        nbins = len(sig_array)

        emtx = np.zeros((nbins, nbins))
        np.fill_diagonal(emtx, bkg_array)

        emtxinv = np.linalg.inv(emtx)
        chisq = float(sig_array.dot(emtxinv).dot(sig_array.T))


        return np.sqrt(chisq)

    @staticmethod
    def _ratio_err(num, den, num_err, den_err):
        n, d, n_e, d_e = num, den, num_err, den_err
        return np.array([
            n[i] / d[i] * math.sqrt((n_e[i] / n[i])**2 + (d_e[i] / d[i])**2)
            for i in range(len(num))
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
        return sum(num / den) / np.count_nonzero(data)

    def _select_showers(self, variable, variable_name, sample, query="selected==1", score=0.5, extra_cut=None):
        variable = variable.ravel()

        if variable.size > 0:
            if isinstance(variable[0], np.ndarray):
                variable = np.hstack(variable)
                if "shr" in variable_name and variable_name != "trkshr_score_v":
                    shr_score = np.hstack(self._selection(
                        "trkshr_score_v", sample, query=query, extra_cut=extra_cut).ravel())
                    shr_score_id = shr_score < score
                    variable = variable[shr_score_id]

        return variable

    def _selection(self, variable, sample, query="selected==1", extra_cut=None):
        sel_query = query

        if extra_cut is not None:
            sel_query += "& %s" % extra_cut

        return sample.query(sel_query).eval(variable).ravel()

    def _categorize_entries_pdg(self, sample, variable, query="selected==1", extra_cut=None):

        if "trk" in variable:
            pfp_id_variable = "trk_pfp_id"
        else:
            pfp_id_variable = "shr_pfp_id_v"

        pfp_id = self._selection(
            pfp_id_variable, sample, query=query, extra_cut=extra_cut)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut)
        pfp_id = np.subtract(pfp_id, 1)
        backtracked_pdg = np.abs(self._selection(
            "backtracked_pdg", sample, query=query, extra_cut=extra_cut))

        plotted_variable = self._select_showers(
            plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        pfp_pdg = np.array([pdg[pfp_id]
                            for pdg, pfp_id in zip(backtracked_pdg, pfp_id)])
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
                category = np.array([
                    np.array([c] * len(v)) for c, v in zip(category, plotted_variable)
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

    def plot_variable(self, variable, query="selected==1", title="", kind="event_category", **plot_options):
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

        if "range" in plot_options:
            query += "& %s > %g & %s < %g" % (variable, plot_options["range"][0], variable, plot_options["range"][1])

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
        elif kind == "sample":
            return self._plot_variable_samples(variable, query, title, **plot_options)
        else:
            raise ValueError(
                "Unrecognized categorization, valid options are 'sample', 'event_category', and 'particle_pdg'")

        nu_pdg = "~(nu_pdg == 12 & ccnc == 0)"
        category, mc_plotted_variable = categorization(
            self.samples["mc"], variable, query=query, extra_cut=nu_pdg)

        var_dict = defaultdict(list)
        weight_dict = defaultdict(list)
        for c, v in zip(category, mc_plotted_variable):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["mc"])

        category, nue_plotted_variable = categorization(
            self.samples["nue"], variable, query=query)

        for c, v in zip(category, nue_plotted_variable):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["nue"])

        if "dirt" in self.samples:
            category, dirt_plotted_variable = categorization(
                self.samples["dirt"], variable, query=query)

            for c, v in zip(category, dirt_plotted_variable):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["dirt"])

        if "lee" in self.samples:
            category, lee_plotted_variable = categorization(
                self.samples["lee"], variable, query=query)
            leeweight = self.samples["lee"].query(query)["leeweight"]
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
            "%s: %.1f" % (cat_labels[c], sum(weight_dict[c]))  # / total * 100)
            for c in var_dict.keys()
        ]

        if kind == "event_category":
            plot_options["color"] = [category_colors[c]
                                     for c in var_dict.keys()]
        else:
            plot_options["color"] = [pdg_colors[c]
                                     for c in var_dict.keys()]

        ax1.hist(
            var_dict.values(),
            **plot_options,
            weights=list(weight_dict.values()),
            stacked=True,
            label=labels)

        total_array = np.concatenate(list(var_dict.values()))
        total_weight = np.concatenate(list(weight_dict.values()))

        plot_options.pop('color', None)

        total_hist, total_bins = np.histogram(
            total_array, **plot_options, weights=total_weight)

        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)
        n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            **plot_options,
            weights=ext_weight,
            bottom=total_hist,
            label="EXT: %.1f" %
            # / total * 100),
            (sum([self.weights["ext"]] * len(ext_plotted_variable))),
            hatch="//",
            color="white")

        total_array = np.concatenate([total_array, ext_plotted_variable])
        total_weight = np.concatenate([total_weight, ext_weight])

        n_tot, bin_edges, patches = ax1.hist(
            total_array,
            **plot_options,
            weights=total_weight,
            histtype="step",
            edgecolor="black")

        if "lee" in self.samples:
            try:
                self.significance = self._sigma_calc_matrix(
                    lee_hist, n_tot-lee_hist, scale_factor=1.3e21/4.26e19)
                # print("Significance stat. only: %g sigma" % self.significance)
            except np.linalg.LinAlgError:
                print("Error calculating the significance")

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

        err_ext = np.array(
            [n * self.weights["ext"] * self.weights["ext"] for n in n_ext])

        exp_err = np.sqrt(err_mc + err_ext + err_nue + err_dirt)

        bin_size = [(bin_edges[i + 1] - bin_edges[i]) / 2
                    for i in range(len(bin_edges) - 1)]
        ax1.bar(bincenters, n_tot, width=0, yerr=exp_err)

        n_data, bins = np.histogram(data_plotted_variable, **plot_options)
        data_err = np.sqrt(n_data)
        ax1.errorbar(
            bincenters,
            n_data,
            xerr=bin_size,
            yerr=data_err,
            fmt='ko',
            label="BNB: %i" % len(data_plotted_variable))

        leg = ax1.legend(
            frameon=False, ncol=3, title=r'MicroBooNE Preliminary %g POT' % self.pot)
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), fontweight='bold')

        unit = title[title.find("[") +
                     1:title.find("]")] if "[" and "]" in title else ""
        xrange = plot_options["range"][1] - plot_options["range"][0]
        if isinstance(plot_options["bins"], Iterable):
            ax1.set_ylabel("N. Entries")
        else:
            ax1.set_ylabel(
                "N. Entries / %g %s" % (xrange / plot_options["bins"], unit))
        ax1.set_xticks([])
        ax1.set_xlim(plot_options["range"][0], plot_options["range"][1])

        self._draw_ratio(ax2, bins, n_tot, n_data, exp_err, data_err)

        # ax2.text(
        #     0.88,
        #     0.845,
        #     r'$\chi^2 /$n.d.f. = %.2f' % self._chisquare(n_data, n_tot, data_err, exp_err) +
        #     '\n' +
        #     'K.S. prob. = %.2f' % scipy.stats.ks_2samp(n_data, n_tot)[1],
        #     va='center',
        #     ha='center',
        #     ma='right',
        #     fontsize=12,
        #     transform=ax2.transAxes)

        ax2.set_xlabel(title)
        ax2.set_xlim(plot_options["range"][0], plot_options["range"][1])
        fig.tight_layout()
        if title == variable:
            ax1.set_title(query)
        #     fig.suptitle(query)
        # fig.savefig("plots/%s_cat.pdf" % variable.replace("/", "_"))
        return fig, ax1, ax2

    def _plot_variable_samples(self, variable, query, title, **plot_options):
        nu_pdg = "~(nu_pdg == 12 & ccnc == 0)"
        mc_plotted_variable = self._selection(
            variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["mc"], query=query, extra_cut=nu_pdg)
        mc_weight = [self.weights["mc"]] * len(mc_plotted_variable)

        nue_plotted_variable = self._selection(
            variable, self.samples["nue"], query=query,)
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

        fig = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        n_mc, mc_bins, patches = ax1.hist(
            mc_plotted_variable,
            **plot_options,
            weights=mc_weight,
            label="BNB overlay: %g entries" % sum(mc_weight))

        n_nue, nue_bins, patches = ax1.hist(
            nue_plotted_variable,
            **plot_options,
            bottom=n_mc,
            weights=nue_weight,
            label=r"$\nu_{e}$ overlay: %g entries" % sum(nue_weight))

        n_dirt = 0
        if "dirt" in self.samples:
            n_dirt, dirt_bins, patches = ax1.hist(
                dirt_plotted_variable,
                **plot_options,
                bottom=n_mc + n_nue,
                weights=dirt_weight,
                label=r"Dirt: %g entries" % sum(dirt_weight))

        n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            **plot_options,
            bottom=n_mc + n_nue + n_dirt,
            weights=ext_weight,
            label="EXT: %g entries" % sum(ext_weight),
            hatch="//",
            color="white")

        n_tot, tot_bins, patches = ax1.hist(
            total_variable,
            **plot_options,
            weights=total_weight,
            histtype="step",
            edgecolor="black")

        err_mc = np.array(
            [n * self.weights["mc"] * self.weights["mc"] for n in n_mc])
        err_nue = np.array(
            [n * self.weights["nue"] * self.weights["nue"] for n in n_nue])
        err_ext = np.array(
            [n * self.weights["ext"] * self.weights["ext"] for n in n_ext])
        err_dirt = np.array([0 for n in n_mc])
        if "dirt" in self.samples:
            err_dirt = np.array(
                [n * self.weights["dirt"] * self.weights["dirt"] for n in n_dirt])
        tot_uncertainties = np.sqrt(err_mc + err_ext + err_nue + err_dirt)

        bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
        exp_err = tot_uncertainties
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

        ax.set_ylim(0.5, 1.5)
        ax.set_ylabel("BNB / (MC+EXT)")
        ax.axhline(1, linestyle="--", color="k")
