"""Module to plot histograms for runs of data and simulation."""

import numpy as np
import matplotlib.pyplot as plt
from .histogram import RunHistGenerator
import unblinding_far_sideband as far_sb

# TODO: add more variables
# There are many more variables defined in far_sb, but it's not clear which list to use when.
variables = [
    ("n_showers_contained", 1, (0.5, 1.5), "normalization", "onebin"),
    # ('n_showers_contained',10,(-0.5, 9.5),"n showers contained"),
    ("n_tracks_contained", 6, (-0.5, 5.5), "n tracks contained"),
    # ('n_tracks_tot',6,(-0.5, 5.5),"n tracks total"),
    # ('reco_e',21,(0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',20,(0.05,3.05),r"Reconstructed Energy [GeV]","extended"),
    # ('reco_e',7,(0.05,2.85),r"Reconstructed Energy [GeV]","coarse"),
    # ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',21,(-0.05,4.15),r"Reconstructed Energy [GeV]","extended"),
    # ('reco_e',20,(0.15,2.95),r"Reconstructed Energy [GeV]","note"),
    # ('reco_e',10,(0.9,3.9),r"Reconstructed Energy [GeV]","highe"),
    ("reco_e", 17, (0.01, 2.39), r"Reconstructed Energy [ GeV ]"),
]


class Plotter(RunHistGenerator):
    def __init__(self, rundata_dict, selection, preselection, weight_column="weights", variable=None):
        query, title = self.get_query_and_title(selection, preselection)
        VARIABLE, BINS, RANGE, XTIT = self.get_variable_definitions(variable)
        self.title = title
        self.xtit = XTIT
        bin_edges = np.linspace(*RANGE, BINS + 1)
        super().__init__(rundata_dict, weight_column, variable=VARIABLE, query=query, binning=bin_edges)

    def get_variable_definitions(self, variable):
        for var_tuple in variables:
            if var_tuple[0] == variable:
                return var_tuple
        raise ValueError(f"Variable {variable} not found in variable definitions.")

    def get_query_and_title(self, selection, preselection):
        presel_query = far_sb.preselection_categories[preselection]["query"]
        presel_title = far_sb.preselection_categories[preselection]["title"]

        sel_query = far_sb.selection_categories[selection]["query"]
        sel_title = far_sb.selection_categories[selection]["title"]

        if presel_query is None:
            query = sel_query
            presel_title = "No Presel."
        elif sel_query is None:
            query = presel_query
            sel_title = "No Sel."
        else:
            query = f"{presel_query} and {sel_query}"

        title = f"{presel_title} and {sel_title}"
        return query, title

    def plot(
        self,
        ax=None,
        show_errorband=True,
        uncertainty_color="gray",
        uncertainty_label="Uncertainty",
        category_column="dataset_name",
        include_multisim_errors=False,
        **kwargs,
    ):
        data_hist = self.get_data_hist()
        ext_hist = self.get_data_hist(type="ext")
        ext_hist.tex_string = "EXT"
        mc_hists = self.get_mc_hists(category_column=category_column, include_multisim_errors=include_multisim_errors)
        background_hists = list(mc_hists.values()) + [ext_hist]
        print(kwargs)
        ax = self.plot_stacked_hists(
            background_hists,
            ax=ax,
            show_errorband=show_errorband,
            uncertainty_color=uncertainty_color,
            uncertainty_label=uncertainty_label,
            **kwargs,
        )
        ax = self.plot_hist(data_hist, ax=ax, label="Data", color="black", as_errorbars=True,  **kwargs)
        ax.set_xlabel(self.xtit)
        ax.set_ylabel("Events")
        ax.set_title(self.title)
        ax.legend()
        return ax

    def plot_hist(
        self,
        hist,
        ax=None,
        show_errorband=True,
        as_errorbars=False,
        uncertainty_color=None,
        uncertainty_label=None,
        **kwargs,
    ):
        """Plot a histogram with uncertainties."""

        if ax is None:
            ax = plt.gca()
        # make a step plot of the histogram
        bin_counts = hist.nominal_values
        bin_edges = hist.bin_edges
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = hist.tex_string
        if as_errorbars:
            ax.errorbar(
                hist.bin_centers, bin_counts, yerr=hist.std_devs, linestyle="none", marker=".", label=label, **kwargs
            )
            return ax
        # Be sure to repeat the last bin count
        bin_counts = np.append(bin_counts, bin_counts[-1])
        p = ax.step(bin_edges, bin_counts, where="post", label=label, **kwargs)
        if not show_errorband:
            return ax
        # plot uncertainties as a shaded region
        uncertainties = hist.std_devs
        uncertainties = np.append(uncertainties, uncertainties[-1])
        # ensure that error band has the same color as the plot we made earlier unless otherwise specified
        if uncertainty_color is None:
            kwargs["color"] = p[0].get_color()
        else:
            kwargs["color"] = uncertainty_color

        ax.fill_between(
            bin_edges,
            np.clip(bin_counts - uncertainties, 0, None),
            bin_counts + uncertainties,
            alpha=0.5,
            step="post",
            label=uncertainty_label,
            **kwargs,
        )

        return ax

    def plot_stacked_hists(
        self, hists, ax=None, show_errorband=True, uncertainty_color=None, uncertainty_label=None, show_counts=True, **kwargs
    ):
        """Plot a stack of histograms."""
        if ax is None:
            ax = plt.gca()

        x = hists[0].bin_edges

        def repeated_nom_values(hist):
            # repeat the last bin count
            y = hist.nominal_values
            y = np.append(y, y[-1])
            return y

        # to use plt.stackplot,  we need y to be a 2D array of shape (N, len(x))
        y = np.array([repeated_nom_values(hist) for hist in hists])
        labels = [hist.tex_string for hist in hists]
        if show_counts:
            labels = [f"{label}: {hist.sum():.1f}" for label, hist in zip(labels, hists)]
        colors = None
        # If all hist.color are not None, we can pass them to stackplot
        if all([hist.color is not None for hist in hists]):
            colors = [hist.color for hist in hists]

        ax.stackplot(x, y, step="post", labels=labels, colors=colors, **kwargs)
        if not show_errorband:
            return ax
        # plot uncertainties as a shaded region, but only for the sum of all hists
        summed_hist = sum(hists)
        # show sum as black line
        p = ax.step(summed_hist.bin_edges, repeated_nom_values(summed_hist), where="post", color="k", lw=0.5)
        # show uncertainty as shaded region
        uncertainties = summed_hist.std_devs
        uncertainties = np.append(uncertainties, uncertainties[-1])
        # ensure that error band has the same color as the plot we made earlier unless otherwise specified
        if uncertainty_color is None:
            kwargs["color"] = p[0].get_color()
        else:
            kwargs["color"] = uncertainty_color
        ax.fill_between(
            summed_hist.bin_edges,
            np.clip(repeated_nom_values(summed_hist) - uncertainties, 0, None),
            repeated_nom_values(summed_hist) + uncertainties,
            alpha=0.6,
            step="post",
            label=uncertainty_label,
            **kwargs,
        )
        return ax
