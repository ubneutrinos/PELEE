"""Module to plot histograms for runs of data and simulation."""

import numpy as np
import matplotlib.pyplot as plt
from .histogram import RunHistGenerator, Binning
from . import selections


class RunHistPlotter:
    def __init__(self, run_hist_generator, selection_title=None):
        self.title = selection_title
        self.run_hist_generator = run_hist_generator

    def get_selection_title(self, selection, preselection):
        presel_title = selections.preselection_categories[preselection]["title"]
        sel_title = selections.selection_categories[selection]["title"]

        if preselection.lower() == "none":
            presel_title = "No Presel."
        elif selection.lower() == "none":
            sel_title = "No Sel."

        title = f"{presel_title} and {sel_title}"
        return title

    def get_pot_label(self, scale_to_pot):
        if self.run_hist_generator.data_pot is None:
            return ""
        if scale_to_pot is None:
            return f"Data POT: {self.run_hist_generator.data_pot:.1e}"
        else:
            return f"MC Scaled to {scale_to_pot:.1e} POT"

    def plot(
        self,
        category_column="dataset_name",
        include_multisim_errors=None,
        add_ext_error_floor=None,
        show_data_mc_ratio=False,
        use_sideband=None,
        ax=None,
        scale_to_pot=None,
        uncertainty_color="gray",
        **kwargs,
    ):
        gen = self.run_hist_generator
        if use_sideband:
            assert gen.sideband_generator is not None
        # we want the uncertainty defaults of the generator to be used if the user doesn't specify
        gen_defaults = gen.uncertainty_defaults
        if include_multisim_errors is None:
            include_multisim_errors = gen_defaults.get("include_multisim_errors", False)
        if add_ext_error_floor is None:
            add_ext_error_floor = gen_defaults.get("add_ext_error_floor", True)
        if use_sideband is None:
            use_sideband = gen_defaults.get("use_sideband", False)
        ext_hist = gen.get_data_hist(type="ext", add_error_floor=add_ext_error_floor, scale_to_pot=scale_to_pot)
        ext_hist.tex_string = "EXT"
        mc_hists = gen.get_mc_hists(
            category_column=category_column, include_multisim_errors=False, scale_to_pot=scale_to_pot
        )
        background_hists = list(mc_hists.values()) + [ext_hist]
        total_mc_hist = gen.get_mc_hist(
            include_multisim_errors=include_multisim_errors, scale_to_pot=scale_to_pot, use_sideband=use_sideband
        )
        total_pred_hist = total_mc_hist + ext_hist
        data_hist = gen.get_data_hist()
        if self.title is None:
            selection, preselection = gen.selection, gen.preselection
            title = self.get_selection_title(selection, preselection)
        else:
            title = self.title

        if show_data_mc_ratio:
            # TODO: implement plotting within inset axes
            assert ax is None, "Can't plot within an ax when showing data/mc ratio"
            fig, (ax, ax_ratio) = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

        ax = self._plot(
            total_pred_hist,
            background_hists,
            title=title,
            data_hist=data_hist,
            ax=ax,
            scale_to_pot=scale_to_pot,
            **kwargs,
        )
        if not show_data_mc_ratio:
            ax.set_xlabel(total_pred_hist.binning.label)
            return ax

        # plot data/mc ratio
        # The way this is typically shown is to have the MC prediction divided by its central
        # data with error bands to show the MC uncertainty, and then to overlay the data points
        # with error bars.
        mc_nominal = total_mc_hist.nominal_values
        mc_error_band = total_mc_hist / mc_nominal
        data_mc_ratio = data_hist / mc_nominal

        self.plot_hist(
            mc_error_band,
            ax=ax_ratio,
            show_errorband=True,
            as_errorbars=False,
            color="k",
            uncertainty_color=uncertainty_color,
        )
        self.plot_hist(data_mc_ratio, ax=ax_ratio, show_errorband=False, as_errorbars=True, color="k")

        ax_ratio.set_ylabel("Data/MC")
        ax_ratio.set_xlabel(total_pred_hist.binning.label)
        # ax_ratio.set_ylim(0, 5)

        return ax, ax_ratio

    def _plot(
        self,
        total_pred_hist,
        background_hists,
        title=None,
        data_hist=None,
        ax=None,
        show_errorband=True,
        uncertainty_color="gray",
        uncertainty_label="Uncertainty",
        scale_to_pot=None,
        **kwargs,
    ):
        ax = self.plot_stacked_hists(
            background_hists,
            ax=ax,
            show_errorband=False,
            **kwargs,
        )
        ax = self.plot_hist(
            total_pred_hist,
            ax=ax,
            show_errorband=show_errorband,
            uncertainty_color=uncertainty_color,
            uncertainty_label=uncertainty_label,
            color="k",
            lw=0.5,
        )
        if scale_to_pot is None:
            if data_hist is not None:  # skip if no data (as is the case for blind analysis)
                # rescaling data to a different POT doesn't make sense
                data_label = f"Data: {data_hist.sum():.1f}"
                ax = self.plot_hist(data_hist, ax=ax, label=data_label, color="black", as_errorbars=True, **kwargs)
        # make text label for the POT
        pot_label = self.get_pot_label(scale_to_pot)
        ax.text(
            0.05,
            0.95,
            pot_label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )
        
        ax.set_ylabel("Events")
        ax.set_title(title)
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
        bin_counts[bin_counts <= 0] = np.nan
        bin_edges = hist.binning.bin_edges
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = hist.tex_string
        if as_errorbars:
            ax.errorbar(
                hist.binning.bin_centers,
                bin_counts,
                yerr=hist.std_devs,
                linestyle="none",
                marker=".",
                label=label,
                **kwargs,
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
        self,
        hists,
        ax=None,
        show_errorband=True,
        uncertainty_color=None,
        uncertainty_label=None,
        show_counts=True,
        **kwargs,
    ):
        """Plot a stack of histograms."""
        if ax is None:
            ax = plt.gca()

        x = hists[0].binning.bin_edges

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
        p = ax.step(summed_hist.binning.bin_edges, repeated_nom_values(summed_hist), where="post", color="k", lw=0.5)
        # show uncertainty as shaded region
        uncertainties = summed_hist.std_devs
        uncertainties = np.append(uncertainties, uncertainties[-1])
        # ensure that error band has the same color as the plot we made earlier unless otherwise specified
        if uncertainty_color is None:
            kwargs["color"] = p[0].get_color()
        else:
            kwargs["color"] = uncertainty_color
        ax.fill_between(
            summed_hist.binning.bin_edges,
            np.clip(repeated_nom_values(summed_hist) - uncertainties, 0, None),
            repeated_nom_values(summed_hist) + uncertainties,
            alpha=0.6,
            step="post",
            label=uncertainty_label,
            **kwargs,
        )
        return ax
