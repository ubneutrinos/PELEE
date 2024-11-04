"""Module to plot histograms for runs of data and simulation."""

from typing import List, Optional, Tuple
import matplotlib as mpl
mpl.rc('hatch', linewidth=0.5)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
from scipy.stats import chi2
import itertools
import matplotlib.pyplot as plt
from .histogram import (
    RunHistGenerator,
    Binning,
    MultiChannelBinning,
    MultiChannelHistogram,
    Histogram,
)
from . import selections
from .statistics import chi_square as chi_square_func
import warnings


class RunHistPlotter:
    def __init__(self, run_hist_generator, selection_title=None):
        self.title = selection_title
        self.run_hist_generator = run_hist_generator

    def get_selection_title(self, selection, preselection):
        warnings.warn(
            "This class method has been replaced with the function in the selections module.",
            DeprecationWarning,
        )
        return selections.get_selection_title(selection, preselection)

    def get_pot_label(self, scale_to_pot, has_data=True, data_pot=None):
        data_pot = data_pot or self.run_hist_generator.data_pot
        if data_pot is None:
            return None
        if scale_to_pot is None:
            pot_in_sci_notation = "{:.2e}".format(data_pot)
            base, exponent = pot_in_sci_notation.split("e")
            return f"${base} \\times 10^{{{int(exponent)}}}$ POT"
        else:
            pot_in_sci_notation = "{:.2e}".format(scale_to_pot)
            base, exponent = pot_in_sci_notation.split("e")
            return f"MC scaled to ${base} \\times 10^{{{int(exponent)}}}$ POT"

    def plot(
        self,
        category_column="dataset_name",
        include_multisim_errors=None,
        show_chi_square=False,
        add_ext_error_floor=None,
        smooth_ext_histogram=False,
        show_data_mc_ratio=False,
        use_sideband=None,
        ax=None,
        ax_ratio=None,
        scale_to_pot=None,
        uncertainty_color="gray",
        stacked=True,
        show_total=True,
        channel=None,
        add_precomputed_detsys=False,
        print_tot_pred_norm=False,
        title=None,
        data_pot=None,
        show_data=True,
        separate_signal=True,
        run_title=None,
        legend_cols=3,
        legend_kwargs=None,
        draw_legend=True,
        sums_in_legend=True,
        extra_text=None,
        extra_text_location="left",
        figsize=(6, 4),
        mb_label_location="right",
        mb_preliminary=True,
        signal_label=None,
        show_signal_in_ratio=False,
        signal_color="red",
        **kwargs,
    ) -> Tuple[plt.Axes, Optional[plt.Axes]]:
        gen = self.run_hist_generator

        def flatten(hist) -> Histogram:
            if isinstance(hist, MultiChannelHistogram):
                if channel is None:
                    return hist.get_unrolled_histogram()
                else:
                    return hist[channel]
            else:
                return hist

        # we want the uncertainty defaults of the generator to be used if the user doesn't specify
        gen_defaults = gen.uncertainty_defaults
        if include_multisim_errors is None:
            include_multisim_errors = gen_defaults.get("include_multisim_errors", False)
        if add_ext_error_floor is None:
            add_ext_error_floor = gen_defaults.get("add_ext_error_floor", True)
        if use_sideband is None:
            use_sideband = gen_defaults.get("use_sideband", False)
        ext_hist = gen.get_data_hist(
            type="ext",
            add_error_floor=add_ext_error_floor,
            scale_to_pot=scale_to_pot,
            smooth_ext_histogram=smooth_ext_histogram,
        )
        assert isinstance(ext_hist, Histogram)
        if ext_hist is not None:
            ext_hist.tex_string = "Cosmics"
            ext_hist = flatten(ext_hist)

        mc_hists = gen.get_mc_hists(
            category_column=category_column,
            include_multisim_errors=False,
            scale_to_pot=scale_to_pot,
        )
        signal_hist = None
        no_signal_query = None
        if separate_signal:
            if 111 in mc_hists:
                no_signal_query = f"{category_column} != 111"
                signal_hist = flatten(mc_hists.pop(111))
            elif "lee" in mc_hists:
                no_signal_query = f"{category_column} != 'lee'"
                signal_hist = flatten(mc_hists.pop("lee"))
            else:
                warnings.warn("No signal category found in the MC hists. Not separating signal.")
        background_hists = list(mc_hists.values())
        if ext_hist is not None:
            background_hists.append(ext_hist)
        background_hists = [flatten(hist) for hist in background_hists]
        extra_query = no_signal_query if separate_signal else None
        total_mc_hist = gen.get_mc_hist(
            include_multisim_errors=include_multisim_errors,
            scale_to_pot=scale_to_pot,
            use_sideband=use_sideband,
            add_precomputed_detsys=add_precomputed_detsys,
            extra_query=extra_query,
        )
        total_pred_hist = flatten(total_mc_hist)
        total_pred_hist.tex_string = "Total (MC)"
        if ext_hist is not None:
            total_pred_hist += ext_hist
            total_pred_hist.tex_string = "Total predicted"
        if use_sideband:
            total_pred_hist.tex_string += ",\nconstrained"
        # This should not be the method to blind the analysis! The only purpose of this
        # flag is to hide the data in plots where all the data bin counts have been set to
        # zero. This happens inside a multi-band analysis, where not all bands might be
        # blinded. In that case, the analysis fills in the blinded histograms with zeros,
        # and that can give the wrong chi-square value if the data is shown in the plot.
        data_hist = flatten(gen.get_data_hist()) if show_data else None
        if title is None:
            if hasattr(total_pred_hist.binning, "selection_tex"):
                title = total_pred_hist.binning.selection_tex
            else:
                title = self.title

        if print_tot_pred_norm:
            print(
                "print_tot_pred_norm:",
                total_mc_hist.bin_counts / np.sum(total_mc_hist.bin_counts),
            )

        if show_data_mc_ratio:
            if ax is None and ax_ratio is None:
                fig, (ax, ax_ratio) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                    constrained_layout=True,
                    figsize=figsize,
                )  # type: ignore
            else:
                assert (
                    ax is not None and ax_ratio is not None
                ), "Must provide both ax and ax_ratio to show data/mc ratio"
        if show_chi_square:
            assert not scale_to_pot, "Can't show chi square when scaling to POT"
            assert data_hist is not None, "Can't show chi square when no data is available"
            chi_square = chi_square_func(
                data_hist.bin_counts,
                total_pred_hist.bin_counts,
                total_pred_hist.covariance_matrix,
            )
        else:
            chi_square = None
        ax = self._plot(
            total_pred_hist,
            background_hists,
            title=title,
            data_hist=data_hist,
            ax=ax,
            scale_to_pot=scale_to_pot,
            chi_square=chi_square,
            stacked=stacked,
            show_total=show_total,
            data_pot=data_pot,
            signal_hist=signal_hist,
            run_title=run_title,
            legend_cols=legend_cols,
            legend_kwargs=legend_kwargs,
            draw_legend=draw_legend,
            sums_in_legend=sums_in_legend,
            extra_text=extra_text,
            extra_text_location=extra_text_location,
            figsize=figsize,
            mb_label_location=mb_label_location,
            mb_preliminary=mb_preliminary,
            signal_label=signal_label,
            signal_color=signal_color,
            **kwargs,
        )
        if not show_data_mc_ratio:
            if total_pred_hist.binning.variable_tex is not None:
                ax.set_xlabel(total_pred_hist.binning.variable_tex)
            return ax, None

        assert ax_ratio is not None, "Must provide ax_ratio to show data/mc ratio"
        # plot data/mc ratio
        # The way this is typically shown is to have the MC prediction divided by its central
        # data with error bands to show the MC uncertainty, and then to overlay the data points
        # with error bars.
        mc_nominal = total_mc_hist.bin_counts
        mc_error_band = flatten(total_mc_hist / mc_nominal)
        data_mc_ratio = flatten(data_hist / total_pred_hist.bin_counts)

        self.plot_hist(
            mc_error_band,
            ax=ax_ratio,
            show_errorband=True,
            as_errorbars=False,
            color="k",
            uncertainty_color=uncertainty_color,
        )
        self.plot_hist(
            data_mc_ratio,
            ax=ax_ratio,
            show_errorband=False,
            as_errorbars=True,
            color="k",
        )
        if show_signal_in_ratio and signal_hist is not None:
            y_bkg = total_pred_hist.bin_counts
            y_sig = signal_hist.bin_counts
            y_sig_ratio = (y_sig + y_bkg) / y_bkg
            
            y_sig_ratio = np.append(y_sig_ratio, y_sig_ratio[-1])

            ax_ratio.step(
                signal_hist.binning.bin_edges,
                y_sig_ratio,
                where="post",
                color=signal_color,
                linestyle="--",
                lw=2,
            )

        ax_ratio.set_ylabel("Ratio w.r.t. MC")
        ax_ratio.set_xlabel(total_pred_hist.binning.variable_tex)

        return ax, ax_ratio

    def _plot(
        self,
        total_pred_hist,
        background_hists: List[Histogram],
        title=None,
        data_hist=None,
        ax=None,
        show_errorband=True,
        uncertainty_color="gray",
        uncertainty_label="Uncertainty",
        scale_to_pot=None,
        chi_square=None,
        stacked=True,
        show_total=True,
        total_linestyle="--",
        show_total_unconstrained=True,
        data_pot=None,
        signal_hist: Optional[Histogram] = None,
        run_title=None,
        include_empty_hists=False,
        legend_cols=3,
        legend_kwargs=None,
        draw_legend=True,
        sums_in_legend=True,
        extra_text=None,
        extra_text_location="left",
        figsize=(6, 4),
        mb_label_location="right",
        mb_preliminary=True,
        signal_label=None,
        signal_color="red",
        **kwargs,
    ):
        if not include_empty_hists:
            background_hists = [hist for hist in background_hists if hist.sum() > 0]
        if stacked:
            ax = self.plot_stacked_hists(
                background_hists,
                ax=ax,
                show_errorband=False,
                figsize=figsize,
                show_counts=sums_in_legend,
                show_total=show_total_unconstrained,
                total_label="Total predicted,\nunconstrained",
                **kwargs,
            )
            if signal_hist is not None and signal_hist.sum() != 0:
                # Plot signal on top of total prediction (incl. constraints)
                y_bkg = total_pred_hist.bin_counts
                y_sig = signal_hist.bin_counts
                # Repeat the last element so that we can make a step plot
                y_bkg = np.append(y_bkg, y_bkg[-1])
                y_sig = np.append(y_sig, y_sig[-1])
                ax.step(
                    signal_hist.binning.bin_edges,
                    y_bkg + y_sig,
                    where="post",
                    color=signal_color,
                    linestyle="--",
                    lw=2,
                )
                # Add vertical lines to "cap off" the signal at the ends and connect it
                # to the background
                ax.vlines(
                    signal_hist.binning.bin_edges[0],
                    y_bkg[0],
                    y_bkg[0] + y_sig[0],
                    color=signal_color,
                    linestyle="--",
                    lw=2,
                )
                ax.vlines(
                    signal_hist.binning.bin_edges[-1],
                    y_bkg[-1],
                    y_bkg[-1] + y_sig[-1],
                    color=signal_color,
                    linestyle="--",
                    lw=2,
                )
        else:
            for background_hist in background_hists:
                ax = self.plot_hist(
                    background_hist,
                    ax=ax,
                    show_errorband=False,
                    sums_in_legend=sums_in_legend,
                    **kwargs,
                )
        if show_total:
            ax = self.plot_hist(
                total_pred_hist,
                ax=ax,
                show_errorband=show_errorband,
                uncertainty_color=uncertainty_color,
                uncertainty_label=uncertainty_label,
                linestyle=total_linestyle,
                color="k",
                lw=2,
            )
        assert ax is not None
        if scale_to_pot is None:
            if data_hist is not None:  # skip if no data (as is the case for blind analysis)
                # rescaling data to a different POT doesn't make sense
                if sums_in_legend:
                    data_label = f"Data: {data_hist.sum():.1f}"
                else:
                    data_label = "Data"
                ax = self.plot_hist(
                    data_hist,
                    ax=ax,
                    label=data_label,
                    color="black",
                    as_errorbars=True,
                    lw=1.50,
                    **kwargs,
                )
        chi2_text = None
        if chi_square is not None:
            n_bins = total_pred_hist.binning.n_bins
            # calculate the p-value corresponding to the observed chi-square
            # and dof using scipy
            p_value = 1 - chi2.cdf(chi_square, n_bins)
            chi2_label = rf"$\chi^2$ = {chi_square:.1f}, p={p_value*100:.1f}%"
            chi2_text = ax.text(
                0.03,
                0.92,
                chi2_label,
                ha="left",
                va="baseline",
                transform=ax.transAxes,
                fontsize=9,
            )
        if run_title is not None:
            if title is not None:
                title = f"{run_title}, {title}"
            else:
                title = run_title
        pot_label = self.get_pot_label(scale_to_pot, data_pot=data_pot)
        mb_label = "MicroBooNE"
        if mb_preliminary:
            mb_label += " preliminary"
        if pot_label is not None:
            mb_label += f", {pot_label}"
        # if title is not None:
        #     title += "\n" + mb_label
        # else:
        #     title = mb_label
        
        title_text = None
        if title is not None:
            title_text = ax.text(
                0.97,
                0.92,
                title,
                ha="right",
                va="baseline",
                transform=ax.transAxes,
                fontsize=9,
            )
        if extra_text is not None:
            if title_text is not None:
                # Write the extra text as an annotation below the
                # existing text. We set the "title_text" to the newly
                # created text, so that the MB label will be annotated
                # underneath.
                title_text = ax.annotate(
                    extra_text,
                    xy=(1, 0),
                    xycoords=title_text,
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="right",
                    # Using top alignment here, because this text could
                    # have more than one line.
                    va="top",
                    fontsize=9,
                )
            else:
                title_text = ax.text(
                    0.97,
                    0.92,
                    extra_text,
                    ha="right",
                    va="baseline",
                    transform=ax.transAxes,
                    fontsize=9,
                )
        if title_text is not None:
            # Place the microboone and POT label right below this text
            # using annotate
            if mb_label_location == "right":
                mb_label_text = ax.annotate(
                    mb_label,
                    xy=(1, 0),
                    xycoords=title_text,
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="right",
                    va="baseline",
                    fontsize=8,
                )
            elif mb_label_location == "left":
                if chi2_text is not None:
                    # place below chi2 text, left aligned
                    mb_label_text = ax.annotate(
                        mb_label,
                        xy=(0, 0),
                        xycoords=chi2_text,
                        xytext=(0, -10),
                        textcoords="offset points",
                        ha="left",
                        va="baseline",
                        fontsize=8,
                    )
                else:
                    # place where the chi2 text would have been
                    mb_label_text = ax.text(
                        0.03,
                        0.92,
                        mb_label,
                        ha="left",
                        va="baseline",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
        ax.set_ylabel("Events")
        # Get existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        if signal_hist is not None and signal_hist.sum() != 0:
            # Create a Line2D object for the new legend entry
            if signal_label is None:
                signal_label = signal_hist.tex_string
            red_line = Line2D(
                [],
                [],
                color=signal_color,
                linestyle="--",
                linewidth=2,
                label=f"{signal_label}: {signal_hist.sum():.1f}",
            )
            # Append new handle and label
            handles.append(red_line)
            if sums_in_legend:
                labels.append(f"{signal_label}: {signal_hist.sum():.1f}")
            else:
                labels.append(signal_label)

        default_legend_kwargs = dict(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            ncol=legend_cols,
            mode="expand",
            borderaxespad=0.0,
            bbox_transform=ax.transAxes,
            fontsize="small",
            frameon=False,
        )
        if legend_kwargs is None:
            legend_kwargs = default_legend_kwargs
        legend_kwargs.update(handles=handles, labels=labels)
        if draw_legend:
            ax.legend(**legend_kwargs)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        # Set x-limits to the outermost bin edges
        bin_edges = total_pred_hist.binning.bin_edges
        ax.set_xlim((bin_edges[0], bin_edges[-1]))

        return ax

    def plot_hist(
        self,
        hist: Histogram,
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
        bin_counts = hist.bin_counts
        bin_counts[bin_counts <= 0] = np.nan
        bin_edges = hist.binning.bin_edges
        label = kwargs.pop("label", hist.tex_string)
        color = kwargs.pop("color", hist.color)
        if "linewidth" in kwargs:
            linewidth = kwargs.pop("linewidth", 1.0)
        elif "lw" in kwargs:
            linewidth = kwargs.pop("lw", 1.0)
        else:
            linewidth = 1.0
        if as_errorbars:
            bin_widths = np.diff(bin_edges)
            ax.errorbar(
                hist.binning.bin_centers,
                bin_counts,
                xerr=bin_widths / 2,
                yerr=hist.std_devs,
                linestyle="none",
                marker=".",
                label=label,
                color=color,
                linewidth=linewidth,
                **kwargs,
            )
            return ax
        # Be sure to repeat the last bin count
        bin_counts = np.append(bin_counts, bin_counts[-1])
        default_linestyle = "--" if hist.hatch == "///" else "-"
        linestyle = kwargs.pop("linestyle", default_linestyle)
        p = ax.step(
            bin_edges,
            bin_counts,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs,
        )
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
            alpha=1.0,
            step="post",
            label=uncertainty_label,
            linewidth=0.0,
            hatch="///////",
            facecolor="none",
            edgecolor=(0.1, 0.1, 0.1),
            # **kwargs,
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
        show_total=True,
        total_label="Total",
        figsize=(6, 4),
        **kwargs,
    ):
        """Plot a stack of histograms."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        x = hists[0].binning.bin_edges

        def repeated_nom_values(hist):
            # repeat the last bin count
            y = hist.bin_counts
            y = np.append(y, y[-1])
            return y

        # to use plt.stackplot,  we need y to be a 2D array of shape (N, len(x))
        y = np.array([repeated_nom_values(hist) for hist in hists])
        labels = [hist.tex_string for hist in hists]
        if show_counts:
            labels = [f"{label}: {hist.sum():.0f}" for label, hist in zip(labels, hists)]
        colors = None
        colors = [hist.color for hist in hists]
        # Hatches may be None
        hatches = [hist.hatch for hist in hists]

        stackplot(
            ax,
            x,
            y,
            step="post",
            labels=labels,
            colors=colors,
            hatches=hatches,
            **kwargs,
        )
        if not show_total:
            return ax
        # plot uncertainties as a shaded region, but only for the sum of all hists
        summed_hist = sum(hists)
        assert isinstance(summed_hist, Histogram)
        # show sum as black line
        p = ax.step(
            summed_hist.binning.bin_edges,
            repeated_nom_values(summed_hist),
            where="post",
            color="k",
            linestyle="-",
            lw=1.5,
            label=total_label,
        )
        if not show_errorband:
            return ax
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


def stackplot(axes, x, *args, labels=(), colors=None, hatches=None, **kwargs):
    """
    Draw a stacked area plot.

    This function is copied from matplotlib.Axes.stackplot, but with the
    addition of the `hatches` kwarg. The baseline option has been
    removed, as we only need the zero baseline.

    Parameters
    ----------
    x : (N,) array-like

    y : (M, N) array-like
        The data is assumed to be unstacked. Each of the following
        calls is legal::

            stackplot(x, y)           # where y has shape (M, N)
            stackplot(x, y1, y2, y3)  # where y1, y2, y3, y4 have length N

    labels : list of str, optional
        A sequence of labels to assign to each data series. If unspecified,
        then no labels will be applied to artists.

    colors : list of color, optional
        A sequence of colors to be cycled through and used to color the stacked
        areas. The sequence need not be exactly the same length as the number
        of provided *y*, in which case the colors will repeat from the
        beginning.

        If not specified, the colors from the Axes property cycle will be used.

        If the ``hatch`` property is not None for a member of the stack, the
        ``color`` will be passed to the ``edgecolor`` keyword argument for
        patch objects, and the ``facecolor`` will be set to "none".
        Otherwise, the given color is used as the ``facecolor``.

    hatches : list of str, optional
        A sequence of hatches to be cycled through and used to fill the
        stacked areas. The sequence need not be exactly the same length as
        the number of provided *y*, in which case the hatches will repeat
        from the beginning.

        If not specified, no hatches will be used.

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between`.

    Returns
    -------
    list of `.PolyCollection`
        A list of `.PolyCollection` instances, one for each element in the
        stacked area plot.
    """

    y = np.vstack(args)

    labels = iter(labels)
    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = (axes._get_lines.get_next_color() for _ in y)
    if hatches is not None:
        hatches = itertools.cycle(hatches)
    else:
        hatches = itertools.repeat(None)

    # Assume data passed has not been 'stacked', so stack it here.
    # We'll need a float buffer for the upcoming calculations.
    stack = np.cumsum(y, axis=0, dtype=np.promote_types(y.dtype, np.float32))
    first_line = 0.0

    # Color between x = 0 and the first array.
    color = next(colors)
    hatch = next(hatches)
    if hatch is not None:
        edgecolor = color
        facecolor = "none"
    else:
        edgecolor = None
        facecolor = color
    coll = axes.fill_between(
        x,
        first_line,
        stack[0, :],
        facecolor=facecolor,
        edgecolor=edgecolor,
        hatch=hatch,
        label=next(labels, None),
        **kwargs,
    )
    coll.sticky_edges.y[:] = [0]
    r = [coll]

    # Color between array i-1 and array i
    for i in range(len(y) - 1):
        color = next(colors)
        hatch = next(hatches)
        if hatch is not None:
            edgecolor = color
            facecolor = "none"
        else:
            edgecolor = None
            facecolor = color
        r.append(
            axes.fill_between(
                x,
                stack[i, :],
                stack[i + 1, :],
                facecolor=facecolor,
                edgecolor=edgecolor,
                hatch=hatch,
                label=next(labels, None),
                linewidth=0,
                **kwargs,
            )
        )
    return r
