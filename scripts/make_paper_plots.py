"""This script produces the official plots for the PeLEE paper."""

# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
# %%
sys.path.append("../")
import logging
from microfit.analysis import MultibandAnalysis
from microfit.run_plotter import RunHistPlotter
logging.basicConfig(level=logging.INFO)

# %%
from plot_analysis_histograms_correlations import plot_constraint_update

# %%
# Set to False since we are now making plots for the paper
PRELIMINARY = False

override_channel_titles = {
    "NPBDT_SHR_E": "1eNp0$\\pi$ selection",
    "ZPBDT_SHR_E": "1e0p0$\\pi$ selection",
    "NPBDT_SHR_COSTHETA": "1eNp0$\\pi$ selection",
    "ZPBDT_SHR_COSTHETA": "1e0p0$\\pi$ selection",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$ selection",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$ selection",
    "NUMUCRT": "$\\nu_\\mu$ selection",
    "TWOSHR": "NC $\\pi^0$ selection",
    "NPBDT": "1eNp0$\\pi$ selection",
    "ZPBDT": "1e0p0$\\pi$ selection",
    "ZPBDT_NOCRT": "1e0p0$\\pi$ selection",
}

override_channel_titles_short = {
    "NPBDT_SHR_E": "1eNp0$\\pi$",
    "ZPBDT_SHR_E": "1e0p0$\\pi$",
    "NPBDT_SHR_COSTHETA": "1eNp0$\\pi$",
    "ZPBDT_SHR_COSTHETA": "1e0p0$\\pi$",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$",
    "NUMUCRT": "$\\nu_\\mu$",
    "TWOSHR": "NC $\\pi^0$",
    "NPBDT": "1eNp0$\\pi$",
    "ZPBDT": "1e0p0$\\pi$",
    "ZPBDT_NOCRT": "1e0p0$\\pi$",
}

# %%
def plot_signal_model(
    analysis,
    signal_channels,
    show_data=True,
    show_chi_square=False,
    figsize=(5, 5.2),
    mb_label_location="right",
    **plot_kwargs
):
    if analysis is None:
        raise ValueError("Analysis object must be provided.")

    analysis.signal_channels = signal_channels

    fig, ax = plt.subplots(
        len(signal_channels),
        1,
        sharex=True,
        constrained_layout=False,
        figsize=figsize,
        gridspec_kw={"hspace": 0}
    )

    plotter = RunHistPlotter(analysis)
    default_plot_kwargs = dict(
        category_column="paper_category",
        show_chi_square=show_chi_square,
        sums_in_legend=False,
        add_precomputed_detsys=True,
        show_data=show_data,
    )
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}
    empirical_p_values = {
        # These are calculated by the unblinding scripts and transferred
        # here by hand. Look, I cannot be asked to automate this.
        "NPBDT": 0.184,
        "ZPBDT": 0.561,
        "NPBDT_SHR_E": 0.104,
        "ZPBDT_SHR_E": 0.625,
        "NPBDT_SHR_COSTHETA": 0.153,
        "ZPBDT_SHR_COSTHETA": 0.776,
    }
    chi2_values = {
        "NPBDT": 15.0,
        "ZPBDT": 9.9,
        "NPBDT_SHR_E": 23.3,
        "ZPBDT_SHR_E": 13.3,
        "NPBDT_SHR_COSTHETA": 14.4,
        "ZPBDT_SHR_COSTHETA": 6.2,
    }
    # After looping through all histograms, we want to set the x-limits
    # to the widest out of all the histograms we have shown. 
    min_xlim = np.inf
    max_xlim = -np.inf
    if len(signal_channels) > 1:
        for i, channel in enumerate(signal_channels):
            plotter.plot(
                channel=channel,
                ax=ax[i],
                draw_legend=(i == 0),
                title=override_channel_titles.get(channel, None),
                data_pot=analysis._get_pot_for_channel(channel),
                mb_label_location=mb_label_location,
                mb_preliminary=PRELIMINARY,
                **plot_kwargs,  # type: ignore
            )
            pval_text = f"$\\chi^2={chi2_values[channel]}$, $p$-value: {empirical_p_values[channel] * 100:.0f}%"
            # If the MB label is on the left, we need to move this text
            # down a bit so that it does not overlap with the MB label
            text_location = 0.91 if mb_label_location == "right" else 0.82
            ax[i].text(
                0.03,
                text_location,
                pval_text,
                ha="left",
                va="baseline",
                transform=ax[i].transAxes,
                fontsize=9,
            )
            # Get the x-limits of the axis last plotted and update the 
            # min and max
            min_xlim = min(min_xlim, ax[i].get_xlim()[0])
            max_xlim = max(max_xlim, ax[i].get_xlim()[1])
            if i < len(signal_channels) - 1:
                ax[i].set_xlabel("")
    else:
        plotter.plot(channel=signal_channels[0], ax=ax, **plot_kwargs)  # type: ignore

    if len(signal_channels) > 1:
        for a in ax[1:]:
            # Since we are removing the vertical space, it is possible that 
            # the highest y-tick label overlaps with the zero label of the 
            # previous plot. We adjust the y-limits slightly if the highest
            # y-tick label is close to the top of the plot.
            yticks = a.get_yticks()
            ylim_max = a.get_ylim()[1]
            # It appears that the last y-tick may actually be above 
            # the y-limit, so we take the highest one that is actually
            # within the y-limit
            top_tick = yticks[yticks <= ylim_max][-1]
            # If the highest tick is less than half a tick distance from the
            # top of the plot, we adjust the y-limit so that it will be
            # half a tick distance above the highest tick
            half_tick_distance = (yticks[1] - yticks[0]) / 2
            if top_tick > ylim_max - half_tick_distance:
                a.set_ylim((0, top_tick + half_tick_distance))
            # Manually set the yticks now so that the y-tick labels are not
            # changed from having made this adjustment
            a.set_yticks(yticks[yticks <= ylim_max])
    fig.tight_layout()
    return fig, ax

# %%
def plot_sidebands(analysis):
    if analysis is None:
        raise ValueError("Analysis object must be provided.")

    analysis.plot_sideband = True

    fig, ax = plt.subplots(
        3, 1, sharex=True, constrained_layout=False, figsize=(5, 7.2),
    )
    plt.subplots_adjust(hspace=0)
    plotter = RunHistPlotter(analysis)
    plot_kwargs = dict(
        category_column="paper_category_numu",
        show_chi_square=False,
        sums_in_legend=False,
        add_precomputed_detsys=True,
        use_sideband=False,
        show_total_unconstrained=False,
        total_linestyle="-",
    )

    min_xlim = np.inf
    max_xlim = -np.inf
    channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
    empirical_p_values = {
        # These are calculated by the unblinding scripts and transferred
        # here by hand. Look, I cannot be asked to automate this.
        "NUMUCRT0P0PI": 0.21,
        "NUMUCRTNP0PI": 0.10,
        "TWOSHR": 0.09,
    }
    chi2_values = {
        "NUMUCRT0P0PI": 18.0,
        "NUMUCRTNP0PI": 21.1,
        "TWOSHR": 16.8,
    }
    for i, channel in enumerate(channels):
        pval_text = f"$\\chi^2={chi2_values[channel]}$, $p$-value: {empirical_p_values[channel] * 100:.0f}%"
        plotter.plot(
            channel=channel,
            ax=ax[i],
            draw_legend=(i == 0),
            title=override_channel_titles.get(channel, None),
            data_pot=analysis._get_pot_for_channel(channel),
            mb_preliminary=PRELIMINARY,
            **plot_kwargs,  # type: ignore
        )
        ax[i].text(
            0.03,
            0.91,
            pval_text,
            ha="left",
            va="baseline",
            transform=ax[i].transAxes,
            fontsize=9,
        )
        min_xlim = min(min_xlim, ax[i].get_xlim()[0])
        max_xlim = max(max_xlim, ax[i].get_xlim()[1])
        if i < len(channels) - 1:
            ax[i].set_xlabel("")

    analysis.plot_sideband = False
    for a in ax:
        a.set_xlim((min_xlim, max_xlim))
    for a in ax[1:]:
        # Since we are removing the vertical space, it is possible that 
        # the highest y-tick label overlaps with the zero label of the 
        # previous plot. We adjust the y-limits slightly if the highest
        # y-tick label is close to the top of the plot.
        yticks = a.get_yticks()
        ylim_max = a.get_ylim()[1]
        # It appears that the last y-tick may actually be above 
        # the y-limit, so we take the highest one that is actually
        # within the y-limit
        top_tick = yticks[yticks <= ylim_max][-1]
        # If the highest tick is less than half a tick distance from the
        # top of the plot, we adjust the y-limit so that it will be
        # half a tick distance above the highest tick
        half_tick_distance = (yticks[1] - yticks[0]) / 2
        if top_tick > ylim_max - half_tick_distance:
            a.set_ylim((0, top_tick + half_tick_distance))
        # Manually set the yticks now so that the y-tick labels are not
        # changed from having made this adjustment
        a.set_yticks(yticks[yticks <= ylim_max])
    # fig.tight_layout()
    return fig

# ------ Old Model Analysis --------
# %%
config_file = "../config_files/old_model_ana_ext_bin_range.toml"
output_dir = "../old_model_ana_ext_bin_range_output/"

# make output directory
os.makedirs(output_dir, exist_ok=True)

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)
# %%
analysis.parameters["signal_strength"].value = 1.0
analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 1",
    sums_in_legend=False,
    mb_preliminary=PRELIMINARY,
)
# %%
analysis.plot_sidebands(
    category_column="paper_category_numu",
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    separate_figures=True,
    show_chi_square=False,
    show_total_unconstrained=False,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    sums_in_legend=False,
    mb_preliminary=PRELIMINARY,
    total_linestyle="-",
)
# %%
analysis.plot_sidebands(
    category_column="paper_category_numu",
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    separate_figures=True,
    show_chi_square=False,
    show_data_mc_ratio=True,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data_and_ratio.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    sums_in_legend=False,
    mb_preliminary=PRELIMINARY,
    total_linestyle="-",
)
# %%
analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    show_data_mc_ratio=True,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data_and_ratio.pdf",
    figsize=(4.3, 4.5),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 1",
    sums_in_legend=False,
    mb_preliminary=PRELIMINARY,
)
# %%
fig, ax = plot_signal_model(
    analysis=analysis,
    signal_channels=["NPBDT", "ZPBDT"],
    show_data=True,
    show_chi_square=False,
    signal_label="LEE signal\nmodel 1"
)
fig.savefig("paper_histograms/old_signal_model_paper_histograms.pdf")
fig.savefig("paper_histograms/old_signal_model_paper_histograms.png", dpi=200)
for a in ax:
    min_xlim, max_xlim = a.get_xlim()
    a.set_xlim((min_xlim, max_xlim))
    # Draw the off-signal regions as shaded regions, from min_xlim
    # of the plot to 0.15, and from 1.55 to max_xlim
    a.axvspan(min_xlim, 0.15, color="gray", alpha=0.1)
    a.axvspan(1.55, max_xlim, color="gray", alpha=0.1)
    # make a mixed transformation where the x-coordinate is in 
    # data coordinates and the y-coordinate in axes coordinates
    transform = transforms.blended_transform_factory(
        a.transData, a.transAxes
    )
    a.text(
        0.18,
        0.92,
        "Signal region",
        ha="left",
        va="baseline",
        transform=transform,
        fontsize=8,
    )
fig.savefig("paper_histograms/old_signal_model_paper_histograms_marked_signal_region.pdf")
fig.savefig("paper_histograms/old_signal_model_paper_histograms_marked_signal_region.png", dpi=200)

# %%
fig, _ = plot_signal_model(
    analysis=analysis,
    signal_channels=["NPBDT", "ZPBDT"],
    show_data=False,
    show_chi_square=False,
)
fig.savefig("paper_histograms/old_signal_model_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/old_signal_model_paper_histograms_no_data.png", dpi=200)
# %%
fig = plot_sidebands(analysis=analysis)
fig.savefig("paper_histograms/sidebands_paper_histograms.pdf", bbox_inches="tight")
fig.savefig("paper_histograms/sidebands_paper_histograms.png", dpi=200, bbox_inches="tight")

# %%
fig, ax = analysis.plot_correlation(
    override_selection_tex=override_channel_titles_short,
    labels_on_axes=["x", "y"],
    figsize=(5.5, 4.5),
    colorbar_kwargs={"shrink": 1.0},
    use_variable_label=False
)
ax.set_title("")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms.pdf")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms.png", dpi=200)

# %%
fig, ax = analysis.plot_correlation(
    override_selection_tex=override_channel_titles_short,
    labels_on_axes=["x", "y"],
    figsize=(5.5, 4.5),
    colorbar_kwargs={"shrink": 1.0},
    use_variable_label=False,
    detsys_only_diagonal=True
)
ax.set_title("")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms_diagonal_detsys.pdf")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms_diagonal_detsys.png", dpi=200)

# %%
plot_constraint_update(analysis, "paper_histograms", override_channel_titles=override_channel_titles, figsize=(4, 3.5))

# %%
plot_constraint_update(analysis, "paper_histograms", override_channel_titles=override_channel_titles, figsize=(4.5, 3), detsys_only_diagonal=True)

# %%
# Plot old constraints
previous_constraint_channels = analysis.constraint_channels
previous_signal_channels = analysis.signal_channels
analysis.constraint_channels = ["NUMUCRT"]
analysis.signal_channels = ["NPBDT", "ZPBDT_NOCRT"]

fig, ax = analysis.plot_correlation(
    override_selection_tex=override_channel_titles_short,
    labels_on_axes=["x", "y"],
    figsize=(5.5, 4.5),
    colorbar_kwargs={"shrink": 1.0},
    use_variable_label=False
)
ax.set_title("")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms_old_constraints.pdf")
fig.savefig("paper_histograms/correlation_matrix_paper_histograms_old_constraints.png", dpi=200)

# Plot the sideband channel
analysis.plot_sidebands(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    separate_figures=True,
    show_chi_square=False,
    save_path="paper_histograms",
    filename_format="old_constraints_{}_with_data.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    sums_in_legend=False,
    mb_preliminary=PRELIMINARY,
)

plot_constraint_update(analysis, "paper_histograms/old_constraints", override_channel_titles=override_channel_titles, figsize=(4.5, 3))
analysis.constraint_channels = previous_constraint_channels
analysis.signal_channels = previous_signal_channels


# %%
# ---------- New Model Analysis ----------
config_file = "../config_files/full_ana_with_detvars.toml"
output_dir = "../full_ana_with_run3a_output/"

new_analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)
# %%
signal_color = "blue"

# %%
fig, _ = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    show_data=True,
    show_chi_square=False,
    signal_label="LEE signal\nmodel 2",
    signal_color=signal_color
)
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms.png", dpi=200)

fig, _ = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    show_data=False,
    show_chi_square=False,
    signal_label="LEE signal\nmodel 2",
    signal_color=signal_color
)
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms_no_data.png", dpi=200)

# %%
fig, _ = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    show_data=True,
    show_chi_square=False,
    mb_label_location="left",
    signal_label="LEE signal\nmodel 2",
    signal_color=signal_color
)
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.png", dpi=200)

fig, _ = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    show_data=False,
    show_chi_square=False,
    mb_label_location="left",
    signal_label="LEE signal\nmodel 2",
    signal_color=signal_color
)
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms_no_data.png", dpi=200)

# %%
new_analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
new_analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    save_path="paper_histograms",
    filename_format="new_model_{}_with_data.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 2",
    sums_in_legend=False,
    signal_color=signal_color
)
new_analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    show_data_mc_ratio=True,
    save_path="paper_histograms",
    filename_format="new_model_{}_with_data_and_ratio.pdf",
    figsize=(4.3, 4.5),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 2",
    sums_in_legend=False,
    signal_color=signal_color
)
# %%
new_analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
new_analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    save_path="paper_histograms",
    filename_format="new_model_{}_with_data.pdf",
    figsize=(4.3, 3.6),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 2",
    sums_in_legend=False,
    signal_color=signal_color
)
new_analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    show_data_mc_ratio=True,
    save_path="paper_histograms",
    filename_format="new_model_{}_with_data_and_ratio.pdf",
    figsize=(4.3, 4.5),
    override_channel_titles=override_channel_titles,
    signal_label="LEE signal\nmodel 2",
    sums_in_legend=False,
    signal_color=signal_color
)
# %%
print(new_analysis.channels)
# %%
