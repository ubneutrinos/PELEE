"""This script produces the official plots for the PeLEE paper."""

# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

from matplotlib import pyplot as plt

sys.path.append("../")
import logging
from microfit.analysis import MultibandAnalysis
from microfit.run_plotter import RunHistPlotter
logging.basicConfig(level=logging.INFO)

# %%

override_channel_titles = {
    "NPBDT_SHR_E": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT_SHR_E": "1e0p0$\\pi$ $\\nu_e$ selection",
    "NPBDT_SHR_COSTHETA": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT_SHR_COSTHETA": "1e0p0$\\pi$ $\\nu_e$ selection",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$ $\\nu_\\mu$ selection",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$ $\\nu_\\mu$ selection",
    "TWOSHR": "NC $\\pi^0$ selection",
    "NPBDT": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT": "1e0p0$\\pi$ $\\nu_e$ selection",
}

override_channel_titles_short = {
    "NPBDT_SHR_E": "1eNp0$\\pi$",
    "ZPBDT_SHR_E": "1e0p0$\\pi$",
    "NPBDT_SHR_COSTHETA": "1eNp0$\\pi$",
    "ZPBDT_SHR_COSTHETA": "1e0p0$\\pi$",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$",
    "TWOSHR": "NC $\\pi^0$",
    "NPBDT": "1eNp0$\\pi$",
    "ZPBDT": "1e0p0$\\pi$",
}

# %%
def plot_signal_model(
    analysis,
    signal_channels,
    show_data=True,
    show_chi_square=False,
    figsize=(5.2, 5.2),
    mb_label_location="right"
):
    if analysis is None:
        raise ValueError("Analysis object must be provided.")

    analysis.signal_channels = signal_channels

    fig, ax = plt.subplots(
        len(signal_channels), 1, sharex=True, constrained_layout=False, figsize=figsize
    )
    plt.subplots_adjust(hspace=0.1)

    plotter = RunHistPlotter(analysis)
    plot_kwargs = dict(
        category_column="paper_category",
        show_chi_square=show_chi_square,
        sums_in_legend=False,
        add_precomputed_detsys=True,
        show_data=show_data,
    )

    if len(signal_channels) > 1:
        for i, channel in enumerate(signal_channels):
            plotter.plot(
                channel=channel,
                ax=ax[i],
                draw_legend=(i == 0),
                title=override_channel_titles.get(channel, None),
                data_pot=analysis._get_pot_for_channel(channel),
                mb_label_location=mb_label_location,
                **plot_kwargs,  # type: ignore
            )
            if i < len(signal_channels) - 1:
                ax[i].set_xlabel("")
    else:
        plotter.plot(channel=signal_channels[0], ax=ax, **plot_kwargs)  # type: ignore

    fig.tight_layout()
    return fig


def plot_sidebands(analysis):
    if analysis is None:
        raise ValueError("Analysis object must be provided.")

    analysis.plot_sideband = True

    fig, ax = plt.subplots(
        3, 1, sharex=True, constrained_layout=False, figsize=(5.2, 7.2)
    )
    plt.subplots_adjust(hspace=0.1)

    plotter = RunHistPlotter(analysis)
    plot_kwargs = dict(
        category_column="category",
        show_chi_square=False,
        sums_in_legend=False,
        add_precomputed_detsys=True,
        use_sideband=False,
    )

    channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
    for i, channel in enumerate(channels):
        plotter.plot(
            channel=channel,
            ax=ax[i],
            draw_legend=(i == 0),
            title=override_channel_titles.get(channel, None),
            data_pot=analysis._get_pot_for_channel(channel),
            **plot_kwargs,  # type: ignore
        )
        if i < len(channels) - 1:
            ax[i].set_xlabel("")

    analysis.plot_sideband = False
    fig.tight_layout()
    return fig

# ------ Old Model Analysis --------
# %%
config_file = "../config_files/old_model_ana_with_detvars.toml"
output_dir = "../old_model_ana_output_with_3a/"

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)
# %%
analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=False,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data.pdf",
    figsize=(5, 4),
    override_channel_titles=override_channel_titles
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
    figsize=(5, 5),
    override_channel_titles=override_channel_titles
)
# %%
fig = plot_signal_model(
    analysis=analysis,
    signal_channels=["NPBDT", "ZPBDT"],
    show_data=True,
    show_chi_square=False,
)
fig.savefig("paper_histograms/old_signal_model_paper_histograms.pdf")
fig.savefig("paper_histograms/old_signal_model_paper_histograms.png", dpi=200)

# %%
fig = plot_signal_model(
    analysis=analysis,
    signal_channels=["NPBDT", "ZPBDT"],
    show_data=False,
    show_chi_square=False,
)
fig.savefig("paper_histograms/old_signal_model_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/old_signal_model_paper_histograms_no_data.png", dpi=200)
# %%
fig = plot_sidebands(analysis=analysis)
fig.savefig("paper_histograms/sidebands_paper_histograms.pdf")
fig.savefig("paper_histograms/sidebands_paper_histograms.png", dpi=200)

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
fig = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    show_data=True,
    show_chi_square=False,
)
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms.png", dpi=200)

fig = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    show_data=False,
    show_chi_square=False,
)
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_e_paper_histograms_no_data.png", dpi=200)

# %%
fig = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    show_data=True,
    show_chi_square=False,
    mb_label_location="left"
)
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.png", dpi=200)

fig = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    show_data=False,
    show_chi_square=False,
    mb_label_location="left"
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
    figsize=(5, 4),
    override_channel_titles=override_channel_titles
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
    figsize=(5, 5),
    override_channel_titles=override_channel_titles
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
    figsize=(5, 4),
    override_channel_titles=override_channel_titles
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
    figsize=(5, 5),
    override_channel_titles=override_channel_titles
)
# %%
print(new_analysis.channels)
# %%
