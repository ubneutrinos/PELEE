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
def add_pot_label(ax, analysis, plotter, channel, position="right"):
    data_pot = analysis._get_pot_for_channel(channel)
    pot_text = plotter.get_pot_label(None, data_pot=data_pot)
    if position == "right":
        ax.text(
            0.97,
            0.85,
            f"MicroBooNE, {pot_text}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    elif position == "left":
        ax.text(
            0.05,
            0.85,
            f"MicroBooNE, {pot_text}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    else:
        raise ValueError(f"unknown position '{position}'")

def plot_signal_model(
    analysis,
    signal_channels,
    show_data=True,
    show_chi_square=True,
    figsize=(5.2, 5.2),
    mb_label_pos="right"
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
                **plot_kwargs,  # type: ignore
            )
            add_pot_label(ax[i], analysis, plotter, channel, position=mb_label_pos)
            if i < len(signal_channels) - 1:
                ax[i].set_xlabel("")
    else:
        plotter.plot(channel=signal_channels[0], ax=ax, **plot_kwargs)  # type: ignore
        add_pot_label(ax, analysis, plotter, signal_channels[0], position=mb_label_pos)

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
        show_chi_square=True,
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
            **plot_kwargs,  # type: ignore
        )
        add_pot_label(ax[i], analysis, plotter, channel)
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
    show_chi_square=True,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data.pdf",
    # extra_text="something something",
    figsize=(5, 4)
)
# %%
analysis.plot_signals(
    include_multisim_errors=True,
    add_precomputed_detsys=True,
    use_sideband=True,
    separate_figures=True,
    show_chi_square=True,
    show_data_mc_ratio=True,
    save_path="paper_histograms",
    filename_format="old_model_{}_with_data_and_ratio.pdf",
    # extra_text="something something",
    figsize=(5, 5)
)
# %%
fig = plot_signal_model(
    analysis=analysis,
    signal_channels=["NPBDT", "ZPBDT"],
    show_data=True,
    show_chi_square=True,
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


# Let's not forget to reset the flag after this...
analysis.plot_sideband = False


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
    show_chi_square=True,
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
    show_chi_square=True,
    mb_label_pos="left"
)
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms.png", dpi=200)

fig = plot_signal_model(
    analysis=new_analysis,
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    show_data=False,
    show_chi_square=False,
    mb_label_pos="left"
)
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms_no_data.pdf")
fig.savefig("paper_histograms/new_signal_model_shr_costheta_paper_histograms_no_data.png", dpi=200)

# %%
