# old_signal_ana_unblinding.py
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

from matplotlib import pyplot as plt
import numpy as np

sys.path.append("../")
import logging
from microfit.fileio import from_json, to_json
from microfit.analysis import MultibandAnalysis
from microfit.parameters import Parameter, ParameterSet
from unblinding_functions import (
    plot_chi2_distribution,
)
logging.basicConfig(level=logging.INFO)

# %%
config_file = "../config_files/old_model_ana_normalization_test.toml"
output_dir = "../old_model_ana_normtest_output/"

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)

# %%
# Set channel titles to non-jargon titles for publication
override_channel_titles = {
    "NPBDT": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT": "1e0p0$\\pi$ $\\nu_e$ selection",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$ $\\nu_\\mu$ selection",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$ $\\nu_\\mu$ selection",
    "TWOSHR": "NC $\\pi^0$ selection",
    "NUMUCRT": "inclusive $\\nu_\\mu$ selection"
}

hist_plot_figsize = (5.1, 5)
# %%
def run_unblinding(signal_channels, constraint_channels, plot_suffix, plot_title, with_fc=True):
    chi2_results_file = f"chi2_distribution_{plot_suffix}.json"

    h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
    h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

    analysis.signal_channels = signal_channels
    analysis.constraint_channels = constraint_channels
    analysis.set_parameters(h1_params)
    save_path = os.path.join(output_dir, "pre_fit")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analysis.plot_signals(
        include_multisim_errors=True,
        use_sideband=True,
        separate_figures=True,
        add_precomputed_detsys=True,
        save_path=save_path,
        show_chi_square=True,
        show_data_mc_ratio=True,
        figsize=hist_plot_figsize,
        override_channel_titles=override_channel_titles,
    )
    analysis.set_parameters(h0_params)
    data_hist = analysis.get_data_hist()
    assert data_hist is not None
    total_mc_hist = analysis.get_mc_hist(
        include_multisim_errors=True, use_sideband=True, add_precomputed_detsys=True
    )
    print("Data histogram")
    print(data_hist.bin_counts)
    print(data_hist.std_devs)
    print("Total MC histogram")
    print(total_mc_hist.bin_counts)
    print(total_mc_hist.std_devs)

    chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)

    if not os.path.exists(os.path.join(output_dir, chi2_results_file)):
        chi2_dict = analysis.get_chi_square_distribution(
            h0_params=h0_params, n_trials=100000, run_fit=False
        )
        to_json(os.path.join(output_dir, chi2_results_file), chi2_dict)
    else:
        chi2_dict = from_json(os.path.join(output_dir, chi2_results_file))
    plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_h0, plot_suffix, plot_title)
    
# %%
run_unblinding(
    signal_channels=["NPBDT"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    plot_suffix="npbdt",
    plot_title="1eNp0$\\pi$",
)
# %%
run_unblinding(
    signal_channels=["ZPBDT"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    plot_suffix="zpbdt",
    plot_title="1e0p0$\\pi$",
)
# # %%

# %%
