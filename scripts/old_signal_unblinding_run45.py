# old_signal_ana_unblinding.py
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

sys.path.append("../")
import logging
from microfit.fileio import from_json, to_json
from microfit.analysis import MultibandAnalysis
from microfit.parameters import Parameter, ParameterSet
from unblinding_functions import (
    plot_chi2_distribution,
    plot_two_hypo_result,
    plot_confidence_intervals,
    plot_confidence_interval_diagnostic,
)

logging.basicConfig(level=logging.INFO)
# %%
MAKE_DIAGNOSTIC_PLOTS = True
# %%
config_file = "../config_files/old_model_ana_with_detvars_run45.toml"
output_dir = "../old_model_ana_remerged_crt_run45_output/"

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)
# %%
print("Plotting signal channels...")
analysis.parameters["signal_strength"].value = 1.0
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "pre_fit"),
    show_chi_square=True,
    show_data_mc_ratio=True,
    figsize=(6, 6),
)

