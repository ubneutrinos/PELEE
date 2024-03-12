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
config_file = "../config_files/old_model_ana_opendata_bnb.toml"
output_dir = "../old_model_ana_opendata_bnb_output/"

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
    save_path=output_dir,
    show_chi_square=True,
    show_data_mc_ratio=True,
)
# %%
best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
print(f"Best fit chi2: {best_fit_chi2}")
print(f"Best fit parameters: {best_fit_parameters}")

h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)
chi2_at_h1 = analysis.get_chi2_at_hypothesis(h1_params)
delta_chi2 = chi2_at_h0 - chi2_at_h1
print(f"Delta chi2: {delta_chi2}")

plot_chi2_distribution(
    output_dir, "chi2_distribution.json", chi2_at_h0, "rec_nu_energy", "Reco. $\\nu$ Energy"
)
plot_two_hypo_result(
    output_dir,
    "two_hypo_result_new_constraints.json",
    delta_chi2,
    "rec_nu_energy",
    "Reco. $\\nu$ Energy",
)

analysis.set_parameters(best_fit_parameters)
print("Plotting signal channels at best fit point...")
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=output_dir,
    show_chi_square=True,
    show_data_mc_ratio=True,
    separate_signal=False,
)
# %%
fc_scan_results = from_json(os.path.join(output_dir, "fc_scan_results.json"))
scan_points = fc_scan_results["scan_points"]
scan_chi2 = analysis.scan_chi2(fc_scan_results["parameter_name"], scan_points=scan_points)
delta_chi2_scan = scan_chi2 - best_fit_chi2
fc_scan_results["delta_chi2_scan"] = delta_chi2_scan
# %%
plot_confidence_intervals(output_dir, fc_scan_results, "rec_nu_energy")
# %%
if MAKE_DIAGNOSTIC_PLOTS:
    plot_confidence_interval_diagnostic(fc_scan_results)
