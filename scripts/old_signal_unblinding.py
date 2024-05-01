# old_signal_ana_unblinding.py
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

from matplotlib import pyplot as plt

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
print("Plotting signal channels...")
analysis.parameters["signal_strength"].value = 1.0
# make the pre_fit output directory if it doesn't exist yet
os.makedirs(os.path.join(output_dir, "pre_fit"), exist_ok=True)
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

# %%
print("Plotting signal channels without CRT cuts...")
analysis.signal_channels = ["NPBDT", "ZPBDT_NOCRT"]
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
analysis.signal_channels = ["NPBDT", "ZPBDT"]
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
# %%
plot_chi2_distribution(
    output_dir, "chi2_distribution.json", chi2_at_h0, "rec_nu_energy", "Reco. $\\nu$ Energy"
)
# %%
plot_two_hypo_result(
    output_dir,
    "two_hypo_result_new_constraints.json",
    delta_chi2,
    "rec_nu_energy",
    "Reco. $\\nu$ Energy",
)

# %%
analysis.set_parameters(best_fit_parameters)
print("Plotting signal channels at best fit point...")
print(best_fit_parameters)
extra_text = f"Best fit signal strength: {best_fit_parameters['signal_strength'].m:.3f}"
os.makedirs(os.path.join(output_dir, "post_fit"), exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "post_fit"),
    show_chi_square=True,
    show_data_mc_ratio=True,
    separate_signal=False,
    extra_text=extra_text,
    figsize=(6, 6),
)
# %%
fc_scan_results = from_json(os.path.join(output_dir, "fc_scan_results.json"))
scan_points = fc_scan_results["scan_points"]
scan_chi2 = analysis.scan_chi2(fc_scan_results["parameter_name"], scan_points=scan_points)
delta_chi2_scan = scan_chi2 - best_fit_chi2
fc_scan_results["delta_chi2_scan"] = delta_chi2_scan
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
plot_confidence_intervals(output_dir, fc_scan_results, "rec_nu_energy", ax=ax, xlim=[0, 2])
# %%
if MAKE_DIAGNOSTIC_PLOTS:
    plot_confidence_interval_diagnostic(fc_scan_results)


# Repeat fit using only the NPBDT channel
# %%
analysis.signal_channels = ["NPBDT"]
best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
print(f"Best fit chi2: {best_fit_chi2}")
print(f"Best fit parameters: {best_fit_parameters}")

# %%
data_gen = analysis._get_channel_gen("NPBDT").data_hist_generator
assert data_gen is not None
data_df = data_gen.dataframe
# Type technically requires the binning to be a MultiChannelBinning, which could theoretically
# not be true, but we just assume that it is here.
zpbdt_crt_query = data_gen.binning["ZPBDT"].selection_query  # type: ignore
zpbdt_no_crt_query = data_gen.binning["ZPBDT_NOCRT"].selection_query  # type: ignore
assert isinstance(zpbdt_crt_query, str)
assert isinstance(zpbdt_no_crt_query, str)
data_with_crt = data_df.query(zpbdt_crt_query, engine="python")
data_no_crt = data_df.query(zpbdt_no_crt_query, engine="python")
# %%
# write the dataframes for the EXT with and without the CRT into separate
# csv files
data_with_crt.to_csv(os.path.join(output_dir, "data_zpbdt_with_crt.csv"), index=False)
data_no_crt.to_csv(os.path.join(output_dir, "data_zpbdt_no_crt.csv"), index=False)
# %%
# Print the "evt" and "run" columns for those events that were removed by the 
# crt cut, i.e. those events that are present in the data_no_crt dataframe
# that are no longer present in the data_with_crt dataframe.
# We compare events according to "evt" and "run".
data_with_crt = data_with_crt.set_index(["evt", "run"])
data_no_crt = data_no_crt.set_index(["evt", "run"])
removed_events = data_no_crt.index.difference(data_with_crt.index)
print(removed_events)
# %%
# show NPBDT channel at best fit point
analysis.set_parameters(best_fit_parameters)
print("Plotting signal channels at best fit point...")
print(best_fit_parameters)
extra_text = f"Best fit signal strength: {best_fit_parameters['signal_strength'].m:.3f}"
os.makedirs(os.path.join(output_dir, "post_fit_npbdt"), exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "post_fit_npbdt"),
    show_chi_square=True,
    show_data_mc_ratio=True,
    separate_signal=False,
    extra_text=extra_text,
    figsize=(6, 6),
)
# %%
# Repeat using only the ZPBDT signal channel
analysis.signal_channels = ["ZPBDT"]
best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
print(f"Best fit chi2: {best_fit_chi2}")
print(f"Best fit parameters: {best_fit_parameters}")

# %%
# show ZPBDT channel at best fit point
analysis.set_parameters(best_fit_parameters)
print("Plotting signal channels at best fit point...")
print(best_fit_parameters)
extra_text = f"Best fit signal strength: {best_fit_parameters['signal_strength'].m:.3f}"
os.makedirs(os.path.join(output_dir, "post_fit_zpbdt"), exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "post_fit_zpbdt"),
    show_chi_square=True,
    show_data_mc_ratio=True,
    separate_signal=False,
    extra_text=extra_text,
    figsize=(6, 6),
)

# %%
import numpy as np
analysis.signal_channels = ["ZPBDT"]

scan_points = np.linspace(0, 5, 40)
fit_grid = {
    "signal_strength": np.linspace(0, 10, 50),
}
fc_scan_results = analysis.scan_asimov_fc_sensitivity(
    scan_points=scan_points,
    parameter_name="signal_strength",
    n_trials=1000,
    fit_method="grid_scan",
    fit_grid=fit_grid,
)
to_json(os.path.join(output_dir, "fc_scan_results_zpbdt.json"), fc_scan_results)

# %%
analysis.signal_channels = ["ZPBDT"]
fc_scan_results = from_json(os.path.join(output_dir, "fc_scan_results_zpbdt.json"))
scan_points = fc_scan_results["scan_points"]
scan_chi2 = analysis.scan_chi2(fc_scan_results["parameter_name"], scan_points=scan_points)
delta_chi2_scan = scan_chi2 - best_fit_chi2
fc_scan_results["delta_chi2_scan"] = delta_chi2_scan
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
plot_confidence_intervals(output_dir, fc_scan_results, "rec_nu_energy_zpbdt", ax=ax, xlim=[0, 5])
# %%
