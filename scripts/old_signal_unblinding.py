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
    plot_two_hypo_result,
    plot_confidence_intervals,
    plot_confidence_interval_diagnostic,
    get_signal_count_channels,
    get_signal_n_bins,
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
    two_hypo_results_file = f"two_hypo_result_{plot_suffix}.json"

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

    best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
    print(f"Best fit chi2: {best_fit_chi2}")
    print(f"Best fit parameters: {best_fit_parameters}")

    chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)
    chi2_at_h1 = analysis.get_chi2_at_hypothesis(h1_params)
    delta_chi2 = chi2_at_h0 - chi2_at_h1
    print(f"Delta chi2: {delta_chi2}")

    if not os.path.exists(os.path.join(output_dir, chi2_results_file)):
        chi2_dict = analysis.get_chi_square_distribution(
            h0_params=h0_params, n_trials=100000, run_fit=False
        )
        to_json(os.path.join(output_dir, chi2_results_file), chi2_dict)
    else:
        chi2_dict = from_json(os.path.join(output_dir, chi2_results_file))
    plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_h0, plot_suffix, plot_title)
    if not os.path.exists(os.path.join(output_dir, two_hypo_results_file)):
        two_hypo_dict = analysis.two_hypothesis_test(
            h0_params=h0_params,
            h1_params=h1_params,
            sensitivity_only=False,
            # increased trials to get a better estimate of the p-value
            n_trials=1000000,
        )
        to_json(os.path.join(output_dir, two_hypo_results_file), two_hypo_dict)
    else:
        two_hypo_dict = from_json(os.path.join(output_dir, two_hypo_results_file))
        if "pval_h0" not in two_hypo_dict or two_hypo_dict["pval_h0"] is None:
            # When we already have the distributions, we can just do the unblinding.
            two_hypo_dict = analysis.two_hypothesis_test(
                h0_params=h0_params,
                h1_params=h1_params,
                sensitivity_only=False,
                # increased trials to get a better estimate of the p-value
                n_trials=1000000,
                sens_only_dict=two_hypo_dict,
            )
            to_json(os.path.join(output_dir, two_hypo_results_file), two_hypo_dict)
    plot_two_hypo_result(
        output_dir,
        two_hypo_results_file,
        delta_chi2,
        plot_suffix + "sensitivity_only",
        plot_title,
        sensitivity_only=True,
    )
    plot_two_hypo_result(
        output_dir,
        two_hypo_results_file,
        delta_chi2,
        plot_suffix,
        plot_title,
        sensitivity_only=False,
    )

    analysis.set_parameters(best_fit_parameters)
    print("Plotting signal channels at best fit point...")
    save_path = os.path.join(output_dir, f"post_fit_{plot_suffix}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    extra_text = f"Best fit signal strength: {best_fit_parameters['signal_strength'].m:.3f}"
    analysis.plot_signals(
        include_multisim_errors=True,
        use_sideband=True,
        separate_figures=True,
        add_precomputed_detsys=True,
        save_path=save_path,
        show_chi_square=True,
        show_data_mc_ratio=True,
        separate_signal=False,
        extra_text=extra_text,
        figsize=hist_plot_figsize,
        override_channel_titles=override_channel_titles,
    )

    signal_counts_dict = get_signal_count_channels(analysis, signal_channels)

    if not with_fc:
        return
    if not os.path.exists(os.path.join(output_dir, f"fc_scan_results_{plot_suffix}.json")):
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
        to_json(os.path.join(output_dir, f"fc_scan_results_{plot_suffix}.json"), fc_scan_results)
    fc_scan_results = from_json(os.path.join(output_dir, f"fc_scan_results_{plot_suffix}.json"))
    scan_points = fc_scan_results["scan_points"]
    scan_chi2 = analysis.scan_chi2(fc_scan_results["parameter_name"], scan_points=scan_points)
    delta_chi2_scan = scan_chi2 - best_fit_chi2
    fc_scan_results["delta_chi2_scan"] = delta_chi2_scan

    conf_interval_dict = plot_confidence_intervals(output_dir, fc_scan_results, plot_suffix)

    pelee_data_dict = signal_counts_dict.copy()
    pelee_data_dict["name"] = plot_suffix
    pelee_data_dict["chi2_x0"] = chi2_at_h0
    pelee_data_dict["chi2_x1"] = chi2_at_h1
    pelee_data_dict["chi2_df"] = get_signal_n_bins(analysis)
    # from the simple chi-square distribution
    chi2_trials = chi2_dict["chi2_h0"]
    pval_h0 = (chi2_trials > chi2_at_h0).sum() / len(chi2_trials)
    pelee_data_dict["p_x0"] = pval_h0
    pelee_data_dict["p_x1"] = None
    # From the two-hypothesis test
    pelee_data_dict["p_x0_x1"] = 1.0 - two_hypo_dict["pval_h0"]  # Assuming H0
    pelee_data_dict["p_x1_x0"] = 1.0 - two_hypo_dict["pval_h1"]  # Assuming H1
    # Best fit point
    pelee_data_dict["x_fit"] = best_fit_parameters["signal_strength"].m
    # Confidence intervals
    pelee_data_dict["x_fc_1sigma"] = [(conf_interval_dict["p_1sig_lower"],conf_interval_dict["p_1sig_upper"])]
    pelee_data_dict["x_fc_90pct"] = [(conf_interval_dict["p_90_lower"], conf_interval_dict["p_90_upper"])]
    pelee_data_dict["x_fc_2sigma"] = [(conf_interval_dict["p_2sig_lower"], conf_interval_dict["p_2sig_upper"])]
    # Asimov sensitivities
    pelee_data_dict["x_fc_1sigma_exp"] = [(conf_interval_dict["p_1sig_lower_sens"], conf_interval_dict["p_1sig_upper_sens"])]
    pelee_data_dict["x_fc_90pct_exp"] = [(conf_interval_dict["p_90_lower_sens"], conf_interval_dict["p_90_upper_sens"])]
    pelee_data_dict["x_fc_2sigma_exp"] = [(conf_interval_dict["p_2sig_lower_sens"], conf_interval_dict["p_2sig_upper_sens"])]
    if MAKE_DIAGNOSTIC_PLOTS:
        plot_confidence_interval_diagnostic(fc_scan_results)
    
    to_json(os.path.join(output_dir, f"pelee_data_{plot_suffix}.json"), pelee_data_dict)
    return pelee_data_dict

# %%
def get_crt_removed_events():
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
    # write the dataframes for the EXT with and without the CRT into separate
    # csv files
    data_with_crt.to_csv(os.path.join(output_dir, "data_zpbdt_with_crt.csv"), index=False)
    data_no_crt.to_csv(os.path.join(output_dir, "data_zpbdt_no_crt.csv"), index=False)
    # Print the "evt" and "run" columns for those events that were removed by the 
    # crt cut, i.e. those events that are present in the data_no_crt dataframe
    # that are no longer present in the data_with_crt dataframe.
    # We compare events according to "evt" and "run".
    data_with_crt = data_with_crt.set_index(["evt", "run"])
    data_no_crt = data_no_crt.set_index(["evt", "run"])
    removed_events = data_no_crt.index.difference(data_with_crt.index)
    print(removed_events)
# %%
run_unblinding(
    signal_channels=["NPBDT", "ZPBDT"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    plot_suffix="npbdt_zpbdt",
    plot_title="Combined Channels",
)
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
    figsize=hist_plot_figsize,
    override_channel_titles=override_channel_titles,
)
analysis.signal_channels = ["NPBDT", "ZPBDT"]


# %%
# Calculate the empirical p-value for each sideband channel. We can do this 
# by setting the channel as the only signal channel without constraint channels,
# and then letting the analysis class produce the chi-square distribution.

for channel in ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]:
    analysis.signal_channels = [channel]
    analysis.constraint_channels = []
    chi2_results_file = f"chi2_distribution_{channel}.json"
    save_path = os.path.join(output_dir, "pre_fit")
    h0_params = ParameterSet([Parameter("signal_strength", 0.0)])

    if not os.path.exists(os.path.join(output_dir, chi2_results_file)):
        chi2_dict = analysis.get_chi_square_distribution(
            h0_params=h0_params, n_trials=100000, run_fit=False
        )
        to_json(os.path.join(output_dir, chi2_results_file), chi2_dict)
    else:                                                                                                   
        chi2_dict = from_json(
            os.path.join(output_dir, chi2_results_file)
        )
    chi2_trials = chi2_dict["chi2_h0"]
    chi2_at_data = analysis.get_chi2_at_hypothesis(h0_params)
    pval_data = (chi2_trials > chi2_at_data).sum() / len(chi2_trials)
    print(f"Empirical p-value for channel {channel}: {pval_data}")
    plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_data, channel, channel)
# %%
def negative_signal_strength_fit(signal_channels, constraint_channels, plot_suffix):
    analysis.parameters["signal_strength"].bounds = (-10, 10)
    h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
    analysis.signal_channels = signal_channels
    analysis.constraint_channels = constraint_channels
    analysis.set_parameters(h0_params)
    analysis.fit_to_data()
    save_path = os.path.join(output_dir, f"post_fit_{plot_suffix}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(analysis.parameters)
    analysis.plot_signals(
        include_multisim_errors=True,
        use_sideband=True,
        separate_figures=True,
        add_precomputed_detsys=True,
        save_path=save_path,
        show_chi_square=False,
        show_data_mc_ratio=True,
        figsize=hist_plot_figsize,
        override_channel_titles=override_channel_titles,
        show_signal_in_ratio=True
    )

# %%
negative_signal_strength_fit(
    signal_channels=["NPBDT", "ZPBDT"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    plot_suffix="npbdt_zpbdt_negative",
)

# %%
negative_signal_strength_fit(
    signal_channels=["NPBDT"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    plot_suffix="npbdt_negative",
)
# %%
from scipy.stats import chi2
# compute total goodness-of-fit for the combined signal and sideband channels
analysis.signal_channels = [
    "NPBDT",
    "ZPBDT",
    "NUMUCRTNP0PI",
    "NUMUCRT0P0PI",
    # "TWOSHR",
]
analysis.constraint_channels = []
h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
total_chi2 = analysis.get_chi2_at_hypothesis(h0_params)
print("Total chi-square: ", total_chi2)
joint_mc_hist = analysis.get_mc_hist()
joint_ext_hist = analysis.get_data_hist(type="ext")
joint_total_prediction = joint_mc_hist + joint_ext_hist
joint_data_hist = analysis.get_data_hist()
assert joint_data_hist is not None

n_bins = joint_data_hist.n_bins
print(f"Number of bins: {n_bins}")
print(f"Chi-square p-value: {chi2.sf(total_chi2, n_bins) * 100}")

fig, ax = plt.subplots()
joint_total_prediction.draw(ax=ax, label="Total prediction", show_channel_labels=False)
joint_data_hist.draw(ax=ax, label="Data", as_errorbars=True, color="k", show_channel_labels=False)
ax.legend()
ax.set_ylim(bottom=0.1)
ax.set_yscale("log")
plt.show()
# %%
