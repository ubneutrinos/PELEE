# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

import numpy as np
# %%
sys.path.append("../")
import logging
from microfit.analysis import MultibandAnalysis
from microfit.fileio import from_json, to_json
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

config_file = "../config_files/full_ana_with_detvars.toml"
output_dir = "../full_ana_with_run3a_output/"

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
    "NPBDT_SHR_E": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT_SHR_E": "1e0p0$\\pi$ $\\nu_e$ selection",
    "NPBDT_SHR_COSTH": "1eNp0$\\pi$ $\\nu_e$ selection",
    "ZPBDT_SHR_COSTH": "1e0p0$\\pi$ $\\nu_e$ selection",
    "NUMUCRTNP0PI": "1$\\mu$Np0$\\pi$ $\\nu_\\mu$ selection",
    "NUMUCRT0P0PI": "1$\\mu$0p0$\\pi$ $\\nu_\\mu$ selection",
    "TWOSHR": "NC $\\pi^0$ selection"
}

hist_plot_figsize = (5.1, 5)
# %%
h1_params = ParameterSet([Parameter("signal_strength", 1.0)])
analysis.set_parameters(h1_params)
save_path = os.path.join(output_dir, "pre_fit")
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=save_path,
    show_chi_square=True,
    show_data_mc_ratio=True,
    figsize=hist_plot_figsize,
    mb_label_location="right",
    mb_preliminary=True,
    override_channel_titles=override_channel_titles,
)

# %%
analysis.plot_sidebands(
    include_multisim_errors=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "pre_fit"),
    show_chi_square=True,
    show_data_mc_ratio=True,
    figsize=hist_plot_figsize,
    mb_label_location="right",
    mb_preliminary=True,
    override_channel_titles=override_channel_titles,
)

# %%
def run_unblinding(signal_channels, constraint_channels, control_channels, plot_suffix, plot_title, with_fc=True):
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
            n_trials=100000,
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
                n_trials=100000,
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

    to_json(os.path.join(output_dir, f"pelee_data_{plot_suffix}.json"), pelee_data_dict)

    if MAKE_DIAGNOSTIC_PLOTS:
        plot_confidence_interval_diagnostic(fc_scan_results)

    print("Plotting control channel post fit...")
    analysis.signal_channels = control_channels
    original_sideband_channels = analysis.constraint_channels.copy()
    analysis.constraint_channels = signal_channels + original_sideband_channels
    save_path = os.path.join(output_dir, f"post_fit_{plot_suffix}_control_channels")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analysis.set_parameters(best_fit_parameters)
    analysis.plot_sidebands(
        include_multisim_errors=True,
        use_sideband=True,
        separate_figures=True,
        add_precomputed_detsys=True,
        save_path=save_path,
        show_chi_square=True,
        show_data_mc_ratio=True,
        separate_signal=False,
        figsize=hist_plot_figsize,
    )
    extra_text = f"Best fit signal strength: {best_fit_parameters['signal_strength'].m:.3f}"
    extra_text += "\n" + f"Measured {plot_title} included in constraints."
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
    )
    fig, ax = analysis.plot_correlation(
        figsize=(15, 14),
        add_precomputed_detsys=True,
        smooth_detsys_variations=True,
        colorbar_kwargs={"shrink": 0.7},
    )
    fig.savefig(os.path.join(save_path, "correlation_matrix_with_control.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(save_path, "correlation_matrix_with_control.pdf"), bbox_inches="tight")

    analysis.constraint_channels = original_sideband_channels
    return pelee_data_dict

# %%
# Run the analysis for shower cos(theta)
run_unblinding(
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    plot_suffix="shr_costheta",
    plot_title="Shr. $\\cos(\\theta)$",
)
# %%
# Run the analysis for shower energy
run_unblinding(
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    plot_suffix="shr_e",
    plot_title="Shr. Energy",
)

# %%
# run the unblinding using only the ZPBDT channel
run_unblinding(
    signal_channels=["ZPBDT_SHR_E"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["ZPBDT_SHR_COSTHETA"],
    plot_suffix="shr_e_zpbdt",
    plot_title="Shr. Energy, $1e0p0\\pi$",
    with_fc=True,
)

# %%
run_unblinding(
    signal_channels=["ZPBDT_SHR_COSTHETA"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["ZPBDT_SHR_E"],
    plot_suffix="shr_costheta_zpbdt",
    plot_title="Shr. $\\cos(\\theta)$, $1e0p0\\pi$",
    with_fc=True,
)
# %%
# run the unblinding using only the NPBDT channel
run_unblinding(
    signal_channels=["NPBDT_SHR_E"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["NPBDT_SHR_COSTHETA"],
    plot_suffix="shr_e_npbdt",
    plot_title="Shr. Energy, $1eNp0\\pi$",
    with_fc=True,
)

# %%
run_unblinding(
    signal_channels=["NPBDT_SHR_COSTHETA"],
    constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    control_channels=["NPBDT_SHR_E"],
    plot_suffix="shr_costheta_npbdt",
    plot_title="Shr. $\\cos(\\theta)$, $1eNp0\\pi$",
    with_fc=True,
)
# %%
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.parameters["signal_strength"].value = 0.0
os.makedirs(os.path.join(output_dir, "interaction"), exist_ok=True)
analysis.plot_signals(
    category_column="interaction",
    add_precomputed_detsys=True,
    separate_figures=True,
    save_path=os.path.join(output_dir, "interaction"),
    figsize=hist_plot_figsize,
    override_channel_titles=override_channel_titles,
)

analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
analysis.plot_signals(
    category_column="interaction",
    add_precomputed_detsys=True,
    separate_figures=True,
    save_path=os.path.join(output_dir, "interaction"),
    figsize=hist_plot_figsize,
    override_channel_titles=override_channel_titles,
)

# %%
