# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

sys.path.append("../")
import logging
from microfit.analysis import MultibandAnalysis
from microfit.fileio import from_json
from microfit.parameters import Parameter, ParameterSet
from unblinding_functions import (
    plot_chi2_distribution,
    plot_two_hypo_result,
    plot_confidence_intervals,
    plot_confidence_interval_diagnostic,
)

logging.basicConfig(level=logging.INFO)

MAKE_DIAGNOSTIC_PLOTS = True

config_file = "../config_files/new_signal_ana_opendata_bnb.toml"
output_dir = "../new_signal_ana_opendata_bnb_output/"

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)


# %%
def run_unblinding(signal_channels, control_channels, plot_suffix, plot_title):
    chi2_results_file = f"chi2_distribution_{plot_suffix}.json"
    two_hypo_results_file = f"two_hypo_result_{plot_suffix}.json"

    analysis.signal_channels = signal_channels
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

    best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
    print(f"Best fit chi2: {best_fit_chi2}")
    print(f"Best fit parameters: {best_fit_parameters}")

    h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
    h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

    chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)
    chi2_at_h1 = analysis.get_chi2_at_hypothesis(h1_params)
    delta_chi2 = chi2_at_h0 - chi2_at_h1
    print(f"Delta chi2: {delta_chi2}")

    plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_h0, plot_suffix, plot_title)
    plot_two_hypo_result(output_dir, two_hypo_results_file, delta_chi2, plot_suffix, plot_title)

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

    fc_scan_results = from_json(os.path.join(output_dir, f"fc_scan_results_{plot_suffix}.json"))
    scan_points = fc_scan_results["scan_points"]
    scan_chi2 = analysis.scan_chi2(fc_scan_results["parameter_name"], scan_points=scan_points)
    delta_chi2_scan = scan_chi2 - best_fit_chi2
    fc_scan_results["delta_chi2_scan"] = delta_chi2_scan

    plot_confidence_intervals(output_dir, fc_scan_results, plot_suffix)
    if MAKE_DIAGNOSTIC_PLOTS:
        plot_confidence_interval_diagnostic(fc_scan_results)

    print("Plotting control channel post fit...")
    analysis.signal_channels = control_channels
    original_sideband_channels = analysis.constraint_channels.copy()
    analysis.constraint_channels = original_sideband_channels + signal_channels
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
    )

    analysis.constraint_channels = original_sideband_channels


# %%
# Run the analysis for shower cos(theta)
run_unblinding(
    signal_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    control_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    plot_suffix="shr_costheta",
    plot_title="Shr. $\\cos(\\theta)$",
)
# %%
# Run the analysis for shower energy
run_unblinding(
    signal_channels=["NPBDT_SHR_E", "ZPBDT_SHR_E"],
    control_channels=["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"],
    plot_suffix="shr_e",
    plot_title="Shr. Energy",
)

# %%
