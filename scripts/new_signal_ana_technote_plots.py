# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys, os

sys.path.append("../")
import numpy as np

# %%
import logging
import sys

# Configure logging to display messages of level INFO and above
logging.basicConfig(level=logging.INFO)

# %%
from microfit.analysis import MultibandAnalysis
import logging

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
print("Plotting sidebands...")
analysis.parameters["signal_strength"].value = 1.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.plot_sidebands(
    include_multisim_errors=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=output_dir,
    show_chi_square=True,
    show_data_mc_ratio=True,
)
# %%
print("Plotting old sidebands...")
analysis.parameters["signal_strength"].value = 1.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.constraint_channels = ["NUMUCRT"]
analysis.plot_sidebands(
    include_multisim_errors=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=output_dir,
    show_chi_square=True,
    show_data_mc_ratio=True,
)
# %%
print("Plotting shower energy signal channels...")
analysis.parameters["signal_strength"].value = 1.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=output_dir,
)

# %%
print("Plotting shower angle signal channels...")
analysis.parameters["signal_strength"].value = 1.0
analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=output_dir,
)

# %%
# analysis.parameters["signal_strength"].value = 1.0
# analysis.signal_channels = ["NPBDT_SHR_THETA", "ZPBDT_SHR_THETA"]
# analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
# analysis.plot_signals(
#     include_multisim_errors=True,
#     use_sideband=True,
#     separate_figures=True,
#     add_precomputed_detsys=True,
#     save_path=output_dir,
# )

# %%
print("Plotting impact of new constraint procedure...")
def plot_new_constraint_impact(signal_channels, plot_name):
    analysis.signal_channels = signal_channels
    analysis.parameters["signal_strength"].value = 0.0

    multi_channel_hist_unconstrained = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
    )

    multi_channel_hist_new_constraints = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=True,
        constraint_channels=["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"],
    )

    multi_channel_hist_old_constraints = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=True,
        constraint_channels=["NUMUCRT"],
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(6, 4.5),
        constrained_layout=True,
        nrows=2,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    multi_channel_hist_unconstrained.draw(ax=ax[0], label="Unconstrained", color="black")
    multi_channel_hist_old_constraints.draw(ax=ax[0], label="Inclusive $\\nu_\\mu$ Constraints")
    multi_channel_hist_new_constraints.draw(ax=ax[0], label="New Constraints w/ CRT")

    rel_error_unconstr = multi_channel_hist_unconstrained / multi_channel_hist_unconstrained.bin_counts
    rel_error_unconstr.draw(ax=ax[1], color="k", show_channel_labels=False)

    rel_error_old = multi_channel_hist_old_constraints / multi_channel_hist_old_constraints.bin_counts
    rel_error_old.draw(ax=ax[1], show_channel_labels=False)

    rel_error_new = multi_channel_hist_new_constraints / multi_channel_hist_new_constraints.bin_counts
    rel_error_new.draw(ax=ax[1], show_channel_labels=False)
    ax[0].set_xlabel("")
    ax[1].set_xlabel("Global bin number")
    ax[1].set_ylabel("Rel. error")
    ax[0].legend()
    ax[0].set_title("Signal Channels, Runs 1-5, MC+EXT")
    fig.savefig(os.path.join(output_dir, plot_name))

plot_new_constraint_impact(["NPBDT_SHR_E", "ZPBDT_SHR_E"], "new_constraint_impact_shr_e.pdf")
plot_new_constraint_impact(
    ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"], "new_constraint_impact_shr_costheta.pdf"
)
# %%
from plot_analysis_histograms_correlations import (
    print_error_budget_tables,
    print_constraint_error_reduction_table,
    plot_correlations,
    plot_constraint_update,
)

# %%
print("Printing error budget tables...")
analysis.parameters["signal_strength"].value = 0.0
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
print_error_budget_tables(analysis, os.path.join(output_dir, "error_budget_tables_null_shr_e.tex"))

analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
print_error_budget_tables(
    analysis, os.path.join(output_dir, "error_budget_tables_null_shr_costheta.tex")
)

# %%
print("Plotting constraint update in signal channels...")
analysis.parameters["signal_strength"].value = 0.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
plot_constraint_update(analysis, output_dir)
analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
plot_constraint_update(analysis, output_dir)

# %%
print("Printing constraint update tables...")
analysis.parameters["signal_strength"].value = 0.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
print_constraint_error_reduction_table(
    analysis, os.path.join(output_dir, "constraint_error_reduction_table_shr_e.tex")
)
analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
print_constraint_error_reduction_table(
    analysis, os.path.join(output_dir, "constraint_error_reduction_table_shr_costheta.tex")
)

# %%
analysis.parameters["signal_strength"].value = 0.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
fig, ax = analysis.plot_correlation(
    figsize=(5, 4),
    ms_columns=[],
    add_precomputed_detsys=True,
    channels=analysis.signal_channels,
    smooth_detsys_variations=False,
    as_correlation=True,
)
ax.set_title("Correlations of unsmoothed detector systematics")
fig.savefig(os.path.join(output_dir, "correlation_plot_signal_channels_detvar.pdf"))

fig, ax = analysis.plot_correlation(
    figsize=(5, 4),
    ms_columns=[],
    add_precomputed_detsys=True,
    channels=analysis.signal_channels,
    smooth_detsys_variations=True,
    as_correlation=True,
)
ax.set_title("Correlations of smoothed detector systematics")
fig.savefig(os.path.join(output_dir, "correlation_plot_signal_channels_smooth_detvar.pdf"))

# %%
analysis.parameters["signal_strength"].value = 0.0
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
fig, ax = analysis.plot_correlation(
    figsize=(10, 9),
    add_precomputed_detsys=True,
    smooth_detsys_variations=True,
    colorbar_kwargs={"shrink": 0.7},
)
fig.savefig(os.path.join(output_dir, "correlation_plot_smoothed_detvar_total_shr_e.pdf"))

analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
fig, ax = analysis.plot_correlation(
    figsize=(10, 9),
    add_precomputed_detsys=True,
    smooth_detsys_variations=True,
    colorbar_kwargs={"shrink": 0.7},
)
fig.savefig(os.path.join(output_dir, "correlation_plot_smoothed_detvar_total_shr_costheta.pdf"))

# %%
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

print("Plotting expected chi-square distribution at null...")
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]

chi2_dict = analysis.get_chi_square_distribution(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]), n_trials=100000, run_fit=False
)
to_json(os.path.join(output_dir, "chi2_distribution_shr_e.json"), chi2_dict)
# %%
from microfit.fileio import from_json
import matplotlib.pyplot as plt

chi2_dict = from_json(os.path.join(output_dir, "chi2_distribution_shr_e.json"))
chi2_h0 = chi2_dict["chi2_h0"]
# Exclude large outliers in the distribution to make the histogram more readable
chi2_h0 = chi2_h0[chi2_h0 < np.percentile(chi2_h0, 99.9)]

plt.hist(chi2_h0, bins=100, histtype="step")
plt.xlabel(r"$\chi^2$")
plt.ylabel("Samples")
plt.title("Expected $\\chi^2$ distribution at null, Reco. Shower Energy")
plt.savefig(os.path.join(output_dir, "chi2_distribution_shr_e.pdf"))
# %%
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

print("Plotting expected chi-square distribution at null...")
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]

chi2_dict = analysis.get_chi_square_distribution(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]), n_trials=100000, run_fit=False
)
to_json(os.path.join(output_dir, "chi2_distribution_shr_costheta.json"), chi2_dict)
# %%
from microfit.fileio import from_json
import matplotlib.pyplot as plt

chi2_dict = from_json(os.path.join(output_dir, "chi2_distribution_shr_costheta.json"))
chi2_h0 = chi2_dict["chi2_h0"]
# Exclude large outliers in the distribution to make the histogram more readable
chi2_h0 = chi2_h0[chi2_h0 < np.percentile(chi2_h0, 99.9)]

plt.hist(chi2_h0, bins=100, histtype="step")
plt.xlabel(r"$\chi^2$")
plt.ylabel("Samples")
plt.title("Expected $\\chi^2$ distribution at null, Reco. Shower $\\cos(\\theta)$")
plt.savefig(os.path.join(output_dir, "chi2_distribution_shr_costheta.pdf"))
# %%
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

print("Computing sensitivity for shower energy...")
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]

two_hypo_result_new_constraints = analysis.two_hypothesis_test(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]),
    h1_params=ParameterSet([Parameter("signal_strength", 1.0)]),
    sensitivity_only=True,
    n_trials=100000,
)

to_json(os.path.join(output_dir, "two_hypo_result_shr_e.json"), two_hypo_result_new_constraints)

# %%
from microfit.fileio import from_json
import matplotlib.pyplot as plt


def plot_two_hypo_result(results_path, plot_path, title):
    # read results from file
    two_hypo_results = from_json(results_path)
    minimum = np.min(two_hypo_results["samples_h0"])
    minimum = min(minimum, np.min(two_hypo_results["samples_h1"]))
    maximum = np.max(two_hypo_results["samples_h0"])
    maximum = max(maximum, np.max(two_hypo_results["samples_h1"]))

    n_trials = len(two_hypo_results["samples_h0"])
    bin_edges = np.linspace(minimum, maximum, int(np.sqrt(n_trials)))

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.hist(
        two_hypo_results["samples_h0"], bins=bin_edges, histtype="step", density=False, label="H0"
    )
    ax.hist(
        two_hypo_results["samples_h1"], bins=bin_edges, histtype="step", density=False, label="H1"
    )
    ax.axvline(
        x=two_hypo_results["ts_median_h1"],
        color="k",
        linestyle="--",
        label=f"Median H1\np-val: {two_hypo_results['median_pval']*100:0.3f}%",
    )
    ax.legend()
    ax.set_xlabel(r"$\Delta \chi^2$")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    fig.savefig(plot_path)


plot_two_hypo_result(
    os.path.join(output_dir, "two_hypo_result_shr_e.json"),
    os.path.join(output_dir, "two_hypo_result_shr_e.pdf"),
    "Sensitivity, Runs 1-5, Reco. Shower Energy",
)
# %%
print("Computing sensitivity for shower angle...")
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]

two_hypo_result_new_constraints_shr_theta = analysis.two_hypothesis_test(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]),
    h1_params=ParameterSet([Parameter("signal_strength", 1.0)]),
    sensitivity_only=True,
    n_trials=100000,
)

to_json(
    os.path.join(output_dir, "two_hypo_result_shr_costheta.json"),
    two_hypo_result_new_constraints_shr_theta,
)

# %%
plot_two_hypo_result(
    os.path.join(output_dir, "two_hypo_result_shr_costheta.json"),
    os.path.join(output_dir, "two_hypo_result_shr_costheta.pdf"),
    "Sensitivity, Runs 1-5, Reco. $\\cos(\\theta)$",
)

# %%
print("Computing signal strength sensitivity for shower energy...")
import numpy as np

analysis.signal_channels = ["NPBDT_SHR_E", "ZPBDT_SHR_E"]

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
# %%
to_json(os.path.join(output_dir, "fc_scan_results_shr_e.json"), fc_scan_results)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

MultibandAnalysis.plot_fc_scan_results(
    from_json(os.path.join(output_dir, "fc_scan_results_shr_e.json")),
    parameter_tex="signal strength",
    ax=ax,
)
ax.set_xlim((0, 5))
ax.set_ylim(bottom=0, top=5)

ax.set_title("Runs 1-5, Reco. Shower Energy")
fig.savefig(os.path.join(output_dir, "fc_scan_results_shr_e.pdf"))
# %%
print("Computing signal strength sensitivity for shower angle...")
import numpy as np

analysis.signal_channels = ["NPBDT_SHR_COSTHETA", "ZPBDT_SHR_COSTHETA"]

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

to_json(os.path.join(output_dir, "fc_scan_results_shr_costheta.json"), fc_scan_results)

# %%
fig, ax = plt.subplots(figsize=(6, 5))

MultibandAnalysis.plot_fc_scan_results(
    from_json(os.path.join(output_dir, "fc_scan_results_shr_costheta.json")),
    parameter_tex="signal strength",
    ax=ax,
)

ax.set_xlim((0, 5))
ax.set_ylim(bottom=0, top=5)

ax.set_title("Runs 1-5, Reco. $\\cos(\\theta)$")
fig.savefig(os.path.join(output_dir, "fc_scan_results_shr_costheta.pdf"))
# %%
