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
print("Plotting sidebands...")
analysis.parameters["signal_strength"].value = 0.0
analysis.plot_sidebands(
    include_multisim_errors=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    show_chi_square=True,
    show_data_mc_ratio=True,
    save_path=output_dir,
    figsize=(6, 6),
)

# %%
print("Plotting signal channels...")
analysis.parameters["signal_strength"].value = 1.0
os.makedirs(os.path.join(output_dir, "mc_only"), exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    save_path=os.path.join(output_dir, "mc_only"),
    show_data=False,
    figsize=(6, 4.5),
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

print_error_budget_tables(analysis, os.path.join(output_dir, "error_budget_tables_null.tex"))


# %%
print("Plotting constraint update in signal channels...")
analysis.parameters["signal_strength"].value = 0.0
plot_constraint_update(analysis, output_dir)


# %%
print("Printing constraint update tables...")
analysis.parameters["signal_strength"].value = 0.0
print_constraint_error_reduction_table(
    analysis, os.path.join(output_dir, "constraint_error_reduction_table.tex")
)

# %%
analysis.parameters["signal_strength"].value = 0.0
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
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
fig, ax = analysis.plot_correlation(
    figsize=(10, 9),
    add_precomputed_detsys=True,
    smooth_detsys_variations=True,
)
fig.savefig(os.path.join(output_dir, "correlation_plot_smoothed_detvar_total.pdf"))

# %%
print("Plotting total correlation for old sidebands...")
analysis.constraint_channels = ["NUMUCRT"]
analysis.parameters["signal_strength"].value = 0.0
fig, ax = analysis.plot_correlation(
    figsize=(10, 9),
    add_precomputed_detsys=True,
    smooth_detsys_variations=True,
    colorbar_kwargs={"shrink": 0.7},
)
fig.savefig(os.path.join(output_dir, "correlation_plot_inclusive_numu_sideband_total.pdf"))

# %%
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

print("Plotting expected chi-square distribution at null...")
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]

chi2_dict = analysis.get_chi_square_distribution(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]), n_trials=100000, run_fit=False
)
to_json(os.path.join(output_dir, "chi2_distribution.json"), chi2_dict)
# %%
from microfit.fileio import from_json
import matplotlib.pyplot as plt

chi2_dict = from_json(os.path.join(output_dir, "chi2_distribution.json"))
chi2_h0 = chi2_dict["chi2_h0"]
# Exclude large outliers in the distribution to make the histogram more readable
chi2_h0 = chi2_h0[chi2_h0 < np.percentile(chi2_h0, 99.9)]

fig, ax = plt.subplots()
ax.hist(chi2_h0, bins=100, histtype="step")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel("Samples")
ax.set_title("Expected $\\chi^2$ distribution at null, Reco. $\\nu$ Energy")
fig.savefig(os.path.join(output_dir, "chi2_distribution.pdf"))

# %%
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

print("Computing sensitivity for new constraints...")
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]

two_hypo_result_new_constraints = analysis.two_hypothesis_test(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]),
    h1_params=ParameterSet([Parameter("signal_strength", 1.0)]),
    sensitivity_only=True,
    n_trials=100000,
)

to_json(
    os.path.join(output_dir, "two_hypo_result_new_constraints.json"),
    two_hypo_result_new_constraints,
)

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
    os.path.join(output_dir, "two_hypo_result_new_constraints.json"),
    os.path.join(output_dir, "two_hypo_result_new_constraints.pdf"),
    "Sensitivity, Runs 1-5, New Constraints",
)
# %%
print("Computing sensitivity for old constraints...")
from microfit.parameters import ParameterSet, Parameter
from microfit.fileio import to_json

analysis.constraint_channels = ["NUMUCRT"]

two_hypo_result_old_constraints = analysis.two_hypothesis_test(
    h0_params=ParameterSet([Parameter("signal_strength", 0.0)]),
    h1_params=ParameterSet([Parameter("signal_strength", 1.0)]),
    sensitivity_only=True,
    n_trials=100000,
)

to_json(
    os.path.join(output_dir, "two_hypo_result_old_constraints.json"),
    two_hypo_result_old_constraints,
)

# %%
plot_two_hypo_result(
    os.path.join(output_dir, "two_hypo_result_old_constraints.json"),
    os.path.join(output_dir, "two_hypo_result_old_constraints.pdf"),
    "Sensitivity, Runs 1-5, Old Constraints",
)
# %%
print("Plotting impact of new constraint procedure...")
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
fig.savefig(os.path.join(output_dir, "new_constraint_impact.pdf"))

# %%
print("Computing signal strength sensitivity...")
import numpy as np

analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]

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
to_json(os.path.join(output_dir, "fc_scan_results.json"), fc_scan_results)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

MultibandAnalysis.plot_fc_scan_results(
    from_json(os.path.join(output_dir, "fc_scan_results.json")),
    parameter_tex="signal strength",
    ax=ax,
)
ax.set_xlim((0, 5))
ax.set_ylim(bottom=0, top=5)

ax.set_title("Runs 1-5, Old Signal Model")
fig.savefig(os.path.join(output_dir, "fc_scan_results.pdf"))

# %%
