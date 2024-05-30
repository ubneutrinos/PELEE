"""This script reproduces the old analysis as closely as possible and shows the impact of the new constraint procedure in isolation."""

# %%
import sys, os

sys.path.append("../")
import numpy as np

# %%
import logging
import sys
from unblinding_functions import (
    plot_chi2_distribution,
    plot_two_hypo_result,
    plot_confidence_intervals,
    plot_confidence_interval_diagnostic,
)
# Configure logging to display messages of level INFO and above
logging.basicConfig(level=logging.INFO)

# %%
from microfit.analysis import MultibandAnalysis
import logging

config_file = "../config_files/first_round_analysis_runs_1-3.toml"
output_dir = "../old_ana_reprod_run123_output/"

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
analysis.constraint_channels = ["NUMUCRT"]
save_path = os.path.join(output_dir, "sidebands")
os.makedirs(save_path, exist_ok=True)
analysis.plot_sidebands(
    include_multisim_errors=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    show_chi_square=True,
    show_data_mc_ratio=True,
    save_path=save_path,
    figsize=(6, 6),
)

# %%
print("Plotting signal channels...")
analysis.parameters["signal_strength"].value = 1.0
analysis.constraint_channels = ["NUMUCRT"]
save_path = os.path.join(output_dir, "numucrt_constraint")
os.makedirs(save_path, exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    show_chi_square=True,
    show_data_mc_ratio=True,
    save_path=save_path,
    extra_text="Using inclusive $\\nu_\\mu$ constraints",
    figsize=(6, 6),
)

# %%
# Plot signals when using the new constraint channels
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
save_path = os.path.join(output_dir, "new_constraint_channels")
os.makedirs(save_path, exist_ok=True)
analysis.plot_signals(
    include_multisim_errors=True,
    use_sideband=True,
    separate_figures=True,
    add_precomputed_detsys=True,
    show_chi_square=True,
    show_data_mc_ratio=True,
    save_path=save_path,
    extra_text="Using new constraint channels",
    figsize=(6, 6),
)

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
analysis.constraint_channels = ["NUMUCRT"]
best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
print(f"Best fit chi2: {best_fit_chi2}")
print(f"Best fit parameters: {best_fit_parameters}")

h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)
chi2_at_h1 = analysis.get_chi2_at_hypothesis(h1_params)
delta_chi2 = chi2_at_h0 - chi2_at_h1
print(f"Delta chi2: {delta_chi2}")

plot_two_hypo_result(
    output_dir,
    "two_hypo_result_old_constraints.json",
    delta_chi2,
    "rec_nu_energy_old_constraints",
    "Runs 1-3, Old Constraints",
)

# %%
analysis.constraint_channels = ["NUMUCRTNP0PI", "NUMUCRT0P0PI", "TWOSHR"]
best_fit_chi2, best_fit_parameters = analysis.fit_to_data(disp=True)  # type: ignore
print(f"Best fit chi2: {best_fit_chi2}")
print(f"Best fit parameters: {best_fit_parameters}")

h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

chi2_at_h0 = analysis.get_chi2_at_hypothesis(h0_params)
chi2_at_h1 = analysis.get_chi2_at_hypothesis(h1_params)
delta_chi2 = chi2_at_h0 - chi2_at_h1
print(f"Delta chi2: {delta_chi2}")
plot_two_hypo_result(
    output_dir,
    "two_hypo_result_new_constraints.json",
    delta_chi2,
    "rec_nu_energy_new_constraints",
    "Runs 1-3, New Constraints",
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
ax[0].set_title("Signal Channels, Runs 1-3, MC+EXT")
fig.savefig(os.path.join(output_dir, "new_constraint_impact.pdf"))

# %%
print("Computing signal strength sensitivity with old constraints...")
import numpy as np

analysis.constraint_channels = ["NUMUCRT"]

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

ax.set_title("Runs 1-3, Old Signal Model")
fig.savefig(os.path.join(output_dir, "fc_scan_results.pdf"))

# %%
