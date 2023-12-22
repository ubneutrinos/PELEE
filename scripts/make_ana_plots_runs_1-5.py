# %%
import sys, os
sys.path.append(".")

# %%
from microfit.analysis import MultibandAnalysis
import toml

analysis = MultibandAnalysis(toml.load("config_files/first_round_analysis_runs_1-5.toml"))

# %%
import matplotlib.pyplot as plt
ax = analysis.plot_sideband()
ax.set_ylim((0, 10000))
ax.set_xlim((0.1, 2))
ax.set_title(r"$\nu_\mu$ sideband, Runs 1-5")
plt.savefig("sideband_numu_runs_1-5.pdf")

# %%
fig, axes = analysis.plot_signals()
for ax in axes.flatten():
    ax.set_xlim((0.1, 2))
    ax.set_ylim((0, 30))
fig.savefig("signals_runs_1-5_constrained.pdf")

# %%
# Combine genie multisim with unisim, since the unisim knobs come from GENIE as well
fig, ax = analysis.plot_correlation(ms_column="weightsGenie", with_unisim=True)
fig.savefig("multiband_correlation_runs_1-5_weightsGenie.pdf")

fig, ax = analysis.plot_correlation(ms_column="weightsFlux")
fig.savefig("multiband_correlation_runs_1-5_weightsFlux.pdf")

fig, ax = analysis.plot_correlation(ms_column="weightsReint")
fig.savefig("multiband_correlation_runs_1-5_weightsReint.pdf")

fig, ax = analysis.plot_correlation()
fig.savefig("multiband_correlation_runs_1-5_total.pdf")

# %%



