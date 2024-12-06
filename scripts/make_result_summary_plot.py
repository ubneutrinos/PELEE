"""This script generates the summary plot for the results of the analysis.
This requires the unblinding scripts to have been run first. They store
the fit results to JSON as 'confidence_intervals_{plot_suffix}.json' in the
output directory. This script reads these files and generates a summary 
as a bar plot with one bar for every signal channel.
"""

# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os
import numpy as np
# %%
sys.path.append("../")
from microfit.fileio import from_json, to_json

# %%
new_signal_output_dir = "../full_ana_with_run3a_output/"
old_signal_output_dir = "../old_model_ana_output_with_3a"

# %%
import matplotlib.pyplot as plt
# %%
variables = [
    "rec_nu_energy",
    "shr_e",
    "shr_costheta",
]
variable_labels = [
    "Neutrino Energy,\nneutrino energy\nunfolded signal model",
    "Shower Energy,\nelectron kinematics\nunfolded\nsignal model",
    "Shower $\\cos(\\theta)$,\nelectron kinematics\nunfolded\nsignal model", 
]
variable_labels_short = [
    "Neutrino energy",
    "Shower energy",
    "Shower $\\cos(\\theta)$",
]
channel_suf = ["_zpbdt", "_npbdt", ""]
channel_labels = ["$1e0p0\\pi$", "$1eNp0\\pi$", "combined"]

# %%
def get_data(variable, channel_suffix):
    if variable == "rec_nu_energy":
        output_dir = old_signal_output_dir
        if channel_suffix == "":
            channel_suffix = "_npbdt_zpbdt"
        return from_json(os.path.join(output_dir, f"confidence_intervals{channel_suffix}.json"))
    else:
        output_dir = new_signal_output_dir
    return from_json(os.path.join(output_dir, f"confidence_intervals_{variable}{channel_suffix}.json"))

def get_bfp_from_data(data):
    scan_points = data["scan_points"]
    p_values = data["p_values"]
    return scan_points[np.argmax(p_values)]

def get_limits_from_data(data, conf="68"):
    lower = f"p_{conf}_lower"
    upper = f"p_{conf}_upper"
    return data[lower], data[upper]
# %%
fig, ax = plt.subplots(constrained_layout=True, figsize=(5.5, 4.5))

# As the x-value, make groups of three, one group per variable and one
# group element per channel.
group_width = 0.2
x = []
central_values = []
lower_68 = []
upper_68 = []
lower_90 = []
upper_90 = []
lower_99 = []
upper_99 = []
labels = []
for i, var in enumerate(variables):
    for j, suf in enumerate(channel_suf):
        x.append(i + j * group_width)
        labels.append(f"{channel_labels[j]}")
        data = get_data(var, suf)
        central_values.append(get_bfp_from_data(data))
        lower_68_, upper_68_ = get_limits_from_data(data, "68")
        lower_90_, upper_90_ = get_limits_from_data(data, "90")
        lower_99_, upper_99_ = get_limits_from_data(data, "99")
        lower_68.append(lower_68_)
        upper_68.append(upper_68_)
        lower_90.append(lower_90_)
        upper_90.append(upper_90_)
        lower_99.append(lower_99_)
        upper_99.append(upper_99_)

central_values = np.array(central_values)
lower_68 = np.array(lower_68)
upper_68 = np.array(upper_68)
lower_90 = np.array(lower_90)
upper_90 = np.array(upper_90)
lower_99 = np.array(lower_99)
upper_99 = np.array(upper_99)

# Calculate where the middle between the groups is, i.e. the midpoint between the 
# last element of one group and the first element of the next group.
# Then, draw a shaded region to highlight the different variables.
n_per_group = len(channel_labels)
n_groups = len(variables)
between_group_spacing = 1.0 - group_width * n_groups
mid_before = -between_group_spacing / 2
plot_lower_xlim = mid_before
plot_upper_xlim = x[-1] + between_group_spacing / 2
shade = True
midpoints = [mid_before]
for i in range(n_groups - 1):
    end_first_group = x[i * n_per_group + n_per_group - 1]
    start_next_group = x[(i + 1) * n_per_group]
    mid = (end_first_group + start_next_group) / 2
    midpoints.append(mid)
    ax.axvline(x=mid, color="gray", linestyle="-", lw=0.5)
    if shade:
        ax.axvspan(mid_before, mid, alpha=0.1, color="gray")
    mid_before = mid
    shade = not shade
# if we get here and shade is True, then we need to shade the last column
# in post
if shade:
    ax.axvspan(mid_before, plot_upper_xlim, alpha=0.1, color="gray")
midpoints.append(plot_upper_xlim)
midpoints = np.array(midpoints)
group_centers = midpoints[:-1] + (midpoints[1:] - midpoints[:-1]) / 2

ax.axhline(y=0, color="k", linestyle="--", lw=1)
ax.axhline(y=1, color="k", linestyle="--", lw=1)

cmap = plt.get_cmap("winter")  # type: ignore

ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_99, upper_99 - central_values],
    fmt="none",
    label="99% C.L.",
    lw=2,
    capsize=3,
    color=cmap(0.8),
)

ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_90, upper_90 - central_values],
    fmt="none",
    label="90% C.L.",
    lw=3,
    capsize=5,
    color=cmap(0.5),
)


ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_68, upper_68 - central_values],
    fmt="none",
    label="68% C.L.",
    lw=4,
    capsize=8,
    color=cmap(0.1),
)

ax.plot(x, central_values, "D", label="Best Fit Point", color="k")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
# ax.grid(axis="y")
ax.set_ylabel("Signal Strength")
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, columnspacing=0.9)
ax.set_xlim(left=plot_lower_xlim, right=plot_upper_xlim)

# in the center of each group, write the variable label at some y-position
max_y = 3.8  # this is the y-value where you're adding text
for i, mid in enumerate(group_centers):
    ax.text(mid, max_y, variable_labels_short[i], ha="center", va="top", color="k", fontsize=9)

# Add the MicroBooNE and POT label
ax.text(group_centers[1], max_y - 0.25, "MicroBooNE, $1.1\\times10^{21}$ POT", ha="center", va="top", color="k", fontsize=8)

# get current y-limits
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, max_y + 0.1)  # add some padding

fig.savefig("summary_plot.pdf")
fig.savefig("summary_plot.png", dpi=200)
# %%
# Make another plot where we only show the results from the combined channels. Here, we want the x-labels to be the variable labels.

fig, ax = plt.subplots(figsize=(4, 3.5))
x = []
central_values = []
lower_68 = []
upper_68 = []
lower_90 = []
upper_90 = []
lower_99 = []
upper_99 = []
labels = []

for i, var in enumerate(variables):
    x.append(i)
    labels.append(variable_labels_short[i])
    data = get_data(var, "")
    central_values.append(get_bfp_from_data(data))
    lower_68_, upper_68_ = get_limits_from_data(data, "68")
    lower_90_, upper_90_ = get_limits_from_data(data, "90")
    lower_99_, upper_99_ = get_limits_from_data(data, "99")
    lower_68.append(lower_68_)
    upper_68.append(upper_68_)
    lower_90.append(lower_90_)
    upper_90.append(upper_90_)
    lower_99.append(lower_99_)
    upper_99.append(upper_99_)

central_values = np.array(central_values)
lower_68 = np.array(lower_68)
upper_68 = np.array(upper_68)
lower_90 = np.array(lower_90)
upper_90 = np.array(upper_90)
lower_99 = np.array(lower_99)
upper_99 = np.array(upper_99)

ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_99, upper_99 - central_values],
    fmt="none",
    label="99% C.L.",
    lw=2,
    capsize=3,
    color=cmap(0.8),
)
ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_90, upper_90 - central_values],
    fmt="none",
    label="90% C.L.",
    lw=3,
    capsize=5,
    color=cmap(0.5),
)
ax.errorbar(
    x,
    central_values,
    yerr=[central_values - lower_68, upper_68 - central_values],
    fmt="none",
    label="68% C.L.",
    lw=4,
    capsize=8,
    color=cmap(0.1),
)
ax.plot(x, central_values, "D", label="Best Fit Point", color="k")

ax.axhline(y=0, color="k", linestyle="--", lw=1)
ax.axhline(y=1, color="k", linestyle="--", lw=1)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_xlim((-0.5, 2.5))
# ax.grid(axis="y")
ax.set_ylabel("Signal Strength")
ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
fig.savefig("summary_plot_combined_channels.pdf", bbox_inches="tight")
# %%
