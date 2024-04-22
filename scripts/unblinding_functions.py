# unblinding_functions.py
import os
import numpy as np
import logging
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from microfit.fileio import from_json
import matplotlib.pyplot as plt


def plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_h0, plot_suffix, plot_title):
    chi2_dict = from_json(os.path.join(output_dir, chi2_results_file))
    chi2_h0 = chi2_dict["chi2_h0"]
    # Exclude large outliers in the distribution to make the histogram more readable
    chi2_h0 = chi2_h0[chi2_h0 < np.percentile(chi2_h0, 99.9)]
    # Get p-value of the observed chi2
    chi2_h0_pval = (chi2_h0 > chi2_at_h0).sum() / len(chi2_h0)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.hist(chi2_h0, bins=100, histtype="step")
    ax.axvline(
        x=chi2_at_h0,
        color="k",
        label=f"Obs. $\\chi^2$ at H0: {chi2_at_h0:.3f}\np-val: {chi2_h0_pval*100:0.3f}%",
    )
    ax.legend()
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("Samples")
    ax.set_title(f"Expected $\\chi^2$ distribution at null, {plot_title}")
    fig.savefig(os.path.join(output_dir, f"chi2_distribution_with_result_{plot_suffix}.pdf"))


def plot_two_hypo_result(output_dir, two_hypo_results_file, delta_chi2, plot_suffix, plot_title):
    two_hypo_results = from_json(os.path.join(output_dir, two_hypo_results_file))
    minimum = np.min(two_hypo_results["samples_h0"])
    minimum = min(minimum, np.min(two_hypo_results["samples_h1"]))
    maximum = np.max(two_hypo_results["samples_h0"])
    maximum = max(maximum, np.max(two_hypo_results["samples_h1"]))

    n_trials = len(two_hypo_results["samples_h0"])
    bin_edges = np.linspace(minimum, maximum, int(np.sqrt(n_trials)))

    # Compute the p-value of the observed delta chi2
    samples_h0 = two_hypo_results["samples_h0"]
    p_val = (samples_h0 > delta_chi2).sum() / len(samples_h0)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    n_h0, *_ = ax.hist(
        two_hypo_results["samples_h0"], bins=bin_edges, histtype="step", density=False, label="H0"
    )
    n_h1, *_ = ax.hist(
        two_hypo_results["samples_h1"], bins=bin_edges, histtype="step", density=False, label="H1"
    )
    # Identify which bin the observed delta_chi2 falls into. Note that
    # digitize returns 1 if the value falls into the first bin, so we
    # subtract 1 to get the correct bin index.
    bin_index = (np.digitize([delta_chi2], bin_edges) - 1)[0]
    bayes_factor = n_h1[bin_index] / n_h0[bin_index]
    bayes_factor_str = f"{bayes_factor:.1f}"
    if bayes_factor < 1:
        bayes_factor = 1 / bayes_factor
        bayes_factor_str = f"1/{bayes_factor:.2g}"
    ax.axvline(
        x=two_hypo_results["ts_median_h1"],
        color="k",
        linestyle="--",
        label=f"Median H1\np-val: {two_hypo_results['median_pval']*100:0.3g}%",
    )
    ax.axvline(
        x=delta_chi2,
        color="k",
        linestyle="-",
        label=f"Obs. $\\Delta \\chi^2$= {delta_chi2:.1f}\nH0 p-val: {p_val*100:0.3g}%\nBF: {bayes_factor_str}",
    )
    ax.legend()
    ax.set_xlabel(r"$\Delta \chi^2$")
    ax.set_ylabel("Samples")
    ax.set_title(f"Two-Hypo. Test, {plot_title}")
    fig.savefig(os.path.join(output_dir, f"two_hypo_result_{plot_suffix}.pdf"))


def compute_confidence_interval(p_values, scan_points, confidence_level):
    p_interp = interp1d(scan_points, p_values)

    def loss(x):
        return (1 - confidence_level - p_interp(x[0])) ** 2

    bounds = [(np.min(scan_points), np.max(scan_points))]
    # Minimization can be tricky because the function isn't perfectly smooth.
    # We run it from a whole grid of starting points.
    starting_grid = np.linspace(bounds[0][0], bounds[0][1], 20)
    fit_results = [
        minimize(loss, x0=[x0], bounds=bounds, method="L-BFGS-B") for x0 in starting_grid
    ]
    # We filter out any bad solution where the fit got stuck in a local
    # miminum. We can expect that, for a good fit, the loss will be
    # very small.
    fit_results_x = [r.x[0] for r in fit_results if r.success and r.fun < 1e-5]

    sorted_crossings = np.sort(fit_results_x)

    # There are two cases to be distinguished: Either there is only one crossing,
    # in which case the values will be very close to each other. Otherwise,
    # the first and last value will be the two crossings.
    if np.abs(sorted_crossings[0] - sorted_crossings[-1]) < 0.01:
        # If there is only one crossing, we want to put the interval between
        # it and either the upper or lower bound of the scan, depending on
        # which one has the smallest p-value.
        p_lower_bound = p_interp(np.min(scan_points))
        p_upper_bound = p_interp(np.max(scan_points))
        if p_lower_bound < p_upper_bound:
            p_lower, p_upper = sorted_crossings[0], np.max(scan_points)
        else:
            p_lower, p_upper = np.min(scan_points), sorted_crossings[-1]
    else:
        p_lower, p_upper = sorted_crossings[0], sorted_crossings[-1]
    return p_lower, p_upper


def plot_confidence_intervals(output_dir, fc_scan_results, plot_suffix, ax=None, xlim=None):
    scan_points = fc_scan_results["scan_points"]
    p_values = np.array(
        [
            np.sum(fc_trials["delta_chi2"] > chi2) / len(fc_trials["delta_chi2"])
            for fc_trials, chi2 in zip(
                fc_scan_results["results"], fc_scan_results["delta_chi2_scan"]
            )
        ]
    )
    p_68_lower, p_68_upper = compute_confidence_interval(p_values, scan_points, 0.68)
    p_90_lower, p_90_upper = compute_confidence_interval(p_values, scan_points, 0.90)
    p_95_lower, p_95_upper = compute_confidence_interval(p_values, scan_points, 0.95)
    p_99_lower, p_99_upper = compute_confidence_interval(p_values, scan_points, 0.99)

    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.plot(scan_points, p_values, color="k", label="p-value from FC trials")

    ax1.fill_between(
        [p_99_lower, p_99_upper],
        [0, 0],
        [1, 1],
        alpha=0.2,
        label=f"99% C.L.: [{p_99_lower:.2f}, {p_99_upper:.2f}]",
        color="C0",
    )
    ax1.fill_between(
        [p_95_lower, p_95_upper],
        [0, 0],
        [1, 1],
        alpha=0.4,
        label=f"95% C.L.: [{p_95_lower:.2f}, {p_95_upper:.2f}]",
        color="C0",
    )
    ax1.fill_between(
        [p_90_lower, p_90_upper],
        [0, 0],
        [1, 1],
        alpha=0.6,
        label=f"90% C.L.: [{p_90_lower:.2f}, {p_90_upper:.2f}]",
        color="C0",
    )
    ax1.fill_between(
        [p_68_lower, p_68_upper],
        [0, 0],
        [1, 1],
        alpha=0.8,
        label=f"68% C.L.: [{p_68_lower:.2f}, {p_68_upper:.2f}]",
        color="C0",
    )

    # ax1.axhline(1 - 0.68, color="k", linestyle="--", lw=1.0)
    # ax1.axhline(1 - 0.90, color="k", linestyle="--", lw=1.0)
    # ax1.text(0, 1 - 0.68, "68% C.L.", ha="left", va="bottom", color="k")
    # ax1.text(0, 1 - 0.90, "90% C.L.", ha="left", va="bottom", color="k")
    ax1.set_ylabel("p-value")
    ax1.set_xlabel("signal strength")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))  # type: ignore
    # ax1.grid()

    if xlim is not None:
        ax1.set_xlim(xlim)
    ax2 = ax1.twinx()
    ax2.plot(scan_points, fc_scan_results["delta_chi2_scan"], color="r", label="Observed $\\Delta \\chi^2$")
    ax2.set_ylabel("$\\Delta \\chi^2$")
    ax2.tick_params(axis='y', colors='r')  # Set tick color to red
    ax2.yaxis.label.set_color('r')  # Set label color to red

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.savefig(os.path.join(output_dir, f"confidence_intervals_{plot_suffix}.pdf"))


def plot_confidence_interval_diagnostic(fc_scan_results):
    scan_points = fc_scan_results["scan_points"]
    fig, ax = plt.subplots()
    ax.plot(scan_points, fc_scan_results["delta_chi2_scan"], label="Real data")
    ax.set_ylabel(r"$\Delta \chi^2$")
    ax.set_xlabel("signal strength")
    fc_trial_delta_chi2 = [r["delta_chi2"] for r in fc_scan_results["results"]]
    ax.boxplot(
        fc_trial_delta_chi2, positions=scan_points, widths=0.1, showfliers=False, manage_ticks=False
    )
    plt.show()
