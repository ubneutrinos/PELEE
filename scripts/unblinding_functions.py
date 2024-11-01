# unblinding_functions.py
import os
import numpy as np
import logging
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm
from microfit.fileio import from_json, to_json
import matplotlib.pyplot as plt
from microfit.parameters import ParameterSet, Parameter
from microfit.histogram import Histogram

def extract_bin_range(histogram, bin_range):
    """Extract a range of bins from a given histogram.
    
    The returned Histogram has all of the same properties as the original,
    but with a reduced number of bins. The covariance matrix is also 
    updated accordingly.
    """
    new_binning = histogram.binning.copy()
    new_binning.bin_edges = new_binning.bin_edges[bin_range[0]:bin_range[1] + 2]
    new_covariance_matrix = histogram.covariance_matrix[
        bin_range[0]:bin_range[1]+1, bin_range[0]:bin_range[1]+1
    ]
    new_histogram = Histogram.empty_like(histogram)
    new_histogram.binning = new_binning
    new_histogram.covariance_matrix = new_covariance_matrix
    new_histogram.bin_counts = histogram.bin_counts[bin_range[0]:bin_range[1]+1]
    return new_histogram

def get_signal_count(analysis, channel):
    """Extract the signal count from the analysis for a given channel."""

    original_channels = analysis.signal_channels.copy()
    analysis.signal_channels = [channel]
    h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
    h1_params = ParameterSet([Parameter("signal_strength", 1.0)])

    analysis.set_parameters(h0_params)
    sig_hist_h0 = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        add_precomputed_detsys=True,
        use_sideband=True,
    )[channel]

    analysis.set_parameters(h1_params)
    sig_hist_h1 = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        add_precomputed_detsys=True,
        use_sideband=True,
    )[channel]

    sig_hist_h0_sigrange = extract_bin_range(sig_hist_h0, [0, 4])
    sig_hist_h1_sigrange = extract_bin_range(sig_hist_h1, [0, 4])
    data_hist = analysis.get_data_hist()[channel]
    data_hist_sigrange = extract_bin_range(data_hist, [0, 4])

    analysis.signal_channels = original_channels
    return {
        "e_sig": [(150,850)], #mev
        "observed": data_hist_sigrange.sum(),
        "expected_x0": sig_hist_h0_sigrange.sum(),
        "expected_x0_err_sys": sig_hist_h0_sigrange.sum_std(),
        "expected_x1": sig_hist_h1_sigrange.sum(),
        "expected_x1_err_sys": sig_hist_h1_sigrange.sum_std(),
    }

def get_signal_count_channels(analysis, channels):
    """Get the combined signal counts for one or more channels.
    
    Errors are summed in quadrature.
    """

    # Use get_signal_count on each channel, then combine the results
    signal_counts = [get_signal_count(analysis, channel) for channel in channels]
    combined = {
        "e_sig": [(150,850)],
        "observed": sum([sc["observed"] for sc in signal_counts]),
        "expected_x0": sum([sc["expected_x0"] for sc in signal_counts]),
        "expected_x0_err_sys": sum([sc["expected_x0_err_sys"]**2 for sc in signal_counts])**0.5,
        "expected_x1": sum([sc["expected_x1"] for sc in signal_counts]),
        "expected_x1_err_sys": sum([sc["expected_x1_err_sys"]**2 for sc in signal_counts])**0.5,
    }
    return combined

def get_signal_n_bins(analysis):
    """Get the total number of bins in the signal histogram."""

    sig_hist = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        add_precomputed_detsys=True,
        use_sideband=True,
    )
    return sig_hist.n_bins

def plot_chi2_distribution(output_dir, chi2_results_file, chi2_at_h0, plot_suffix, plot_title):
    chi2_dict = from_json(os.path.join(output_dir, chi2_results_file))
    chi2_h0 = chi2_dict["chi2_h0"]
    # Exclude large outliers in the distribution to make the histogram more readable
    chi2_h0 = chi2_h0[chi2_h0 < np.percentile(chi2_h0, 99.9)]
    # Get p-value of the observed chi2
    chi2_h0_pval = (chi2_h0 > chi2_at_h0).sum() / len(chi2_h0)
    # Get the equivalent significance in units of sigma on a normal
    # distribution
    chi2_h0_sigma = norm.ppf(1 - chi2_h0_pval / 2)  # two-tailed sigma

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.hist(chi2_h0, bins=100, histtype="step")
    ax.axvline(
        x=chi2_at_h0,
        color="k",
        label=f"Obs. $\\chi^2$ at H0: {chi2_at_h0:.1f}\np-val: {chi2_h0_pval*100:0.2g}% $\\approx${chi2_h0_sigma:.2g}$\\sigma$",
    )
    ax.legend()
    ax.set_xlabel(r"$\chi^2$")
    ax.set_ylabel("Samples")
    ax.set_title(f"Expected $\\chi^2$ distribution at null, {plot_title}")
    fig.savefig(os.path.join(output_dir, f"chi2_distribution_with_result_{plot_suffix}.pdf"))


def plot_two_hypo_result(output_dir, two_hypo_results_file, delta_chi2, plot_suffix, plot_title, sensitivity_only=False):
    two_hypo_results = from_json(os.path.join(output_dir, two_hypo_results_file))
    minimum = np.min(two_hypo_results["samples_h0"])
    minimum = min(minimum, np.min(two_hypo_results["samples_h1"]))
    maximum = np.max(two_hypo_results["samples_h0"])
    maximum = max(maximum, np.max(two_hypo_results["samples_h1"]))

    n_trials = len(two_hypo_results["samples_h0"])
    bin_edges = np.linspace(minimum, maximum, int(np.sqrt(n_trials) / 2))

    # Compute the p-value of the observed delta chi2
    samples_h0 = two_hypo_results["samples_h0"]
    samples_h1 = two_hypo_results["samples_h1"]
    p_val_h0 = (samples_h0 < delta_chi2).sum() / len(samples_h0)
    p_val_h1 = (samples_h1 < delta_chi2).sum() / len(samples_h0)
    cls = p_val_h1 / p_val_h0

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    n_h0, *_ = ax.hist(
        two_hypo_results["samples_h0"], bins=bin_edges, histtype="stepfilled", density=True, label="H$_0$", alpha=0.4, edgecolor="C0", linewidth=1
    )
    n_h1, *_ = ax.hist(
        two_hypo_results["samples_h1"], bins=bin_edges, histtype="stepfilled", density=True, label="H$_1$", alpha=0.4, edgecolor="C1", linewidth=1
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
        label=f"Median H$_1$\np-val: {two_hypo_results['median_pval']*100:0.3g}%",
    )
    if not sensitivity_only:
        ax.axvline(
            x=delta_chi2,
            color="k",
            linestyle="-",
            label=f"Obs. $\\Delta \\chi^2$= {delta_chi2:.1f}\nH$_0$ p-val: {p_val_h0*100:0.3g}%\nH$_1$ p-val: {p_val_h1*100:0.3g}%\nCL$_s$: {cls*100:.2g}%\nBF: {bayes_factor_str}",
        )
    ax.legend(title=plot_title)
    ax.set_xlabel(r"$\Delta \chi^2$")
    ax.set_ylabel("Probability density")
    # Extend the xlim to the right
    ax.set_xlim(left=minimum, right=maximum + 0.4 * (maximum - minimum))
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
    # Get the x-value of the maximum p-value
    max_pval_x = scan_points[np.argmax(p_values)]
    # For a two-sided confidence interval, we expect that one crossing will be below
    # the maximum p-value and the other above it. If this is not the case, we
    # take the crossing with the largest value as an upper bound if the crossings are
    # above the maximum, and we take the lowest value as a lower bound if it is below
    # the maximum.
    if len(sorted_crossings) == 0:
        logging.error("No crossing found for confidence interval.")
        return max_pval_x, max_pval_x
    if sorted_crossings[0] > max_pval_x:
        return np.min(scan_points), sorted_crossings[-1]
    if sorted_crossings[-1] < max_pval_x:
        return sorted_crossings[0], np.max(scan_points)
    p_lower = sorted_crossings[0]
    p_upper = sorted_crossings[-1]
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
    # The p-values for the asimov data were inverted
    asimov_sens_pval_at_null = 1.0 - np.array(fc_scan_results["results"][0]["pval"])
    print("Using Asimov sensitivity at injected truth point:")
    print(fc_scan_results["results"][0]["scan_point"])

    # First, calculate all confidence intervals given the p-values from the actual data
    p_68_lower, p_68_upper = compute_confidence_interval(p_values, scan_points, 0.68)
    p_90_lower, p_90_upper = compute_confidence_interval(p_values, scan_points, 0.90)
    p_95_lower, p_95_upper = compute_confidence_interval(p_values, scan_points, 0.95)
    p_99_lower, p_99_upper = compute_confidence_interval(p_values, scan_points, 0.99)
    p_1sig_lower, p_1sig_upper = compute_confidence_interval(p_values, scan_points, 0.6827)
    p_2sig_lower, p_2sig_upper = compute_confidence_interval(p_values, scan_points, 0.9545)
    # Now we also calculate them using the Asimov sensitivity scan
    p_68_lower_sens, p_68_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.68
    )
    p_90_lower_sens, p_90_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.90
    )
    p_95_lower_sens, p_95_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.95
    )
    p_99_lower_sens, p_99_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.99
    )
    p_1sig_lower_sens, p_1sig_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.6827
    )
    p_2sig_lower_sens, p_2sig_upper_sens = compute_confidence_interval(
        asimov_sens_pval_at_null, scan_points, 0.9545
    )

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
    # Create a dictionary containing all of the information needed to 
    # recreate the plot, and store it to JSON.
    confidence_intervals_dict = {
        "scan_points": scan_points.tolist(),
        "p_values": p_values.tolist(),
        "delta_chi2_scan": fc_scan_results["delta_chi2_scan"].tolist(),
        "p_68_lower": p_68_lower,
        "p_68_upper": p_68_upper,
        "p_90_lower": p_90_lower,
        "p_90_upper": p_90_upper,
        "p_95_lower": p_95_lower,
        "p_95_upper": p_95_upper,
        "p_99_lower": p_99_lower,
        "p_99_upper": p_99_upper,
        "p_1sig_lower": p_1sig_lower,
        "p_1sig_upper": p_1sig_upper,
        "p_2sig_lower": p_2sig_lower,
        "p_2sig_upper": p_2sig_upper,
        # Also store Asimov sensitivities
        "p_68_lower_sens": p_68_lower_sens,
        "p_68_upper_sens": p_68_upper_sens,
        "p_90_lower_sens": p_90_lower_sens,
        "p_90_upper_sens": p_90_upper_sens,
        "p_95_lower_sens": p_95_lower_sens,
        "p_95_upper_sens": p_95_upper_sens,
        "p_99_lower_sens": p_99_lower_sens,
        "p_99_upper_sens": p_99_upper_sens,
        "p_1sig_lower_sens": p_1sig_lower_sens,
        "p_1sig_upper_sens": p_1sig_upper_sens,
        "p_2sig_lower_sens": p_2sig_lower_sens,
        "p_2sig_upper_sens": p_2sig_upper_sens,
    }
    print("Storing confidence interval information to JSON...")
    to_json(os.path.join(output_dir, f"confidence_intervals_{plot_suffix}.json"), confidence_intervals_dict)

    return confidence_intervals_dict


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
