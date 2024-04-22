"""Run a two-hypothesis test from a pre-configured analysis."""
import sys, os

sys.path.append(".")
from microfit.analysis import MultibandAnalysis
from microfit.parameters import ParameterSet
from matplotlib import pyplot as plt
import logging
import argparse
import toml
import numpy as np
from microfit.fileio import to_json, from_json


def run_analysis(args):
    # assert that configuration is given
    assert args.configuration is not None, "configuration must be given"
    # make sure that output directory exists
    if args.output_dir is None:
        assert "persistence" in args.configuration
        args.output_dir = args.configuration["persistence"].pop("output_dir")
        assert args.output_dir, "output_dir must be given in configuration or as argument"
    os.makedirs(args.output_dir, exist_ok=True)
    assert "h0_params" in args.configuration, "h0_params must be given in configuration"
    assert "h1_params" in args.configuration, "h1_params must be given in configuration"
    h0_params = ParameterSet.from_dict(args.configuration.pop("h0_params"))
    h1_params = ParameterSet.from_dict(args.configuration.pop("h1_params"))
    analysis = MultibandAnalysis(args.configuration, output_dir=args.output_dir)
    analysis.print_configuration()
    print(f"Null hypothesis parameters: {h0_params}")
    print(f"Alternative hypothesis parameters: {h1_params}")

    two_hypo_results = analysis.two_hypothesis_test(
        h0_params=h0_params,
        h1_params=h1_params,
        n_trials=args.n_trials,
        scale_to_pot=args.scale_to_pot,
        sensitivity_only=args.sensitivity_only,
    )

    # save results to file
    to_json(os.path.join(args.output_dir, "two_hypothesis_test.json"), two_hypo_results)


def plot_results(args):
    # read results from file
    two_hypo_results = from_json(os.path.join(args.output_dir, "two_hypothesis_test.json"))
    n_trials = len(two_hypo_results["samples_h0"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.hist(
        two_hypo_results["samples_h0"],
        bins=int(np.sqrt(n_trials)),
        histtype="step",
        density=True,
        label="H0",
    )
    ax.hist(
        two_hypo_results["samples_h1"],
        bins=int(np.sqrt(n_trials)),
        histtype="step",
        density=True,
        label="H1",
    )
    ax.axvline(
        x=two_hypo_results["ts_median_h1"],
        color="k",
        linestyle="--",
        label=f"Median H1\np-val: {two_hypo_results['median_pval']:0.3f}",
    )
    ax.legend()
    ax.set_xlabel(r"$\Delta \chi^2$")
    ax.set_ylabel("Probability density")
    fig.savefig(os.path.join(args.output_dir, "two_hypothesis_test.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", help="Function to run")
    parser.add_argument("--configuration", help="Path to analysis configuration file")
    parser.add_argument(
        "--output-dir",
        help="Path to output directory. Overrides configuration file if passed.",
        required=False,
        type=str,
    )
    parser.add_argument("--n-trials", type=int, default=10000, help="Number of trials to run")
    parser.add_argument("--scale-to-pot", type=float, default=None, help="Scale to this POT")
    parser.add_argument(
        "--sensitivity-only", action="store_true", help="Only calculate sensitivity"
    )
    args = parser.parse_args()
    if args.configuration is not None:
        assert os.path.exists(
            args.configuration
        ), f"Configuration file {args.configuration} does not exist"
        args.configuration = toml.load(args.configuration)
        assert args.configuration, "Configuration file is empty"

    if args.function == "run_analysis":
        run_analysis(args)
    elif args.function == "plot_results":
        plot_results(args)
