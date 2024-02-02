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
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "fc_scan_results.json")
    if os.path.exists(results_file) and not args.force_rescan:
        print(f"Loading previous results from {results_file}")
        fc_scan_results = from_json(results_file)
        if "scan_points" not in fc_scan_results:
            fc_scan_results["scan_points"] =  np.array([result["scan_point"] for result in fc_scan_results["results"]])
    else:
        fc_scan_results = None
    assert "fc_scan" in args.configuration, "Configuration must contain a 'fc_scan' section"
    fc_config = args.configuration["fc_scan"]
    scan_points = np.linspace(*fc_config["range"], num=fc_config["n_points"])
    if fc_scan_results is not None:
        # check that the scan points are the same
        assert np.allclose(scan_points, fc_scan_results["scan_points"]), "Scan points do not match previous results"
    fit_grid = dict(
        [(k["parameter"], np.linspace(*k["range"], num=k["n_points"])) for k in fc_config["fit_grid"]]
    )
    if fc_scan_results is not None:
        print(f"Resuming from previous results with {len(fc_scan_results['results'][0]['delta_chi2'])} trials per point")
    print(f"Running scan with {len(scan_points)} scan points")
    analysis = MultibandAnalysis(args.configuration)
    print(f"Running analysis with parameters: {analysis.parameters}")
    fc_scan_results = analysis.scan_asimov_fc_sensitivity(
        scan_points=scan_points,
        parameter_name="signal_strength",
        n_trials=args.n_trials,
        fit_method="grid_scan",
        fit_grid=fit_grid,
        fc_scan_results=fc_scan_results,
    )

    # save results to file
    to_json(os.path.join(args.output_dir, "fc_scan_results.json"), fc_scan_results)
    
def plot_results(args):
    # read results from file
    fc_scan_results = from_json(os.path.join(args.output_dir, "fc_scan_results.json"))
    
    fig, ax = plt.subplots(figsize=(6, 5))

    MultibandAnalysis.plot_fc_scan_results(
        fc_scan_results,
        parameter_tex="signal strength",
        ax=ax,
    )
    if args.title:
        ax.set_title(args.title)
    print(f"Saving results to {os.path.join(args.output_dir, 'fc_scan_results.pdf')}")
    fig.savefig(os.path.join(args.output_dir, "fc_scan_results.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", help="Function to run")
    parser.add_argument("--configuration", help="Path to analysis configuration file")
    parser.add_argument("--output-dir", help="Path to output directory", required=True, type=str)
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of trials to run")
    parser.add_argument("--force-rescan", action="store_true", help="Force running of all trials. If not set, will resume from previous results and only rerun asimov scan and computation of p-values.")
    parser.add_argument("--title", help="Title for the plot. Only used when plotting results.")
    args = parser.parse_args()
    if args.configuration is not None:
        assert os.path.exists(args.configuration), f"Configuration file {args.configuration} does not exist"
        args.configuration = toml.load(args.configuration)
        assert args.configuration, "Configuration file is empty"
    
    if args.function == "run_analysis":
        run_analysis(args)
    elif args.function == "plot_results":
        plot_results(args)

    
