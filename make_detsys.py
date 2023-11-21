import argparse
import sys

import data_loading as dl
from microfit.histogram import Binning, HistogramGenerator, RunHistGenerator
from microfit.fileio import to_json
import logging
import os


def make_variation_histogram(run, variation, selection_query, binning_def, data_pot):
    binning = Binning.from_config(*binning_def)
    # TODO: Include other MC sets besides nue
    rundata, mc_weights, data_pot = dl._load_run_detvar(
        run,
        variation,
        data_pot,
        mc_sets=["nue"],
        loadpi0variables=True,
        loadshowervariables=True,
        loadrecoveryvars=True,
        loadsystematics=False,
        use_bdt=True,
        enable_cache=True,
    )
    df = rundata["nue"].query(selection_query, engine="python")
    logging.debug(f"Total number of events: {len(df)}")
    generator = HistogramGenerator(df, binning)
    hist = generator.generate()
    return hist


def main(args):
    RUN = args.run
    # The number here is arbitrary, as the covariance is rescaled to the data POT
    # by the RunHistGenerator anyways.
    data_pot = 1.0e20
    selection = args.selection
    preselection = args.preselection

    binning_def = list(args.binning_def.split(","))
    binning_def[1] = int(binning_def[1])
    binning_def[2] = (float(binning_def[2]), float(binning_def[3]))
    del binning_def[3]
    binning = Binning.from_config(*binning_def)
    logging.debug(repr(binning))

    selection_query = RunHistGenerator.get_selection_query(
        selection=selection, preselection=preselection
    )
    logging.debug(f"Selection query: {selection_query}")
    variation_hists = {}
    variations = dl.detector_variations
    for variation in variations:
        logging.info(f"Making histogram for variation {variation}")
        variation_hists[variation] = make_variation_histogram(
            RUN, variation, selection_query, binning_def, data_pot
        )

    detvar_data = {
        "run": RUN,
        "selection": selection,
        "preselection": preselection,
        "selection_query": selection_query,
        "binning": binning,
        "data_pot": data_pot,
        "variation_hists": variation_hists,
    }

    to_json(args.output_file, detvar_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make variation histograms")
    parser.add_argument("--run", type=str, help="Run number (as string)", required=True)
    parser.add_argument(
        "--selection", type=str, help="Selection criteria", required=True
    )
    parser.add_argument(
        "--preselection", type=str, help="Preselection criteria", required=True
    )
    parser.add_argument("--output-file", type=str, help="Output file", required=True)
    parser.add_argument(
        "--binning-def",
        type=str,
        help="Binning definition. Required format is: ('variable_name', n_bins, min, max)",
        required=True,
    )
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )
    args = parser.parse_args()

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbosity, 2)]
    logging.basicConfig(level=log_level)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)

    main(args)
