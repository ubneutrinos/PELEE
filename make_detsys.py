import argparse
import sys

import data_loading as dl
from microfit.histogram import Binning, HistogramGenerator, RunHistGenerator
from microfit.fileio import to_json
import logging
import os


def _get_mc_filter_query(filter_queries):
    """Given a list of filter queries, combine and then negate them.
    
    The result is a query that will get the same sample as was used to make
    the 'mc' histogram in the detector variations.
    """
    query = " & ".join(filter_queries)
    query = "not (" + query + ")"
    return query

def make_variation_histograms(run, variation, selection_query, binning_def, truth_filtered_sets=["nue"]):
    binning = Binning.from_config(*binning_def)
    rundata, filter_queries = dl.load_run_detvar(
        run,
        variation,
        truth_filtered_sets=truth_filtered_sets,
        loadpi0variables=True,
        loadshowervariables=True,
        loadrecoveryvars=True,
        loadsystematics=False,
        use_bdt=True,
        enable_cache=True,
    )
    hist_dict = {}
    for dataset in rundata:
        df = rundata[dataset].query(selection_query, engine="python")
        generator = HistogramGenerator(df, binning)
        hist_dict[dataset] = generator.generate()
    filter_queries["mc"] = _get_mc_filter_query(list(filter_queries.values()))
    return hist_dict, filter_queries


def main(args):
    RUN = args.run
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
    variation_hist_data = {}
    filter_queries = {}
    variations = dl.detector_variations
    for variation in variations:
        logging.info(f"Making histogram for variation {variation}")
        variation_hist_data[variation], filter_queries[variation] = make_variation_histograms(
            RUN, variation, selection_query, binning_def
        )

    # Switch ordering of keys in the variation_hist_data dict. Instead of
    # variation_hist_data[variation][dataset], we want to have
    # variation_hist_data[dataset][variation]
    variation_hist_data = {
        dataset: {
            variation: variation_hist_data[variation][dataset]
            for variation in variations
        }
        for dataset in variation_hist_data[variations[0]]
    }
    # Also, we can assume that the filter_queries are the same for all
    # variations, so we can just take the first one
    filter_queries = filter_queries[variations[0]]

    detvar_data = {
        "run": RUN,
        "selection": selection,
        "preselection": preselection,
        "selection_query": selection_query,
        "binning": binning,
        "variation_hist_data": variation_hist_data,
        "filter_queries": filter_queries,
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
