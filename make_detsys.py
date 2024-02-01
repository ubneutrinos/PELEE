import argparse
import sys

import data_loading as dl
from microfit.histogram import Binning, HistogramGenerator, RunHistGenerator
from microfit.fileio import to_json
import logging
import os
import hashlib
import localSettings as ls

def _get_mc_filter_query(filter_queries):
    """Given a list of filter queries, combine and then negate them.
    
    The result is a query that will get the same sample as was used to make
    the 'mc' histogram in the detector variations.
    """
    query = " & ".join(filter_queries)
    query = "not (" + query + ")"
    return query


def make_variation_histograms(
    run,
    dataset,
    variation,
    selection_query,
    binning,
    use_kde_smoothing=False,
    **dl_kwargs 
):

    """Load the data and Make the json file to store the detector variation predictions"""

    rundata, _, _ = dl.load_runs_detvar(
        run,
        dataset,
        variation,
        **dl_kwargs
    )

    hist_dict = {}
    if use_kde_smoothing:
        # Skip covariance calculation for KDE smoothing since we only need the CV
        options={"bound_transformation": "both", "calculate_covariance": False}
    else:
        options={}
    for dataset in rundata:
        df = rundata[dataset].query(selection_query, engine="python")
        generator = HistogramGenerator(df, binning)
        hist_dict[dataset] = generator.generate(use_kde_smoothing=use_kde_smoothing, options=options)
    #filter_queries["mc"] = _get_mc_filter_query(list(filter_queries.values()))

    return hist_dict

def make_detvar_plots(detvar_data, output_dir):

    """Make plots of the histograms contained in the detvar data"""

    import matplotlib.pyplot as plt

    for truth_filter in detvar_data["variation_hist_data"]:
        fig, ax = plt.subplots()
        hist_dict = detvar_data["variation_hist_data"][truth_filter]
        hist_dict["cv"].draw(ax=ax, label="CV", color="k", show_errors=False, lw=3)
        for name, hist in hist_dict.items():
            if name == "cv":
                continue
            hist.draw(ax=ax, label=name, show_errors=False)
        ax.set_ylabel("Events / POT")
        ax.set_xlabel(detvar_data["binning"].variable)
        ax.set_ylim(bottom=0)
        ax.legend(ncol=2)
        ax.set_title(f"Dataset: {truth_filter}, Selection: {detvar_data['selection_key']}")
        fig.savefig(
            os.path.join(
                output_dir, f"detvar_{truth_filter}_{detvar_data['selection_key']}.pdf",
            ),
            bbox_inches="tight",
        )

# New function to enable integration of detvar generation into
# notebooks insetad of as a separate function

# TODO: tidy up the kwargs for this function

def make_variations(RUN,dataset,selection,preselection,binning,use_kde_smoothing=False,output_file="",make_plots=False,plot_output_dir="",**dl_kwargs):

    selection_query = RunHistGenerator.get_selection_query(
        selection=selection, preselection=preselection
    )
    logging.debug(f"Selection query: {selection_query}")
    variation_hist_data = {}
    filter_queries = {}
    variations = dl.detector_variations
    for variation in variations:
        logging.info(f"Making histogram for variation {variation}")
        (
            variation_hist_data[variation]
        ) = make_variation_histograms(
            RUN,
            dataset,
            variation,
            selection_query,
            binning,
            use_kde_smoothing=use_kde_smoothing,
            **dl_kwargs
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
    #filter_queries = filter_queries[variations[0]]

    print(selection)
    print(preselection)

    binning.selection_key = selection
    binning.preselection_key = preselection

    # TODO: mc_sets should be read from the yml file, and there should be some checks
    # to confirm we're loading the same set of samples for every variation/run
    detvar_data = {
        "run": RUN,
        "selection_key": selection,
        "preselection_key": preselection,
        "selection_query": selection_query,
        "binning": binning,
        "variation_hist_data": variation_hist_data,
        #"filter_queries": filter_queries,
        "mc_sets": ["mc","nue"]
    }

    if output_file == "":
        runcombo_str=""
        for i_r in range(0,len(RUN)):
            runcombo_str = runcombo_str + RUN[i_r]
        output_file = "run_" + runcombo_str + "_" + preselection + "_" + \
                      selection + "_" + binning.variable +\
                      ".json"

    to_json(ls.detvar_cache_path + "/" + output_file, detvar_data)

    if make_plots:
         make_detvar_plots(detvar_data, plot_output_dir)

    return output_file

'''
def main(args):
    RUN = args.run
    selection = args.selection
    preselection = args.preselection
    truth_filtered_sets = args.truth_filtered_sets
    numu=args.numu
    use_kde_smoothing=args.kde_smoothing
    binning_def = list(args.binning_def.split(","))

    binning_def[1] = int(binning_def[1])
    binning_def[2] = (float(binning_def[2]), float(binning_def[3]))
    del binning_def[3]
    binning = Binning.from_config(*binning_def)

    binning.selection_key = args.selection
    binning.preselection_key = args.preselection

    logging.debug(repr(binning))

    make_variations(RUN,selection,preselection,truth_filtered_sets,numu,use_kde_smoothing,binning,args.output_file,args.make_plots,args.plot_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make variation histograms")
    parser.add_argument("--run", nargs='+', help="Run number (as space separated list of strings)", required=True)
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
    parser.add_argument(
        "--truth-filtered-sets",
        type=str,
        nargs="*",
        default=["nue"],
        help="List of truth-filtered sets to use",
    )
    parser.add_argument(
        "--make-plots", action="store_true", help="Make plots of the histograms",
    )
    parser.add_argument(
        "--plot-output-dir", type=str, default=".", help="Output directory for plots",
    )
    parser.add_argument(
        "--numu", action="store_true", help="Use numu selection instead of nue",
    )
    parser.add_argument(
        "--kde-smoothing", action="store_true", help="Use KDE smoothing",
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
'''


