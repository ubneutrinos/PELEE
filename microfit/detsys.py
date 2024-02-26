from typing import List, Optional, Union

import data_loading as dl
from microfit.histogram import Binning, HistogramGenerator, MultiChannelBinning
from microfit.fileio import to_json, from_json
import logging
import os
import localSettings as ls


def make_variation_histograms(
    run: List[str], dataset: str, variation: str, binning: Union[Binning, MultiChannelBinning], use_kde_smoothing: bool = False, extra_selection_query: Optional[str] = None, **dl_kwargs
):
    """Load the data and Make the json file to store the detector variation predictions"""

    rundata, _, data_pot = dl.load_runs_detvar(run, dataset, variation, **dl_kwargs)

    hist_dict = {}
    if use_kde_smoothing:
        # Skip covariance calculation for KDE smoothing since we only need the CV
        options = {"bound_transformation": "both", "calculate_covariance": False}
    else:
        options = {}
    for dataset in rundata:
        generator = HistogramGenerator(rundata[dataset], binning)
        hist_dict[dataset] = generator.generate(
            use_kde_smoothing=use_kde_smoothing, options=options, extra_query=extra_selection_query
        )

    return hist_dict


def make_detvar_plots(detvar_data, output_dir, plotname):
    """Make plots of the histograms contained in the detvar data"""

    import matplotlib.pyplot as plt

    for truth_filter in detvar_data["variation_hist_data"]:
        print("Making plots for", truth_filter)
        fig, ax = plt.subplots()
        hist_dict = detvar_data["variation_hist_data"][truth_filter]
        hist_dict["cv"].draw(ax=ax, label="CV", color="k", show_errors=False, lw=3)
        for name, hist in hist_dict.items():
            if name == "cv":
                continue
            hist.draw(ax=ax, label=name, show_errors=False)
        # ax.set_ylabel("Events / POT")
        # ax.set_ylabel("Events")
        # ax.set_xlabel(detvar_data["binning"].variable)
        # ax.set_ylim(bottom=0)
        # ax.legend(ncol=2)
        ax.set_title(f"Dataset: {truth_filter}")

        fig.savefig(output_dir + truth_filter + "_" + plotname)

def _sanitize_selection_query(query: str) -> str:
    """Remove any characters that are not legal inside a file name from the selection query."""

    return query.replace(" ", "_").replace(">", "gt").replace("<", "lt").replace("=", "eq").replace("!", "not").replace("&", "and").replace("|", "or")

def _negate_query(query: str) -> str:
    """Negate the selection query."""

    return "!(" + query + ")"

def make_variations(
    run_numbers: List[str],
    data: str,
    binning: Union[Binning, MultiChannelBinning],
    selection: Optional[str] = None,
    preselection: Optional[str] = None,
    use_kde_smoothing: bool = False,
    make_plots: bool = False,
    plot_output_dir: str = ".",
    enable_detvar_cache: bool = False,
    detvar_cache_dir: Optional[str] = None,
    extra_selection_query: Optional[str] = None,
    **dl_kwargs,
):

    runcombo_str = ""
    for i_r in range(0, len(run_numbers)):
        runcombo_str = runcombo_str + run_numbers[i_r]
    channels_str = "_".join(binning.channels)
    output_file = (
        "run_" + runcombo_str + "_" + channels_str + "_" + data
    )
    if extra_selection_query is not None:
        output_file += "_" + _sanitize_selection_query(extra_selection_query)
    detvar_file = ls.detvar_cache_path or detvar_cache_dir
    assert detvar_file is not None, "detvar_cache_dir must be provided if detvar_cache_path is not set"
    detvar_file += "/" + output_file + ".json"

    if enable_detvar_cache and os.path.isfile(detvar_file):
        print("Loading devar histograms from file:", detvar_file)
        detvar_data = from_json(detvar_file)
        if make_plots:
            make_detvar_plots(detvar_data, plot_output_dir, output_file + ".pdf")
        return detvar_data

    if isinstance(binning, MultiChannelBinning):
        assert selection is None and preselection is None, "Cannot pass selection and preselection with MultiChannelBinning"
    elif binning.selection_query is None:
        assert selection is not None, "Selection must be provided if binning.selection_query is None"
        binning.set_selection(selection=selection, preselection=preselection)
    logging.debug(f"Binning: {binning}")
    variation_hist_data = {}
    variations = dl.detector_variations

    for variation in variations:
        logging.info(f"Making histogram for variation {variation}")
        variation_hist_data[variation] = make_variation_histograms(
            run_numbers,
            data,
            variation,
            binning,
            use_kde_smoothing=use_kde_smoothing,
            extra_selection_query=extra_selection_query,
            **dl_kwargs,
        )

    # Switch ordering of keys in the variation_hist_data dict. Instead of
    # variation_hist_data[variation][dataset], we want to have
    # variation_hist_data[dataset][variation]
    variation_hist_data = {
        dataset: {variation: variation_hist_data[variation][dataset] for variation in variations}
        for dataset in variation_hist_data[variations[0]]
    }

    detvar_data = {
        "run": run_numbers,
        "binning": binning,
        "variation_hist_data": variation_hist_data,
        "mc_sets": list(variation_hist_data.keys()),
        "extra_selection_query": extra_selection_query,
    }

    if enable_detvar_cache:
        to_json(detvar_file, detvar_data)

    if make_plots:
        make_detvar_plots(detvar_data, plot_output_dir, output_file + ".pdf")

    return detvar_data