import sys
import os
import argparse
from matplotlib import pyplot as plt

sys.path.append(".")
from microfit.analysis import MultibandAnalysis
from microfit.histogram import MultiChannelHistogram


def print_error_budget_tables(analysis: MultibandAnalysis, output_file: str):
    analysis.parameters["signal_strength"].value = 0.0
    hist_all_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=True,
    )
    # The unisim errors are GENIE knobs, so we include them but only for the GENIE errors
    hist_genie_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        ms_columns=["weightsGenie"],
        include_unisim_errors=True,
        include_stat_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=False,
    )
    hist_flux_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        ms_columns=["weightsFlux"],
        include_stat_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=False,
    )
    hist_reint_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        ms_columns=["weightsReint"],
        include_stat_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=False,
    )
    hist_stat_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=False,
        use_sideband=False,
        include_stat_errors=True,
        include_unisim_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=False,
    )
    hist_detsys_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=False,
        use_sideband=False,
        include_stat_errors=False,
        include_unisim_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=True,
    )
    hist_all_except_stats = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        include_stat_errors=False,
        include_non_signal_channels=True,
        add_precomputed_detsys=True,
    )

    with open(output_file, "w") as file:
        for channel in analysis.signal_channels + analysis.constraint_channels:
            bin_counts = hist_all_errors[channel].bin_counts
            # Total error as fraction of bin count
            total_errors = hist_all_errors[channel].std_devs / bin_counts
            # Every error source as fraction of total error
            genie_errors = hist_genie_errors[channel].std_devs / hist_all_errors[channel].std_devs
            flux_errors = hist_flux_errors[channel].std_devs / hist_all_errors[channel].std_devs
            reint_errors = hist_reint_errors[channel].std_devs / hist_all_errors[channel].std_devs
            detsys_errors = hist_detsys_errors[channel].std_devs / hist_all_errors[channel].std_devs
            all_except_stat_errors = (
                hist_all_except_stats[channel].std_devs / hist_all_errors[channel].std_devs
            )
            stat_errors = hist_stat_errors[channel].std_devs / hist_all_errors[channel].std_devs

            # For every bin, we want to generate the following columns:
            # energy range | genie | flux | reint | genie + flux + reint | stat | total
            # and write them to the file.
            # Print a header first, make output compatible with booktabs package
            selection_tex = hist_all_errors[channel].binning.selection_tex
            file.write("\\begin{table}\n")
            file.write(f"\\caption{{Error budget for {selection_tex}}}\n")
            file.write("\\begin{tabular}{lccccccc}\n")
            file.write("\\toprule\n")
            file.write(
                rf"{hist_all_errors[channel].binning.variable_tex_short} & "
                "GENIE & "
                "Flux & "
                "G4 & "  # The hadronic reintegration errors can be thought of as Geant4 errors
                "Detsys &"
                "\\thead{GENIE \\\\+ Flux \\\\+ Reint \\\\+ Detsys} & "
                "Stat & "
                "Total $\\sigma/\\mu$\\\\\n"
            )
            file.write("\\midrule\n")
            bin_edges = hist_all_errors[channel].binning.bin_edges
            n_bins = len(bin_edges) - 1
            for i in range(n_bins):
                file.write(
                    f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} & "
                    f"{genie_errors[i]:.2f} & "
                    f"{flux_errors[i]:.2f} & "
                    f"{reint_errors[i]:.2f} & "
                    f"{detsys_errors[i]:.2f} & "
                    f"{all_except_stat_errors[i]:.2f} & "
                    f"{stat_errors[i]:.2f} & "
                    f"{total_errors[i]:.2f} \\\\\n"
                )
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")
            file.write("\\end{table}\n")


def print_constraint_error_reduction_table(analysis: MultibandAnalysis, output_file: str):
    """Print a table showing for every bin the reduction of the error due to constraints."""

    analysis.parameters["signal_strength"].value = 0.0
    hist_all_errors = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        include_non_signal_channels=False,
        add_precomputed_detsys=True,
    )
    hist_after_constraint = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=True,
        include_non_signal_channels=False,
        add_precomputed_detsys=True,
    )
    with open(output_file, "w") as file:
        for channel in analysis.signal_channels:
            # We want to make a table with the following columns:
            # energy range | error before constraint | error after constraint | reduction
            # Where the error is shown as a fraction of the bin count, and the reduction is
            # the relative difference between the two.
            # Print a header first, make output compatible with booktabs package
            selection_tex = hist_all_errors[channel].binning.selection_tex
            file.write("\\begin{table}\n")
            file.write(f"\\caption{{Error reduction due to constraints for {selection_tex}}}\n")
            file.write("\\begin{tabular}{lccc}\n")
            file.write("\\toprule\n")
            file.write(" & \\multicolumn{2}{c}{Error} &  \\\\ \n")
            file.write("\\cmidrule(lr){2-3}\n")
            file.write(
                rf"{hist_all_errors[channel].binning.variable_tex_short} & "
                "before constraint & "
                "after constraint & "
                "Reduction \\\\\n"
            )
            file.write("\\midrule\n")
            bin_edges = hist_all_errors[channel].binning.bin_edges
            n_bins = len(bin_edges) - 1
            for i in range(n_bins):
                error_before = (
                    hist_all_errors[channel].std_devs[i]
                    / hist_all_errors[channel].bin_counts[i]
                    * 100
                )
                error_after = (
                    hist_after_constraint[channel].std_devs[i]
                    / hist_after_constraint[channel].bin_counts[i]
                    * 100
                )
                reduction = (error_before - error_after) / error_before * 100
                file.write(
                    f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} & "
                    f"{error_before:.1f}\\% & "
                    f"{error_after:.1f}\\% & "
                    f"{reduction:.1f}\\% \\\\\n"
                )
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")
            file.write("\\end{table}\n")


def plot_correlations(analysis: MultibandAnalysis, plot_dir: str):
    fig, ax = analysis.plot_correlation()
    ax.set_title("Correlation w/ Detector Systematics")
    fig.savefig(os.path.join(plot_dir, "multiband_correlation.pdf"))

    fig, ax = analysis.plot_correlation(add_precomputed_detsys=False)
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_no_detsys.pdf"))

    fig, ax = analysis.plot_correlation(
        ms_columns=["weightsGenie"], include_unisim_errors=True, add_precomputed_detsys=False
    )
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsGenie_w_unisim.pdf"))

    fig, ax = analysis.plot_correlation(
        ms_columns=["weightsFlux"], include_unisim_errors=False, add_precomputed_detsys=False
    )
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsFlux.pdf"))

    fig, ax = analysis.plot_correlation(
        ms_columns=["weightsReint"], include_unisim_errors=False, add_precomputed_detsys=False
    )
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsReint.pdf"))


def plot_constraint_update(
    analysis: MultibandAnalysis, plot_dir: str, print_tables: bool = False, figsize=(5, 4)
):
    analysis.parameters["signal_strength"].value = 0.0
    h0_hist_unconstrained = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=False,
        add_precomputed_detsys=True,
    )
    h0_hist_constrained = analysis.generate_multiband_histogram(
        include_multisim_errors=True,
        use_sideband=True,
        add_precomputed_detsys=True,
    )
    if print_tables:
        print_error_budget_tables(analysis, os.path.join(plot_dir, "error_budget.tex"))
        if len(analysis.constraint_channels) > 0:
            print_constraint_error_reduction_table(
                analysis, os.path.join(plot_dir, "constraint_error_reduction.tex")
            )

    if isinstance(h0_hist_unconstrained, MultiChannelHistogram):
        for channel in h0_hist_unconstrained.channels:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            h0_hist_unconstrained[channel].draw(ax=ax, label="Unconstrained")
            h0_hist_constrained[channel].draw(ax=ax, label="Constrained")
            binning = h0_hist_unconstrained[channel].binning
            ax.legend(title="Total (MC + EXT)")
            ax.set_title(f"Selection: {binning.selection_tex}")
            fig.savefig(os.path.join(plot_dir, f"h0_constrained_vs_unconstrained_{channel}.pdf"))
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        h0_hist_unconstrained.draw(ax=ax, label="Unconstrained")
        h0_hist_constrained.draw(ax=ax, label="Constrained")
        ax.legend(title="Total (MC + EXT)")
        fig.savefig(os.path.join(plot_dir, "h0_constrained_vs_unconstrained.pdf"))


def main(config_file, plot_dir, print_tables):
    # make the output dir if it does not exist yet
    if plot_dir is not None and not os.path.exists(plot_dir):
        print(f"Creating output directory {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)
    print("Setting up analysis...")
    analysis = MultibandAnalysis.from_toml(config_file, output_dir=plot_dir)
    if plot_dir is None:
        plot_dir = analysis.output_dir
        assert (
            plot_dir is not None
        ), "Output directory must be set in the config file or passed as an argument"
    analysis.print_configuration()
    print("Plotting sidebands...")
    analysis.plot_sidebands(
        show_chi_square=True,
        separate_figures=True,
        save_path=plot_dir,
        filename_format="sideband_{}.pdf",
    )
    print("Plotting signals...")
    analysis.plot_signals(
        separate_figures=True, save_path=plot_dir, filename_format="signal_{}.pdf"
    )
    print("Plotting correlations...")
    plot_correlations(analysis, plot_dir)

    print("Plotting constraint update...")
    plot_constraint_update(analysis, plot_dir, print_tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make analysis plots")
    parser.add_argument("--configuration", help="Path to analysis configuration file")
    parser.add_argument(
        "--output-dir",
        help="Path to output directory. Overrides path given in config file.",
        required=False,
        type=str,
    )
    parser.add_argument("--print-tables", action="store_true", help="Print error budget tables")
    args = parser.parse_args()
    main(args.configuration, args.output_dir, args.print_tables)
