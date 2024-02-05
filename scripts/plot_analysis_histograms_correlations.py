
import sys
import os
import argparse
from matplotlib import pyplot as plt
sys.path.append(".")
from microfit.analysis import MultibandAnalysis

def main(config_file, plot_dir):
    print("Setting up analysis...")
    analysis = MultibandAnalysis.from_toml(config_file)
    print("Plotting...")
    analysis.plot_sidebands(show_chi_square=True, separate_figures=True, save_path=plot_dir, filename_format="sideband_{}.pdf")
    analysis.plot_signals(separate_figures=True, save_path=plot_dir, filename_format="signal_{}.pdf")
    fig, ax = analysis.plot_correlation()
    fig.savefig(os.path.join(plot_dir, "multiband_correlation.pdf"))

    fig, ax = analysis.plot_correlation(ms_columns=["weightsGenie"], include_unisim_errors=True)
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsGenie_w_unisim.pdf"))

    fig, ax = analysis.plot_correlation(ms_columns=["weightsFlux"], include_unisim_errors=False)
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsFlux.pdf"))

    fig, ax = analysis.plot_correlation(ms_columns=["weightsReint"], include_unisim_errors=False)
    fig.savefig(os.path.join(plot_dir, "multiband_correlation_weightsReint.pdf"))

    analysis.parameters["signal_strength"].value = 0.0
    h0_hist_unconstrained = analysis.generate_multiband_histogram(
        include_multisim_errors=True, use_sideband=False,
    )
    h0_hist_constrained = analysis.generate_multiband_histogram(
        include_multisim_errors=True, use_sideband=True,
    )

    fig, ax = plt.subplots()
    h0_hist_unconstrained.draw(ax=ax, label="Unconstrained")
    h0_hist_constrained.draw(ax=ax, label="Constrained")
    ax.legend(title="Total (MC + EXT)")
    fig.savefig(os.path.join(plot_dir, "h0_constrained_vs_unconstrained.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make analysis plots")
    parser.add_argument("config_file", type=str, help="Path to the config file")
    parser.add_argument("plot_dir", type=str, help="Directory for plot output")
    args = parser.parse_args()
    main(args.config_file, args.plot_dir)






