import sys, os
sys.path.append(".")
from data_loading import load_runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN = ["1","2","3","4b","4c","4d","5"]  # this can be a list of several runs, i.e. [1,2,3]
rundata, mc_weights, data_pot = load_runs(
    RUN,
    data="two_shr_sideband",  # which data to load
    # truth_filtered_sets=["nue", "drt", "nc_pi0", "cc_pi0", "cc_nopi", "cc_cpi", "nc_nopi", "nc_cpi"],
    # Which truth-filtered MC sets to load in addition to the main MC set. At least nu_e and dirt
    # are highly recommended because the statistics at the final level of the selection are very low.
    truth_filtered_sets=["nue", "drt"],
    # Choose which additional variables to load. Which ones are required may depend on the selection
    # you wish to apply.
    loadpi0variables=True,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    # Load the nu_e set one more time with the LEE weights applied
    load_lee=True,
    # With the cache enabled, by default the loaded dataframes will be stored as HDF5 files
    # in the 'cached_dataframes' folder. This will speed up subsequent loading of the same data.
    enable_cache=True,
    blinded=False,
)

from microfit.histogram import Binning, Histogram, RunHistGenerator, MultiChannelBinning
from microfit.run_plotter import RunHistPlotter

sideband_binning = Binning.from_config("reco_e", 10, (0.15, 1.55), "neutrino reconstructed energy [GeV]")
sideband_generator_ZPBDTTWOSHR = RunHistGenerator(
    rundata,
    sideband_binning,
    data_pot=data_pot,
    selection="ZPBDTTWOSHR",
    preselection="ZPTwoShr",
)
sideband_plotter = RunHistPlotter(sideband_generator_ZPBDTTWOSHR)
sideband_plotter.plot(include_multisim_errors=True, show_data_mc_ratio=True, add_ext_error_floor=False,
                      category_column="category")
plt.savefig("sideband_ZPBDTTWOSHR_runs_1-5.pdf")

RUN = ["3","4b","4c","4d","5"]  # this can be a list of several runs, i.e. [1,2,3]

rundata_numu, mc_weights_numu, data_pot_numu = load_runs(
    RUN,
    data="muon_sideband",
    truth_filtered_sets=["nue", "drt"],
    loadshowervariables=True,
    loadsystematics=True,
    use_bdt=False,
    # set this to true to let the data loading function know that we are loading numu
    numupresel=True,
    loadnumuvariables=True,
    load_crt_vars=True,
    blinded=False,  # sideband needs to be unblinded to work
    enable_cache=True,
)

for key in rundata_numu:
    rundata_numu[key]['npi'] = rundata_numu[key].eval('npion+npi0')

sideband_binning = Binning.from_config("neutrino_energy", 14, (0.15, 1.55), "neutrino reconstructed energy [GeV]")
sideband_generator_NUMUCRTNP0PI = RunHistGenerator(
    rundata_numu,
    sideband_binning,
    data_pot=data_pot_numu,
    selection="NUMUCRTNP0PI",
    preselection="NUMUCRT",
)
sideband_plotter = RunHistPlotter(sideband_generator_NUMUCRTNP0PI)
ax = sideband_plotter.plot(include_multisim_errors=True, show_data_mc_ratio=True, add_ext_error_floor=False,
                           category_column="category")[0]
ax.set_ylim(0.0, ax.get_ylim()[1] * 1.5)
plt.savefig("sideband_NUMUCRTNP0PI_runs_1-5.pdf")

sideband_binning = Binning.from_config("neutrino_energy", 14, (0.15, 1.55), "neutrino reconstructed energy [GeV]")
sideband_generator_NUMUCRT0P0PI = RunHistGenerator(
    rundata_numu,
    sideband_binning,
    data_pot=data_pot_numu,
    selection="NUMUCRT0P0PI",
    preselection="NUMUCRT",
)
sideband_plotter = RunHistPlotter(sideband_generator_NUMUCRT0P0PI)
ax = sideband_plotter.plot(include_multisim_errors=True, show_data_mc_ratio=True, add_ext_error_floor=False,
                           category_column="category")[0]
ax.set_ylim(0.0, ax.get_ylim()[1] * 1.5)
plt.savefig("sideband_NUMUCRT0P0PI_runs_1-5.pdf")

from microfit.analysis import MultibandAnalysis

# This is somewhat of an abuse of the Analysis class, since we treat the sideband as the signal,
# but it is the simplest way to plot correlations.
sideband_analysis = MultibandAnalysis(
    signal_generators = [sideband_generator_ZPBDTTWOSHR, sideband_generator_NUMUCRTNP0PI, sideband_generator_NUMUCRT0P0PI],
    signal_names=["ZPBDTTWOSHR", "NUMUCRTNP0PI", "NUMUCRT0P0PI"],
)

fig, axs = sideband_analysis.plot_signals()
fig.savefig("sidebands_runs_1-5.pdf")

fig, axs = sideband_analysis.plot_correlation()
fig.savefig("sideband_correlation_runs_1-5.pdf")

fig, axs = sideband_analysis.plot_correlation(ms_column="weightsGenie", with_unisim=True)
fig.savefig("sideband_correlation_runs_1-5_weightsGenie.pdf")

fig, axs = sideband_analysis.plot_correlation(ms_column="weightsFlux")
fig.savefig("sideband_correlation_runs_1-5_weightsFlux.pdf")

fig, axs = sideband_analysis.plot_correlation(ms_column="weightsReint")
fig.savefig("sideband_correlation_runs_1-5_weightsReint.pdf")