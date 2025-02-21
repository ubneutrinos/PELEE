# %%
import sys, os 
sys.path.append("../../")
import numpy as np
from microfit.histogram import Binning, RunHistGenerator
from microfit.run_plotter import RunHistPlotter

from data_loading import load_runs
# %%
RUN = ["1","2","3"]

rundata_numu, mc_weights_numu, data_pot_numu = load_runs(
    RUN,
    data="bnb",
    truth_filtered_sets=["nue", "drt"],
    loadshowervariables=False,
    loadrecoveryvars=False,
    loadsystematics=True,
    use_bdt=False,
    numupresel=True,
    loadnumuvariables=True,
    load_crt_vars=False,
    blinded=False, 
    enable_cache=True,
)
# %%
numu_binning = Binning.from_config("muon_energy", 14, (0.15, 1.55), "muon reconstructed energy [GeV]")
numu_binning.label = "NUMU0PI"
numu_binning.set_selection(preselection="NUMU", selection="NUMU0PI")
print(numu_binning)
# %%
numu_sideband_generator = RunHistGenerator(
    rundata_numu,
    numu_binning,
    data_pot=data_pot_numu,
)
# %%
ax, ax_ratio = RunHistPlotter(numu_sideband_generator).plot(
    category_column="paper_category",
    include_multisim_errors=True,
    show_data_mc_ratio=True,
    add_ext_error_floor=False,
    add_precomputed_detsys=False,
    channel="NUMU0PI",
    show_chi_square=True,
    figsize=(5, 4)
)
# %%
ax.get_figure().savefig("numu0pi_runs123.png", dpi=200)
# %%
