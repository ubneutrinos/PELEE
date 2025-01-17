import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections

keep_vars = [
    'run', 'sub', 'evt',
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", "RecoECal", "Reco_mag_q", "RecoPL",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len"
]

#RUN = ["5"]
#RUN = ["1","2","3_nocrt","3_crt","4b","4c","4d","5"]
#RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
RUN = ["1","2","3","4b","4c","4d","5"]

# Choose selections. If not, set as "None"
selection = "OnePL_new"
preselection = "OneP_new"
from microfit import selections as sels
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

cat_1e1p = [2,21,31]

for run in RUN:
    rundata, mc_weights, data_pot = dl.load_runs(
    run,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=False,
    load_lee=False,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=True,
    load_crt_vars=False,
    enable_cache=False,
)
    
    sel_mc = rundata["mc"].query(query, engine='python')
    
    for i in cat_1e1p:
        # Choose the type of events you want to investigate. If not, set as :
        evts = ((sel_mc['interaction'] == 1) & (sel_mc['category_1e1p'] == i))
        filtered = sel_mc.loc[evts, ['run', 'sub', 'evt']]
        filtered.iloc[:50, :].to_csv(f"category1e1p_{i}_run{run}_rse.txt", sep=" ", index=None, header=None)