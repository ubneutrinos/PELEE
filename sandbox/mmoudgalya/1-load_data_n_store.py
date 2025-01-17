# run this script using python3LEE environment

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
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial",
    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", "RecoECal", "Reco_mag_q", "RecoPL",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len", "trk_llr_pid_score_v", "backtracked_pdg",
    "shr_llr_pid_score_v"
]

#RUN = ["3"]
RUN = ["1","2","3","4b","4c","4d","5"] #important that it's a string 1) new format to include latest runs 2) to include 'mc_pdg' otherwise it gets dropped

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=False,
#    load_lee=True,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=True,
    enable_cache=False,
)

# print(rundata.keys())
# print("nu_pdg" in rundata["mc"].columns)
# print("trk_llr_pid_score_v" in rundata["mc"].columns)
# print("backtracked_pdg" in rundata["mc"].columns)
# print("shr_llr_pid_score_v" in rundata["mc"].columns)
# print("Sel_1e1p" in rundata["mc"].columns)
#print(type(rundata["mc"]["trk_llr_pid_score_v"][0]))
print('Data POT:', data_pot)

SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]

for key, df in rundata.items():
    if key not in ['data', 'ext']:
        print(key)
        print(type(df))
        rundata[key] = df.drop(SYSTVARS, axis=1)
    if key != 'data':
        rundata[key].to_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}.pkl')
        
print('Done loading data, removing syst vars, and then storing as pickle files.')