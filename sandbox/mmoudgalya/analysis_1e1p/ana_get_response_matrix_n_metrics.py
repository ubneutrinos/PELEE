# Make sure the local settings ntuple path points to the unfiltered ntuples and adjust data_loading.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../../")
import data_loading as dl
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons", "TrueNElec", "TrueNProt", 
    "TrueDeltaPT_1e1p", "TrueDeltaAlphaT_1e1p", "TruePN_1e1p", "TrueAlpha3D_1e1p",
    "TrueLeadProtonKE_1e1p", "TrueLeadProtonModMom_1e1p", "TrueLeadProtonE_1e1p", "TrueLeadProtonMomX_1e1p", "TrueLeadProtonMomY_1e1p", "TrueLeadProtonMomZ_1e1p",
    "TrueElecKE_1e1p", "TrueElecModMom_1e1p", "TrueElecE_1e1p", "TrueElecMomX_1e1p", "TrueElecMomY_1e1p", "TrueElecMomZ_1e1p",

    "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "mc_KE_prot", "mc_p_prot", "mc_E_prot", "mc_px_prot", "mc_py_prot", "mc_pz_prot",
    "mc_KE_elec", "mc_p_elec", "mc_E_elec", "mc_px_elec", "mc_py_elec", "mc_pz_elec",

    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT_1e1p", "RecoDeltaAlphaT_1e1p", "RecoPN_1e1p", "RecoAlpha3D_1e1p", #"RecoECal_1e1p", "Reco_mag_q_1e1p", "RecoPL_1e1p",
    "RecoLeadProtonKE_1e1p", "RecoLeadProtonModMom_1e1p", "RecoLeadProtonMomX_1e1p", "RecoLeadProtonMomY_1e1p", "RecoLeadProtonMomZ_1e1p",
    "RecoElecE", "RecoElecModMom", "RecoElecMomX", "RecoElecMomY", "RecoElecMomZ",

    "RecoLeadProton_trk_len", "RecoLeadProton_trk_trunk_dEdx_y", "RecoLeadProton_dEdx_y_per_trklen",
    "RecoLeadProtonCandidate_trk_pid", "RecoElectronCandidate_shr_pid", "RecoElectron_conversion_dist",

    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial", "elec_e", "proton_ke",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "trkshrhitdist2", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len", "reco_e",
    "trkpid", "trk_len", "n_showers_contained", "protonenergy_corr", "n_tracks_contained",
    "pi0_radlen1", "pi0_radlen2", "pi0_score", "nonpi0_score", "bkg_score", "trk_id",

    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", #"RecoECal", "Reco_mag_q", "RecoPL",
    "RecoLeadProtonKE", "RecoLeadProtonModMom", "RecoLeadProtonMomX", "RecoLeadProtonMomY", "RecoLeadProtonMomZ",

    # "InFV_1muNp", "TrueMuonIdx_1muNp", "TrueLeadProtonIdx_1muNp", "TrueNProt_1muNp", "TrueFSPions_1muNp", "Signal_1mu1p", 
    # "TrueDeltaPT_1mu1p", "TrueDeltaAlphaT_1mu1p", "TruePN_1mu1p", "TrueAlpha3D_1mu1p",
    # "TrueLeadProtonE_1muNp", "TrueLeadProtonMomX_1muNp", "TrueLeadProtonMomY_1muNp", "TrueLeadProtonMomZ_1muNp",
    # "TrueMuonE_1muNp", "TrueMuonMomX_1muNp", "TrueMuonMomY_1muNp", "TrueMuonMomZ_1muNp",

    # "sel_CC1p0pi", "InFV_reco_1muNp", "MuonCandidateIdx_1muNp", "LeadProtonIdx_1muNp", "LeadProtonPassMomentumCut_1muNp", "PFPStartsInPCV_1muNp", "PassTopoScoreCut_1muNp",
    # "PassNuMuCCSelection_1muNp", "NoRecoShowers_1muNp", "MuonContained_1muNp", "PassMuonMomentumCut_1muNp", "PassMuonQualCut_1muNp",
    # "LeadProtonPassMomentumCut_1muNp", "NProtons_1muNp",
    # "RecoDeltaPT_1mu1p", "RecoDeltaAlphaT_1mu1p", "RecoPN_1mu1p", "RecoAlpha3D_1mu1p", "RecoECal_1mu1p", "RecoPL_1mu1p",
    # "RecoLeadProtonE_1muNp", "RecoLeadProtonMomentum_1muNp", "RecoLeadProtonMomX_1muNp", "RecoLeadProtonMomY_1muNp", "RecoLeadProtonMomZ_1muNp", 
    # "RecoMuonE_1muNp", "RecoMuonMomentum_1muNp", "RecoMuonMomX_1muNp", "RecoMuonMomY_1muNp", "RecoMuonMomZ_1muNp",
]

#RUN = ["3"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"] # use this if using CRT
#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"] # for detvars with bnb or for closure test
RUN = ["1"] # for nuwro_fd
data="nuwro_fd"
#data="bnb"
ingredients = False

# selection = "OnePBDT"
# preselection = "OneP_new"
# category_column="category_1e1p"
# sig_code = 12

selection = "OneP_NPBDTXS"
preselection = "NUE"
category_column="category_1e1p_tki"
sig_code = 12

signal_query = category_column + f" == {sig_code}"

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data=data,
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=True,
    load_lee=False,
    load_numu_tki=False,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=True,
    load_crt_vars=False,
    enable_cache=True,
)

print('Loaded data')

run_combo = "Run"
for run in RUN:
    run_combo += run

        
# Calculating the response matrix:

if data == "nuwro_fd":
    backup_dfs = {}
    for k, df in rundata.items():
        if k in ["drt", "ext"]:
            backup_dfs[k] = df
            rundata[k] = None

###########################################################################################
# If we want to filter the n-tuples for certain studies
filtered_rundata = {}  
for key, df in rundata.items():
    print(key)
    if rundata[key] is None:
        print(key)
        filtered_rundata[key] = None
    elif key in ["data"] and blinded:
        filtered_rundata["data"] = None
    else:
        filtered_rundata[key] = df.query(signal_query, engine='python')

rundata = {}
rundata = filtered_rundata.copy()
###########################################################################################

from microfit import selections as sel
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

all_mc = pd.concat([df for k, df in rundata.items() if k not in ['data','ext']])
# is_sig = all_mc[category_column] == sig_code
# all_sig = all_mc.loc[is_sig]
all_sig = all_mc.query(signal_query, engine='python')
sel_sig = all_sig.query(query, engine='python')

ingredient_vars = {
    # My selection
    # 'Proton Kinetic Energy [GeV]': {'reco': 'RecoLeadProtonKE_1e1p', 'truth': 'TrueLeadProtonKE_1e1p', 'nbins': 10, 'bounds': (0, 1), 'range': [[0, 1],[0, 1]]},
    # 'Proton Momentum [GeV/c]': {'reco': 'RecoLeadProtonModMom_1e1p', 'truth': 'TrueLeadProtonModMom_1e1p', 'nbins': 20, 'bounds': (0, 1.5), 'range': [[0, 1.5],[0, 1.5]]},
    # 'Proton X Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomX_1e1p', 'truth': 'TrueLeadProtonMomX_1e1p', 'nbins': 20, 'bounds': (-1, 1), 'range': [[-1, 1],[-1, 1]]},
    # 'Proton Y Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomY_1e1p', 'truth': 'TrueLeadProtonMomY_1e1p', 'nbins': 20, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    # 'Proton Z Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomZ_1e1p', 'truth': 'TrueLeadProtonMomZ_1e1p', 'nbins': 20, 'bounds': (-1, 1.5),'range': [[-1, 1.5],[-1, 1.5]]},
    # 'Electron Energy [GeV]': {'reco': 'RecoElecE', 'truth': 'TrueElecE_1e1p', 'nbins': 16, 'bounds': (0, 4), 'range': [[0, 4],[0, 4]]},
    # 'Electron Momentum [GeV/c]': {'reco': 'RecoElecModMom', 'truth': 'TrueElecModMom_1e1p', 'nbins': 20, 'bounds': (0, 5), 'range': [[0, 5],[0, 5]]},
    # 'Electron X Momentum [GeV/c]': {'reco': 'RecoElecMomX', 'truth': 'TrueElecMomX_1e1p', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    # 'Electron Y Momentum [GeV/c]': {'reco': 'RecoElecMomY', 'truth': 'TrueElecMomY_1e1p', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    # 'Electron Z Momentum [GeV/c]': {'reco': 'RecoElecMomZ', 'truth': 'TrueElecMomZ_1e1p', 'nbins': 12, 'bounds': (-1, 5), 'range': [[-1, 5],[-1, 5]]},

    # Lucile's selection
    'Proton Kinetic Energy [GeV]': {'reco': 'RecoLeadProtonKE', 'truth': 'mc_KE_prot', 'nbins': 10, 'bounds': (0, 1), 'range': [[0, 1],[0, 1]]},
    'Proton Momentum [GeV/c]': {'reco': 'RecoLeadProtonModMom', 'truth': 'mc_p_prot', 'nbins': 20, 'bounds': (0, 1.5), 'range': [[0, 1.5],[0, 1.5]]},
    'Proton X Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomX', 'truth': 'mc_px_prot', 'nbins': 20, 'bounds': (-1, 1), 'range': [[-1, 1],[-1, 1]]},
    'Proton Y Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomY', 'truth': 'mc_py_prot', 'nbins': 20, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    'Proton Z Momentum [GeV/c]': {'reco': 'RecoLeadProtonMomZ', 'truth': 'mc_pz_prot', 'nbins': 20, 'bounds': (-1, 1.5),'range': [[-1, 1.5],[-1, 1.5]]},
    'Electron Energy [GeV]': {'reco': 'RecoElecE', 'truth': 'mc_E_elec', 'nbins': 16, 'bounds': (0, 4), 'range': [[0, 4],[0, 4]]},
    'Electron Momentum [GeV/c]': {'reco': 'RecoElecModMom', 'truth': 'mc_p_elec', 'nbins': 20, 'bounds': (0, 5), 'range': [[0, 5],[0, 5]]},
    'Electron X Momentum [GeV/c]': {'reco': 'RecoElecMomX', 'truth': 'mc_px_elec', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    'Electron Y Momentum [GeV/c]': {'reco': 'RecoElecMomY', 'truth': 'mc_py_elec', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1.5, 1.5],[-1.5, 1.5]]},
    'Electron Z Momentum [GeV/c]': {'reco': 'RecoElecMomZ', 'truth': 'mc_pz_elec', 'nbins': 12, 'bounds': (-1, 5), 'range': [[-1, 5],[-1, 5]]},
}

variables = {
# #    '': {'reco': , 'truth': , 'nbins': , 'bounds': },
    # My selection
    # '$\\delta p_T$ [GeV/c]': {'reco': 'RecoDeltaPT_1e1p', 'truth': 'TrueDeltaPT_1e1p', 'nbins': 20, 'bounds': (0, 2), 'range': [[0,1.7],[0,1.7]], 'bin_edges': [[0,0.3,1.7],[0,0.3,1.7]], 'bin_edges_1d': [0,0.3,1.7]},
    # '$\\delta \\alpha_T$ [degrees]': {'reco': 'RecoDeltaAlphaT_1e1p', 'truth': 'TrueDeltaAlphaT_1e1p', 'nbins': 20, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,80,180],[0,80,180]], 'bin_edges_1d': [0,80,180]},
    # '$p_n$ [GeV/c]': {'reco': 'RecoPN_1e1p', 'truth': 'TruePN_1e1p', 'nbins': 10, 'bounds': (0, 2), 'range': [[0,1.7],[0,1.7]], 'bin_edges': [[0,0.3,1.7],[0,0.3,1.7]], 'bin_edges_1d': [0,0.3,1.7]},
    # '$\\alpha_{3D}$ [degrees]': {'reco': 'RecoAlpha3D_1e1p', 'truth': 'TrueAlpha3D_1e1p', 'nbins': 10, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,90,180],[0,90,180]], 'bin_edges_1d': [0,90,180]},

    # Lucile's selection
    '$\\delta p_T$ [GeV/c]': {'reco': 'RecoDeltaPT', 'truth': 'TrueDeltaPT', 'nbins': 20, 'bounds': (0, 2), 'range': [[0,1.7],[0,1.7]], 'bin_edges': [[0,0.3,1.7],[0,0.3,1.7]], 'bin_edges_1d': [0,0.3,1.7]},
    '$\\delta \\alpha_T$ [degrees]': {'reco': 'RecoDeltaAlphaT', 'truth': 'TrueDeltaAlphaT', 'nbins': 20, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,80,180],[0,80,180]], 'bin_edges_1d': [0,80,180]},
    '$p_n$ [GeV/c]': {'reco': 'RecoPN', 'truth': 'TruePN', 'nbins': 10, 'bounds': (0, 2), 'range': [[0,1.7],[0,1.7]], 'bin_edges': [[0,0.3,1.7],[0,0.3,1.7]], 'bin_edges_1d': [0,0.3,1.7]},
    '$\\alpha_{3D}$ [degrees]': {'reco': 'RecoAlpha3D', 'truth': 'TrueAlpha3D', 'nbins': 10, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,90,180],[0,90,180]], 'bin_edges_1d': [0,90,180]},
}

with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
    f.write("\nResponse Matrices:\n")

for k, var in variables.items():
    truth = variables[k]['truth']
    reco = variables[k]['reco']
    bounds = variables[k]['range']
    bin_edges = variables[k]['bin_edges']
    bin_edges_1d = variables[k]['bin_edges_1d']
    
    truth_hist, truth_edges = np.histogram(all_sig[truth], bins=bin_edges_1d, weights=all_sig["weights"], range=bounds)
    H, xedges, yedges = np.histogram2d(sel_sig[truth], sel_sig[reco], bins=bin_edges, weights=sel_sig["weights"], range=bounds)
    
    resp = H.T / truth_hist
    
    X, Y = np.meshgrid(xedges,yedges)
    plt.pcolormesh(X, Y, resp, shading='flat') #, norm=LogNorm())
    plt.colorbar()
    plt.xlabel(f'True {k}')
    plt.ylabel(f'Reco {k}')
    
    plt.title(f"Response Matrix")
    label = reco.lstrip("Reco")
    plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{label}_{data}_{preselection}_{selection}_{run_combo}_2bins_recovery.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{label}_{data}_{preselection}_{selection}_{run_combo}_2bins_recovery.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    print('Response matrix for', k, ': \n', resp)
    print()
    # print(type(resp))
    # writing these to a file
    resp_str = "{"
    for i in range(resp.shape[0]):
        for j in range(resp.shape[1]):
            resp_str += f"{resp[i][j]},"
    resp_str = resp_str[:-1] # to remove the last comma
    resp_str += "};"
    print(label)
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.writelines([f"{label}", " = ", f"{resp_str} \n"])

if ingredients:
    for k, var in ingredient_vars.items():
        truth = ingredient_vars[k]['truth']
        reco = ingredient_vars[k]['reco']
        bounds = ingredient_vars[k]['bounds']
        range = ingredient_vars[k]['range']
        bin_edges = ingredient_vars[k]['nbins']
        bin_edges_1d = ingredient_vars[k]['nbins']
        
        truth_hist, truth_edges = np.histogram(all_sig[truth], bins=bin_edges_1d, weights=all_sig["weights"], range=bounds)
        H, xedges, yedges = np.histogram2d(sel_sig[truth], sel_sig[reco], bins=bin_edges, weights=sel_sig["weights"], range=range)
        
        resp = H.T / truth_hist
        
        X, Y = np.meshgrid(xedges,yedges)
        plt.pcolormesh(X, Y, resp, shading='flat') #, norm=LogNorm())
        plt.colorbar()
        plt.xlabel(f'True {k}')
        plt.ylabel(f'Reco {k}')
        
        plt.title(f"Response Matrix")
        label = reco.lstrip("Reco")
        plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{label}_{data}_{preselection}_{selection}_{run_combo}_recovery.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{label}_{data}_{preselection}_{selection}_{run_combo}_recovery.png', bbox_inches='tight')
        plt.show()
        plt.clf()

print('Finished calculating response matrices.')

# Calculating the metrics

print("Calculating metrics:")
print()

# all_sig = all_mc[category_column] == sig_code
# tot_all_sig = np.sum(all_mc.loc[all_sig, 'weights'])
tot_all_sig = np.sum(all_sig['weights'])
print('Total candidate signal events:', tot_all_sig)

all_predict = all_mc.query(query, engine='python')

is_sig = all_predict[category_column] == sig_code
tot_sig = np.sum(all_predict.loc[is_sig, 'weights'])
tot_bkg = np.sum(all_predict.loc[~is_sig, 'weights'])
tot_evt = np.sum(all_predict['weights'])
print('After cuts:')
print('Total signal events:', tot_sig)
print('Total background events:', tot_bkg)
print('Total events:', tot_evt)
print('sig + bkg =', tot_sig+tot_bkg)
print()

efficiency = (tot_sig / tot_all_sig) * 100
purity = (tot_sig / tot_evt) * 100
print('Efficiency:', efficiency)
print('Purity:', purity)
print()

pred_res = all_predict['interaction'] == 1
tot_res = np.sum(all_predict.loc[pred_res, 'weights'])
print('Total resonance events:', tot_res)
pred_qe = all_predict['interaction'] == 0
tot_qe = np.sum(all_predict.loc[pred_qe, 'weights'])
print('Total quasielastic events:', tot_qe)

print('QE/RES =', tot_qe/tot_res)
print('RES/QE =', tot_res/tot_qe)
print('QE/Total =', tot_qe/tot_evt)

with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
    f.write("\nMetrics:\n")
    f.writelines(['\nTotal candidate signal events = ', f"{tot_all_sig} \n"])
    f.writelines(["After cuts: \n"])
    f.writelines(['Total signal events = ', f"{tot_sig} \n"])
    f.writelines(['Total background events = ', f"{tot_bkg} \n"])
    f.writelines(['Total events = ', f"{tot_evt} \n"])
    f.writelines(['\nPurity = ', f"{purity} % \n"])
    f.writelines(['Efficiency = ', f"{efficiency} % \n"])
    f.writelines(['\nQE/RES = ', f"{tot_qe/tot_res} \n"])
    f.writelines(['RES/QE = ', f"{tot_res/tot_qe} \n"])
    f.writelines(['QE/Total = ', f"{tot_qe/tot_evt} \n"])

f.close()
print('Done :)')