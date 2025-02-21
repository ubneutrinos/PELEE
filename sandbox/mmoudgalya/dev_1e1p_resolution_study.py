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

print('Loaded packages')

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial",
    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", "RecoECal", "Reco_mag_q", "RecoPL",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "trkshrhitdist2", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len", "reco_e",
    "RecoLeadProton_trk_len", "RecoLeadProton_trk_trunk_dEdx_y", "RecoLeadProton_dEdx_y_per_trklen",
    "RecoLeadProtonCandidate_trk_pid", "RecoElectronCandidate_shr_pid", "RecoElectron_conversion_dist",
    "pi0_radlen1", "pi0_radlen2", "pi0_score", "nonpi0_score", "bkg_score",
    "RecoElecE", "RecoElecModMom", "RecoElecMomX", "RecoElecMomY", "RecoElecMomZ",
    "RecoLeadProtonKE", "RecoLeadProtonModMom", "RecoLeadProtonMomX", "RecoLeadProtonMomY", "RecoLeadProtonMomZ",
    "ccnc",
]

#RUN = ["1","2","3"]
#RUN = ["1","2","3_nocrt","3_crt","4b","4c","4d","5"]
RUN = RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=True,
    load_lee=False,
    load_nue_tki=True,
    #keep_columns=keep_vars,
    blinded=True,
    load_crt_vars=False,
    enable_cache=True,
)

print('Loaded dataframes')

run_combo = "Run"
for run in RUN:
    run_combo += run

from microfit import selections as sel

selection = "OnePBDT"
preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])
is_sig = all_mc['category_1e1p'] == 12
all_sig = all_mc.loc[is_sig]
sel_sig = all_sig.query(query, engine='python')
    
variables = {

    '$\\delta p_T$ [GeV/c]': {'reco': 'RecoDeltaPT', 'truth': 'TrueDeltaPT', 'nbins': 20, 'bounds': (0, 2), 'range': [[0,1.75],[0,1.75]], 'bin_edges': [[0,0.3,1.75],[0,0.3,1.75]]},
    '$\\delta \\alpha_T$ [degrees]': {'reco': 'RecoDeltaAlphaT', 'truth': 'TrueDeltaAlphaT', 'nbins': 20, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,80,180],[0,80,180]]},
    '$p_n$ [GeV/c]': {'reco': 'RecoPN', 'truth': 'TruePN', 'nbins': 10, 'bounds': (0, 2), 'range': [[0,1.75],[0,1.75]], 'bin_edges': [[0,0.3,1.75],[0,0.3,1.75]]},
    '$\\alpha_{3D}$ [degrees]': {'reco': 'RecoAlpha3D', 'truth': 'TrueAlpha3D', 'nbins': 10, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,90,180],[0,90,180]]},
}

from matplotlib.colors import LogNorm

print('Making the 100-bin resolution plots')

for k, var in variables.items():
    #print(type(k))
    truth = variables[k]['truth']
    reco = variables[k]['reco']
    bounds = variables[k]['range']
    
    H, xedges, yedges = np.histogram2d(sel_sig[truth], sel_sig[reco], bins=100, weights=sel_sig["weights"], range=bounds)
    X, Y = np.meshgrid(xedges,yedges)
    plt.pcolormesh(X, Y, H.T, shading='flat') #, norm=LogNorm())
    plt.colorbar()
    plt.xlabel(f'True {k}')
    plt.ylabel(f'Reco {k}')
    
    #plt.title(f"True signal events with {sel.preselection_categories[preselection]['title']} and {sel.selection_categories[selection]['title']}")
    plt.savefig(f'plots/resolution_plots/resolution_2Dplots_{preselection}_{selection}_{truth}_{run_combo}_100bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/resolution_plots/resolution_2Dplots_{preselection}_{selection}_{truth}_{run_combo}_100bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
print('Making the 2-bin resolution plots')

for k, var in variables.items():
    #print(type(k))
    truth = variables[k]['truth']
    reco = variables[k]['reco']
    bounds = variables[k]['range']
    bin_edges = variables[k]['bin_edges']
    
    H, xedges, yedges = np.histogram2d(sel_sig[truth], sel_sig[reco], bins=bin_edges, weights=sel_sig["weights"], range=bounds)
    X, Y = np.meshgrid(xedges,yedges)
    plt.pcolormesh(X, Y, H.T, shading='flat') #, norm=LogNorm())
    plt.colorbar()
    plt.xlabel(f'True {k}')
    plt.ylabel(f'Reco {k}')
    
    #plt.title(f"True signal events with {sel.preselection_categories[preselection]['title']} and {sel.selection_categories[selection]['title']}")
    plt.savefig(f'plots/resolution_plots/resolution_2Dplots_{preselection}_{selection}_{truth}_{run_combo}_2bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/resolution_plots/resolution_2Dplots_{preselection}_{selection}_{truth}_{run_combo}_2bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
print('Done :)')