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

from microfit import detsys
from microfit import xsec_covariances as xs
from microfit.xsec_signal_generator import XsecCovarHistGenerator

def repeated_nom_values(hist):
    # repeat the last bin count
    y = hist.bin_counts
    y = np.append(y, y[-1])
    return y

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

keep_vars_detsys = keep_vars + ["ccnc", "nu_pdg",]

RUN = ["1"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"] # use this if using CRT
#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"] # for detvars with bnb or for closure test
#RUN = ["1","2","3","4a","4c","5"] # for nuwro_fd, no run 4b and 4d available
blinded = False
data="nuwro_fd"
#data="bnb"

# Choose the selection cuts

# selection = "OnePBDT"
# preselection = "OneP_new"
# category_column="category_1e1p"
# sig_code = 12
# signal_query = category_column + f" == {sig_code}"

# selection = "None"
selection = "OneP_NPBDTXS"
preselection = "NUE"
category_column="category_1e1p_tki"
sig_code = 12
# # category_column="category_fixed"
# # sig_code = 11
# ACCEPTANCE = 'isVtxInFiducial == 1 and ccnc==0 and nu_pdg==12 and npi0==0 and npion==0 and mc_E_elec>0.03051' #elec_e>0.03051'
# ACCEPTANCE += ' and mc_KE_prot>0.05' #' and proton_ke>0.05'
# ACCEPTANCE += ' and nproton == 1'
# #signal_query = ACCEPTANCE
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
    blinded=blinded,
    load_crt_vars=False,
    enable_cache=True,
)

print('Loaded data')
print()

if data == "nuwro_fd":
    backup_dfs = {}
    for k, df in rundata.items():
        if k not in ['nue', 'data']:
            backup_dfs[k] = df
            rundata[k] = None

run_combo = "Run"
for run in RUN:
    run_combo += run

from microfit import selections as sel
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

all_mc = pd.concat([df for k, df in rundata.items() if k not in ['data','ext']])
all_sig = all_mc.query(signal_query, engine='python')
print("length of all_sig is", len(all_sig))
print("length of all_mc is", len(all_mc))
sel_sig = all_sig.query(query, engine='python')

# for k, df in rundata.items():
#     print(f"{k}: {len(rundata[k])}")

for binning_def in vdef.TKI_variables_1e1p:
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")

    # Getting uB tune signal bin counts
    #true_var_name = "mc_KE_prot"
    true_var_name = "True" + label
    print(true_var_name)
    genieUBsig, _ = np.histogram(all_sig[true_var_name], bins=binning.bin_edges, weights=all_sig["weights"])
    #genieUBsig, _ = np.histogram(all_sig[true_var_name], bins=20, weights=all_sig["weights"])

    rm, _, _ = xs.ResponseMatrix(all_sig, signal_query, query, true_var_name, binning_def[0], binning.bin_edges, "weights")
    result = np.array(rm).dot(genieUBsig)
    sig, _ = np.histogram(sel_sig[binning_def[0]], bins=binning.bin_edges, weights=sel_sig["weights"])

    print("Truth level GENIE: \n", genieUBsig)
    print()
    print("Sum of truth level GENIE: ", sum(genieUBsig))
    print()
    print("Response Matrix: \n", rm)
    print()
    print("Result: \n", result)
    print()
    print("Selected sig: \n", sig)
    print()
    print("Difference: \n", sig - result)
    print()

# Calculating the response matrix:

# from microfit import selections as sel
# query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

# all_mc = pd.concat([df for k, df in rundata.items() if k!='data' or k!='ext'])
# is_sig = all_mc[category_column] == sig_code
# all_sig = all_mc.loc[is_sig]
# sel_sig = all_sig.query(query, engine='python')

# Calculating the metrics

print("Calculating metrics:")
print()


#all_sig = all_mc[category_column] == sig_code
#tot_all_sig = np.sum(all_mc.loc[all_sig, 'weights'])
tot_all_sig = np.sum(all_sig['weights'])
print('Total candidate signal events:', tot_all_sig)
#print(len(all_mc.loc[all_sig, 'weights']))

all_predict = all_mc.query(query, engine='python')
sel_bkg = all_predict.query(f"~({signal_query})", engine='python')

is_sig = all_predict[category_column] == sig_code
tot_sig = np.sum(all_predict.loc[is_sig, 'weights'])
tot_bkg = np.sum(all_predict.loc[~is_sig, 'weights'])
tot_evt = np.sum(all_predict['weights'])

# tot_sig = np.sum(sel_sig['weights'])
# tot_bkg = np.sum(sel_bkg['weights'])
print('After cuts:')
print('Total signal events:', tot_sig)
print('Total background events:', tot_bkg)
print('Total events:', tot_evt)
print('sig + bkg =', tot_sig+tot_bkg)
print()


Nominal_UB_XY_Surface = 256.35*233. # cm2
SoftFidSurface = 236. * 210.  # cm2
POTPerSpill = 4997.*5e8
HistoFlux_int = 593641.00 # retrieved from the root file itself by running hEnue_cv->Integral()
#IntegratedFlux = (HistoFlux_int * data_pot / POTPerSpill / Nominal_UB_XY_Surface)
#datapot = 2.81e22
datapot = 1.756e21
IntegratedFlux = (HistoFlux_int * datapot / POTPerSpill / Nominal_UB_XY_Surface)
print('Integrated flux:', IntegratedFlux)