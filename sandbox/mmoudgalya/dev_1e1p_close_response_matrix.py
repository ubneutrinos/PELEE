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

from microfit import detsys
from microfit.xsec_signal_generator import XsecCovarHistGenerator

from microfit import xsec_covariances as xs

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
]

#RUN = ["3"]
RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
blinded = True
data="bnb"

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data=data,
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=True,
    load_lee=False,
    load_numu_tki=True,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=blinded,
    load_crt_vars=False,
    enable_cache=True,
)

# Checking if the response matrix closes

selection = "OnePBDT"
preselection = "OneP_new"

from microfit import selections as sel
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
all_mc = pd.concat([df for k, df in rundata.items() if k!='data' or k!='ext'])
is_sig = all_mc['category_1e1p'] == 12
all_sig = all_mc.loc[is_sig]
sel_sig = all_sig.query(query, engine='python')
print("all_sig:", np.sum(all_sig["weights"]))

tot_sig = all_mc['category_1e1p'] == 12
tot_all_sig = all_mc.loc[tot_sig]
print('Total candidate signal events:', np.sum(all_mc.loc[tot_sig, 'weights']))

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")
    true_var_name = "True" + label

    signal_generator = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=[0.01,0.02]
        )
    
    mc_hists = signal_generator.get_mc_hists(
        category_column="category_1e1p",
    )

    runhist_reco_sig = mc_hists[12].bin_counts
    print("Runhist reco sig: \n", runhist_reco_sig)
    print()

    # reco_sig, reco_sig_edges = np.histogram(sel_sig[binning.variable], bins=binning.bin_edges, weights=sel_sig["weights"])
    # print("Reco sig: \n", reco_sig)
    # print()

    resp, x, y, n = xs.ResponseMatrix(all_mc, "category_1e1p==12", query, true_var_name, binning.variable, binning.bin_edges, "weights", univ=-1, wname="")
    print("Response Matrix: \n", resp)
    print()

    print("n: \n", n)
    print()

    full_sig, edges = np.histogram(all_sig[true_var_name], bins=binning.bin_edges, weights=all_sig["weights"])
    print("Full sig: \n", full_sig)
    print()

    result = resp.dot(full_sig)
    print("Result: \n", result)
    print()

    tot_full_sig, edges = np.histogram(tot_all_sig[binning.variable], bins=binning.bin_edges, weights=tot_all_sig["weights"])
    print("Total full sig: \n", tot_full_sig)
    print()

    tot_result = resp.dot(tot_full_sig)
    print("Total result: \n", tot_result)
    print()

    nresult = resp.dot(n)
    print("n Result: \n", nresult)
    print()

print("Done :)")