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
from microfit.xsec_signal_generator import XsecCovarHistGenerator

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

#RUN = ["5"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"]
#RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
#RUN = ["3","4a","4b","4c","4d","5"]
blinded = True

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
    keep_columns=keep_vars,
    blinded=True,
    load_crt_vars=False,
    enable_cache=True,
)

print('Loaded dataframes')

run_combo = "Run"
for run in RUN:
    run_combo += run
    
selection = "OnePBDT"
#selection = "None"
preselection = "OneP_new"

all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])
all_sig = all_mc.query("category_1e1p == 12", engine='python')

for binning_def in vdef.variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning,
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
    )
    plotter = rp.RunHistPlotter(signal_generator)
    axes = plotter.plot(
        category_column="category_1e1p",
        signal_category_num=12,
        include_multisim_errors=False,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )
    
#     plt.yscale('log')
    plt.savefig(f'plots/reco_study/bdt_scores/topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
    plt.savefig(f'plots/reco_study/bdt_scores/topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=False,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )
    
#     axes[0].set_ylim([0.1, 100000])
#     axes[0].set_yscale('log')
    plt.savefig(f'plots/reco_study/bdt_scores/int_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
    plt.savefig(f'plots/reco_study/bdt_scores/int_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
    plt.show()
    
    #################################################################################
    # Getting bin counts
    
    label = binning_def[0].lstrip("Reco")
    if binning_def[0] in [ "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D"]:
        genieUBsig, _ = np.histogram(all_sig[binning_def[0]], bins=binning_def[-1], weights=all_sig["weights"])
    
    total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True, add_precomputed_detsys=False)
    total_pred_counts = total_prediction.bin_counts
    
    mc_hists = signal_generator.get_mc_hists(
        category_column="category_1e1p",
#         include_multisim_errors=True,
#         add_precomputed_detsys=True,
    )
    mc_sig = mc_hists[12].bin_counts
    
    total_bkg = total_pred_counts - mc_sig
    
    bkg_mc_counts = {k: v.bin_counts for k, v in mc_hists.items() if k != 12}
    bkg_mc_sum = [sum(items) for items in zip(*bkg_mc_counts.values())]
    
    ext = total_bkg - bkg_mc_sum
    
    with open(f'plots/reco_study/bdt_scores/bincounts_{preselection}_{selection}_{run_combo}.txt', 'a') as f:
        f.write(f"\n{label}:\n")
        if binning_def[0] in [ "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D"]:
            f.writelines([f"genieUBsig = ", f"{genieUBsig} \n"])
        for k, h in mc_hists.items():
            f.writelines([f"{k}: ", f"{h.bin_counts} \n"])
        
        f.writelines([f"EXT", " = ", f"{ext} \n"])
#         f.writelines([f"1e1p", " = ", f"{mc_sig} \n"])
        f.writelines([f"total bkg", " = ", f"{total_bkg} \n"])

f.close()
print("Calculating metrics:")
print()

#all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])

all_sig = all_mc['category_1e1p'] == 12
tot_all_sig = np.sum(all_mc.loc[all_sig, 'weights'])
print('Total candidate signal events:', tot_all_sig)

from microfit import selections as sel

# selection = "OnePL_new"
# preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
#print(query)

all_predict = all_mc.query(query, engine='python')

is_sig = all_predict['category_1e1p'] == 12
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

with open(f'plots/reco_study/bdt_scores/metrics_{preselection}_{selection}_{run_combo}.txt', 'a') as f:
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

print('Done :)')