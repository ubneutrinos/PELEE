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
from microfit import detsys
from microfit import variable_definitions as vdef
from microfit import selections
from microfit.xsec_signal_generator import XsecCovarHistGenerator

print('Loaded packages')

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

    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial", 
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

#RUN = ["4b"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"]
#RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
blinded = True
use_detvar = False

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
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

print('Loaded dataframes')

run_combo = "Run"
for run in RUN:
    run_combo += run
    
# selections = ["LucileSEL"]
#selections = ["OnePBDTFarSB", "OnePBDTNearSBpi0", "OnePBDTNearSB0p"]
# selections = ["None"]
# preselection = "NUE"

# selections = ["OnePBDT"]
# preselection = "OneP_new"
    
selections = ["OneP_NPBDTXS"]
preselection = "NUE"

all_mc = pd.concat([df for k, df in rundata.items() if k!='data' or k!='ext'])
all_sig = all_mc.query("category_1e1p_tki == 12", engine='python')
#all_sig = all_mc.query("category_1e1p == 12", engine='python')

for selection in selections:
    for binning_def in vdef.TKI_variables_1e1p:
        # some binning definitions have more than 4 elements,
        # we ignore the last ones for now
        # binning = hist.Binning.from_config(*binning_def[:4])
        binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
        print(binning_def)

        if use_detvar:
            detvar_data = detsys.make_variations(
                run_numbers=RUN,
                data="bnb",
                binning=binning.copy(),
                selection=selection,
                preselection=preselection,
                use_kde_smoothing=False,
                make_plots=False,
                plot_output_dir= "/exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya/analysis_1e1p/analysis_plots/detsys/",
                enable_detvar_cache=True,
                detvar_cache_dir="/exp/uboone/data/users/mmoudgal/PELEE/detvar_cached_dataframes/",
                extra_selection_query=None,
                show_plots=False,
                loadpi0variables=False,
                loadshowervariables=True,
                loadrecoveryvars=False,
                loadsystematics=True,
                numupresel=False,
                loadnumuvariables=False,
                use_bdt=True,
                load_lee=False,
                #load_numu_tki=True,
                load_nue_tki=True,
                keep_columns=keep_vars_detsys,
                blinded=True,
                load_crt_vars=False,
                enable_cache=True,
                )
        else:
            detvar_data = None

        # Total error
        signal_generator = hist.RunHistGenerator(
            rundata,
            binning.copy(),
            data_pot=data_pot,
            selection=selection,
            preselection=preselection,
            sideband_generator=None,
            uncertainty_defaults=None,
            detvar_data=detvar_data,
            mc_hist_generator_cls = XsecCovarHistGenerator,
            true_var_name=None, 
            signal_query="category_1e1p_tki == 12", 
            uncut_signal_df=rundata["nue"],
            normalization_uncertainty=[0.01,0.02]
        )

        plotter = rp.RunHistPlotter(signal_generator)
        axes = plotter.plot(
            category_column="category_1e1p_tki",
            signal_category_num=12,
            include_multisim_errors=True,
            add_ext_error_floor=False,
            show_data_mc_ratio=False,
            show_chi_square=False,
            show_total_unconstrained=False,
            add_precomputed_detsys=use_detvar,
            show_errorband=True,
        )
        
    #     plt.yscale('log')
        # ax = axes[0]
        # ax.set_ylim(0.1, ax.get_ylim()[1] * 1.5)
        # ax.set_yscale('log')
        plt.savefig(f'plots/reco_study/bdt_scores/topo_nodetvar_all_{preselection}_{selection}_{binning_def[0]}_{run_combo}_recovery.pdf', bbox_inches='tight')
        plt.savefig(f'plots/reco_study/bdt_scores/topo_nodetvar_all_{preselection}_{selection}_{binning_def[0]}_{run_combo}_recovery.png', bbox_inches='tight')

        # axes2 = plotter.plot(
        #     category_column="category_1e1p",
        #     signal_category_num=12,
        #     include_multisim_errors=True,
        #     add_ext_error_floor=False,
        #     show_data_mc_ratio=False,
        #     show_chi_square=False,
        #     show_total_unconstrained=False,
        #     add_precomputed_detsys=use_detvar,
        #     show_errorband=True,
        # )
        
        # # ax2 = axes2[0]
        # # ax2.set_ylim(0.1, ax2.get_ylim()[1] * 1.5)
        # # ax2.set_yscale('log')
        # plt.savefig(f'plots/reco_study/bdt_scores/topo_1e1p_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
        # plt.savefig(f'plots/reco_study/bdt_scores/topo_1e1p_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
        # plt.show()

        # Filtered dataframes
        filtered_rundata = {}
        
        for key, df in rundata.items():
            print(key)
            if key in ["data"] and blinded:
                filtered_rundata["data"] = None
            else:
                filtered_rundata[key] = df.query("~(abs(nu_pdg)==12 and ccnc == 0)", engine='python')

        signal_generator_filtered = hist.RunHistGenerator(
            filtered_rundata,
            binning.copy(),
            data_pot=data_pot,
            selection=selection,
            preselection=preselection,
            sideband_generator=None,
            uncertainty_defaults=None,
            detvar_data=detvar_data,
            mc_hist_generator_cls = XsecCovarHistGenerator,
            true_var_name=None, 
            signal_query="category_1e1p_tki == 12", 
            uncut_signal_df=filtered_rundata["nue"],
            normalization_uncertainty=[0.01,0.02]
        )

        plotter = rp.RunHistPlotter(signal_generator_filtered)
        axes3 = plotter.plot(
            category_column="category_1e1p_tki",
            # signal_category_num=12,
            include_multisim_errors=True,
            add_ext_error_floor=False,
            show_data_mc_ratio=False,
            show_chi_square=False,
            show_total_unconstrained=False,
            add_precomputed_detsys=use_detvar,
            show_errorband=True,
        )
        
    #     plt.yscale('log')
        # ax = axes[0]
        # ax.set_ylim(0.1, ax.get_ylim()[1] * 1.5)
        # ax.set_yscale('log')
        plt.savefig(f'plots/reco_study/bdt_scores/topo_nodetvar_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_recovery.pdf', bbox_inches='tight')
        plt.savefig(f'plots/reco_study/bdt_scores/topo_nodetvar_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_recovery.png', bbox_inches='tight')
        


    #     #################################################################################
    #     # Getting bin counts
        
    #     label = binning_def[0].lstrip("Reco")
    #     if binning_def[0] in [ "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D"]:
    #         genieUBsig, _ = np.histogram(all_sig[binning_def[0]], bins=binning_def[-1], weights=all_sig["weights"])
        
    #     total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True, add_precomputed_detsys=False)
    #     total_pred_counts = total_prediction.bin_counts
        
    #     mc_hists = signal_generator.get_mc_hists(
    #         category_column="category_1e1p",
    # #         include_multisim_errors=True,
    # #         add_precomputed_detsys=True,
    #     )
    #     mc_sig = mc_hists[12].bin_counts
        
    #     total_bkg = total_pred_counts - mc_sig
        
    #     bkg_mc_counts = {k: v.bin_counts for k, v in mc_hists.items() if k != 12}
    #     bkg_mc_sum = [sum(items) for items in zip(*bkg_mc_counts.values())]
        
    #     ext = total_bkg - bkg_mc_sum
        
    #     with open(f'plots/reco_study/bdt_scores/bincounts_{preselection}_{selection}_{run_combo}.txt', 'a') as f:
    #         f.write(f"\n{label}:\n")
    #         if binning_def[0] in [ "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D"]:
    #             f.writelines([f"genieUBsig = ", f"{genieUBsig} \n"])
    #         for k, h in mc_hists.items():
    #             f.writelines([f"{k}: ", f"{h.bin_counts} \n"])
            
    #         f.writelines([f"EXT", " = ", f"{ext} \n"])
    # #         f.writelines([f"1e1p", " = ", f"{mc_sig} \n"])
    #         f.writelines([f"total bkg", " = ", f"{total_bkg} \n"])

    # f.close()

print("Calculating metrics:")
print()

#all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])

all_sig = all_mc['category_1e1p_tki'] == 12
#all_sig = all_mc['category_1e1p'] == 12
tot_all_sig = np.sum(all_mc.loc[all_sig, 'weights'])
print('Total candidate signal events:', tot_all_sig)

from microfit import selections as sel

# selection = "OnePL_new"
# preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
#print(query)

all_predict = all_mc.query(query, engine='python')

is_sig = all_predict['category_1e1p_tki'] == 12
#is_sig = all_predict['category_1e1p'] == 12
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