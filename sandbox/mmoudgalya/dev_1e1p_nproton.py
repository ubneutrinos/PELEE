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
    load_nue_tki=False,
    keep_columns=None,
    blinded=True,
    load_crt_vars=False,
    enable_cache=True,
)

print('Loaded dataframes')

run_combo = "Run"
for run in RUN:
    run_combo += run
    

all_mc = pd.concat([df for k, df in rundata.items() if k!='data' or k!='ext'])
plt.hist(all_mc['mc_p_prot'], bins= 30, range=(0, 3), histtype='step', label='All MC (nue presel cuts only)', lw=3)
#all_mc.query('nproton >= 1', inplace=True)
sel_mc = all_mc.query('nproton > 0', inplace=False)
plt.hist(sel_mc['mc_p_prot'], bins= 30, range=(0, 3), histtype='step', label='nproton >= 1')
plt.legend()
#plt.xlim([0,2.5])
#plt.grid(visible=True, which='both', axis='x')
plt.xlabel('True proton momentum (GeV/c)')
plt.ylabel('Unweighted Events')
plt.savefig(f'plots/reco_study/nproton_{run_combo}.pdf', bbox_inches='tight')
plt.savefig(f'plots/reco_study/nproton_{run_combo}.png', bbox_inches='tight')

selection = "OneP_NPBDTXS"
preselection = "NUE"

from microfit import selections as sel
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
sel_evts = all_mc.query(query, engine='python')
# anomaly = ((sel_evts["category_1e1p_tki"] == 12) & (sel_evts["nproton"] == 3))
# sel_evts.loc[anomaly, ("nu_pdg", "mc_pdg", )]
anomalies = sel_evts.query('category_1e1p_tki == 12 and nproton == 3', engine='python')
#header = ["category_1e1p_tki", "nproton", "nu_pdg", "mc_pdg", "mc_KE_prot","RecoLeadProtonKE", "mc_E_prot", "RecoLeadProtonE", "mc_p_prot", "RecoLeadProtonModMom"]
header = ["category_1e1p_tki", "nproton", "nu_pdg", "mc_pdg", "mc_E", "mc_E_prot", "mc_KE_prot", "mc_p_prot"]
for col in header:
    print(f"{col}: ", "selected" in anomalies.columns)
subset = anomalies.loc[:, header]
subset.to_csv(f"plots/reco_study/nproton_anomalies.csv", sep="\t", index=None, header=header)
#anomalies.to_csv('plots/reco_study/nproton_anomalies.csv', columns = header)
#print(anomalies[header])

for binning_def in vdef.sel_variables_1e1p:
        # some binning definitions have more than 4 elements,
        # we ignore the last ones for now
        # binning = hist.Binning.from_config(*binning_def[:4])
        binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
        print(binning_def)

        # Total error
        signal_generator = hist.RunHistGenerator(
            rundata,
            binning.copy(),
            data_pot=data_pot,
            selection=selection,
            preselection=preselection,
            sideband_generator=None,
            uncertainty_defaults=None,
            detvar_data=None,
            # mc_hist_generator_cls = XsecCovarHistGenerator,
            # true_var_name=None, 
            # signal_query="category_1e1p == 12", 
            # uncut_signal_df=rundata["nue"],
            # normalization_uncertainty=[0.01,0.02]
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
            add_precomputed_detsys=False,
            show_errorband=True,
        )
        
    #     plt.yscale('log')
        # ax = axes[0]
        # ax.set_ylim(0.1, ax.get_ylim()[1] * 1.5)
        # ax.set_yscale('log')
        plt.savefig(f'plots/reco_study/nproton_topo_{preselection}_{selection}_{run_combo}_recovery.pdf', bbox_inches='tight')
        plt.savefig(f'plots/reco_study/nproton_topo_{preselection}_{selection}_{run_combo}_recovery.png', bbox_inches='tight')

        axes2 = plotter.plot(
            category_column="category_1e1p_tki",
            signal_category_num=12,
            include_multisim_errors=True,
            add_ext_error_floor=False,
            show_data_mc_ratio=False,
            show_chi_square=False,
            show_total_unconstrained=False,
            add_precomputed_detsys=False,
            show_errorband=True,
        )
        
    #     plt.yscale('log')
        ax2 = axes2[0]
        ax2.set_ylim(0.1, ax2.get_ylim()[1] * 1.5)
        ax2.set_yscale('log')
        plt.savefig(f'plots/reco_study/nproton_topo_{preselection}_{selection}_{run_combo}_recovery_log.pdf', bbox_inches='tight')
        plt.savefig(f'plots/reco_study/nproton_topo_{preselection}_{selection}_{run_combo}_recovery_log.png', bbox_inches='tight')

print("Done :)")