import sys
import localSettings as ls
main_path = ls.main_path
sys.path.append(main_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import nue_booster 

import uproot
import awkward
import plotter


USEBDT = True

def load_data_run123(which_sideband='pi0', return_plotter=True, pi0scaling=0):
    fold = ls.fold
    tree = "NeutrinoSelectionFilter"

    # sample list
    R1BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2_5e19'
    R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C_all_reco2'
#     R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_C2_D1_D2_E1_E2_all_reco2' #Run1 + Run2
    R1NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2'
    R1NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1NCPI0  = 'prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2'
    R1CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2'
    R1CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run1_reco2_reco2'
    R1CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'

    R2NU = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run2_reco2_D1D2_reco2"
    R2NUE = "prodgenie_bnb_intrinsic_nue_overlay_run2_v08_00_00_35_run2a_reco2_reco2"
    R2EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_D_E_all_reco2'
    
    R3BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2_1e19'
    R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_G_all_reco2'
    R3NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run3_reco2_G_reco2'
    R3NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'
    R3DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'
    R3NCPI0  = 'prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2'
    R3CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2'
    R3CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run3_reco2_reco2'
    R3CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run3_reco2_reco2'
    R3NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_new_run3_reco2_reco2'
    R3NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_New_run3_reco2_reco2'


    ur3mc = uproot.open(ls.ntuple_path+ls.RUN3+R3NU+ls.APPEND+".root")[fold][tree]
    ur3ncpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3NCPI0+ls.APPEND+".root")[fold][tree]
    ur3ccpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3CCPI0+ls.APPEND+".root")[fold][tree]
    ur3nue = uproot.open(ls.ntuple_path+ls.RUN3+R3NUE+ls.APPEND+".root")[fold][tree]
    ur3data = uproot.open(ls.ntuple_path+ls.RUN3+R3BNB+ls.APPEND+".root")[fold][tree]
    ur3ext = uproot.open(ls.ntuple_path+ls.RUN3+R3EXT+ls.APPEND+".root")[fold][tree]
    ur3dirt = uproot.open(ls.ntuple_path+ls.RUN3+R3DRT+ls.APPEND+".root")[fold][tree]
    ur3lee = uproot.open(ls.ntuple_path+ls.RUN3+R3NUE+ls.APPEND+".root")[fold][tree]
    ur3ccnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCNOPI+ls.APPEND+".root")[fold][tree]
    ur3cccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCCPI+ls.APPEND+".root")[fold][tree]
    ur3ncnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCNOPI+ls.APPEND+".root")[fold][tree]
    ur3nccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCCPI+ls.APPEND+".root")[fold][tree]


    ur2mc = uproot.open(ls.ntuple_path+ls.RUN2+R2NU+ls.APPEND+".root")[fold][tree]
    ur2nue = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
    ur2lee = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
    ur2ext = uproot.open(ls.ntuple_path+ls.RUN2+R2EXT+ls.APPEND+".root")[fold][tree]



    ur1mc = uproot.open(ls.ntuple_path+ls.RUN1+R1NU+ls.APPEND+".root")[fold][tree]
    ur1ncpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1NCPI0+ls.APPEND+".root")[fold][tree]
    ur1ccpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1CCPI0+ls.APPEND+".root")[fold][tree]
    ur1nue = uproot.open(ls.ntuple_path+ls.RUN1+R1NUE+ls.APPEND+".root")[fold][tree]
    ur1data = uproot.open(ls.ntuple_path+ls.RUN1+R1BNB+ls.APPEND+".root")[fold][tree]
    ur1ext = uproot.open(ls.ntuple_path+ls.RUN1+R1EXT+ls.APPEND+".root")[fold][tree]
    ur1dirt = uproot.open(ls.ntuple_path+ls.RUN1+R1DRT+ls.APPEND+".root")[fold][tree]
    ur1lee = uproot.open(ls.ntuple_path+ls.RUN1+R1NUE+ls.APPEND+".root")[fold][tree]
    ur1ccnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCNOPI+ls.APPEND+".root")[fold][tree]
    ur1cccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCCPI+ls.APPEND+".root")[fold][tree]
    ur1ncnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCNOPI+ls.APPEND+".root")[fold][tree]
    ur1nccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCCPI+ls.APPEND+".root")[fold][tree]
    
    R123_TWO_SHOWERS_SIDEBAND_BNB = '1e_2showers_sidebands'
    R123_NP_FAR_SIDEBAND_BNB = '1enp_far_sidebands'
    ur123data_two_showers_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    
    ur1data_two_showers_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run1_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur2data_two_showers_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run2_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur3data_two_showers_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run3_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]

    ur1data_np_far_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run1_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur2data_np_far_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run2_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur3data_np_far_sidebands = uproot.open(ls.ntuple_path+'data_sidebands/run3_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    
    variables = [
        "shr_dedx_Y", "shr_bkt_pdg", "p", "pt", "selected", "nu_pdg", "shr_theta",
        "slpdg", "trk_score_v", "backtracked_pdg", # modified from shr_score_v
        "shr_pfp_id_v", "category",
        "shr_tkfit_dedx_U","shr_tkfit_dedx_V","shr_tkfit_dedx_Y",
        "shr_tkfit_gap10_dedx_U","shr_tkfit_gap10_dedx_V","shr_tkfit_gap10_dedx_Y",
        "shr_tkfit_2cm_dedx_U","shr_tkfit_2cm_dedx_V","shr_tkfit_2cm_dedx_Y",
        #"shr_energy_tot", 
        "trk_energy_tot", "shr_hits_tot", "ccnc", "trk_chipr",
        "trk_bkt_pdg", "hits_ratio", "n_tracks_contained", 
        "crtveto","crthitpe","_closestNuCosmicDist",
        "NeutrinoEnergy0","NeutrinoEnergy1","NeutrinoEnergy2",
        #"run","sub","evt",
        "CosmicIP","CosmicDirAll3D","CosmicIPAll3D",
        "nu_flashmatch_score","best_cosmic_flashmatch_score","best_obviouscosmic_flashmatch_score",
        #"trk_pfp_id",
        "shrmoliereavg","shrmoliererms",
        "shr_tkfit_npointsvalid","shr_tkfit_npoints", # fitted vs. all hits for shower
        "shrclusfrac0","shrclusfrac1","shrclusfrac2", # track-fitted hits / all hits
        "trkshrhitdist2", "trkshrhitdist0","trkshrhitdist1", #distance between track and shower in 2D
        "shrsubclusters0","shrsubclusters1","shrsubclusters2", # number of sub-clusters in shower
        "trk_llr_pid_score_v", # trk-PID score
        #"pi0_energy2_Y", # pi0 tagger variables
        "_opfilter_pe_beam", "_opfilter_pe_veto", # did the event pass the common optical filter (for MC only)
        "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
        "nproton", "nu_e", "n_showers_contained", "shr_distance", "trk_distance",
        "hits_u", "hits_v", "hits_y", "shr_pz", "shr_energy", "shr_dedx_U", "shr_dedx_V", "shr_phi", "trk_phi", "trk_theta",
        "shr_tkfit_dedx_U", "shr_tkfit_dedx_V", "run", "sub", "evt", "nproton", "trk_pid_chipr_v",
        "trk_len", "mc_pdg", "slnunhits", "slnhits", "shr_score", "trk_score", "trk_hits_tot",
        "true_e_visible", "matched_E", "shr_bkt_E", "trk_bkt_E", "trk_energy", "tksh_distance", "tksh_angle",
        "npi0","npion","pion_e","muon_e","pi0truth_elec_etot",
        "pi0_e", "shr_energy_tot_cali", "shr_dedx_Y_cali", "evnunhits", "nslice", "interaction",
        "slclustfrac", "reco_nu_vtx_x", "reco_nu_vtx_y", "reco_nu_vtx_z","contained_fraction",
        "secondshower_U_nhit","secondshower_U_vtxdist","secondshower_U_dot","secondshower_U_dir","shrclusdir0",
        "secondshower_V_nhit","secondshower_V_vtxdist","secondshower_V_dot","secondshower_V_dir","shrclusdir1",
        "secondshower_Y_nhit","secondshower_Y_vtxdist","secondshower_Y_dot","secondshower_Y_dir","shrclusdir2",
        "shr_tkfit_nhits_Y","shr_tkfit_nhits_U","shr_tkfit_nhits_V",
        "shr_tkfit_2cm_nhits_Y","shr_tkfit_2cm_nhits_U","shr_tkfit_2cm_nhits_V",
        "shr_tkfit_gap10_nhits_Y","shr_tkfit_gap10_nhits_U","shr_tkfit_gap10_nhits_V",
        "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v","shr_id",
        "shrMCSMom","DeltaRMS2h","shrPCA1CMed_5cm","CylFrac2h_1cm",
        "trk_hits_tot", "trk_hits_u_tot", "trk_hits_v_tot", "trk_hits_y_tot",
        "shr_hits_tot", "shr_hits_u_tot", "shr_hits_v_tot", "shr_hits_y_tot",
        "shr_theta_v","shr_phi_v","shr_energy_y_v",
        "shr_start_x_v","shr_start_z_v","shr_start_z_v",
        "trk_start_x_v","trk_start_z_v","trk_start_z_v",
    ]
    #make the list unique
    variables = list(set(variables))
    print(variables)

    variables.remove("_closestNuCosmicDist")
    variables.remove("crtveto")
    variables.remove("crthitpe")

    WEIGHTS = ["weightSpline","weightTune","weightSplineTimesTune", "weightsGenie", "weightsFlux", "weightsReint"]
    WEIGHTSLEE = ["weightSpline","weightTune","weightSplineTimesTune", "leeweight", "weightsGenie", "weightsFlux", "weightsReint"]
    MCFVARS = ["mcf_nu_e","mcf_lep_e","mcf_actvol","mcf_nmm","mcf_nmp","mcf_nem","mcf_nep","mcf_np0","mcf_npp",
               "mcf_npm","mcf_mcshr_elec_etot","mcf_pass_ccpi0","mcf_pass_ncpi0",
               "mcf_pass_ccnopi","mcf_pass_ncnopi","mcf_pass_cccpi","mcf_pass_nccpi"]

    r3nue = ur3nue.pandas.df(variables + WEIGHTS, flatten=False)
    r3mc = ur3mc.pandas.df(variables + WEIGHTS + MCFVARS, flatten=False)
    r3ncpi0 = ur3ncpi0.pandas.df(variables + WEIGHTS, flatten=False)
    r3ccpi0 = ur3ccpi0.pandas.df(variables + WEIGHTS, flatten=False)
    r3ccnopi = ur3ccnopi.pandas.df(variables + WEIGHTS, flatten=False)
    r3cccpi = ur3cccpi.pandas.df(variables + WEIGHTS, flatten=False)
    r3ncnopi = ur3ncnopi.pandas.df(variables + WEIGHTS, flatten=False)
    r3nccpi = ur3nccpi.pandas.df(variables + WEIGHTS, flatten=False)
    r3data = ur3data.pandas.df(variables, flatten=False)
    r3ext = ur3ext.pandas.df(variables, flatten=False)
    r3dirt = ur3dirt.pandas.df(variables + WEIGHTS, flatten=False)
    r3lee = ur3lee.pandas.df(variables + WEIGHTSLEE, flatten=False)
    
    r3data_two_showers_sidebands = ur3data_two_showers_sidebands.pandas.df(variables, flatten=False)
    r3data_np_far_sidebands = ur3data_np_far_sidebands.pandas.df(variables, flatten=False)
    
    r3lee["is_signal"] = r3lee["category"] == 11
    r3data["is_signal"] = r3data["category"] == 11
    r3nue["is_signal"] = r3nue["category"] == 11
    r3mc["is_signal"] = r3mc["category"] == 11
    r3dirt["is_signal"] = r3dirt["category"] == 11
    r3ext["is_signal"] = r3ext["category"] == 11
    r3ncpi0["is_signal"] = r3ncpi0["category"] == 11
    r3ccpi0["is_signal"] = r3ccpi0["category"] == 11
    r3ccnopi["is_signal"] = r3ccnopi["category"] == 11
    r3cccpi["is_signal"] = r3cccpi["category"] == 11
    r3ncnopi["is_signal"] = r3ncnopi["category"] == 11
    r3nccpi["is_signal"] = r3nccpi["category"] == 11
    r3lee.loc[r3lee['category'] == 1, 'category'] = 111
    r3lee.loc[r3lee['category'] == 10, 'category'] = 111
    r3lee.loc[r3lee['category'] == 11, 'category'] = 111
    
    r3data_two_showers_sidebands["is_signal"] = r3data_two_showers_sidebands["category"] == 11
    r3data_np_far_sidebands["is_signal"] = r3data_np_far_sidebands["category"] == 11
    
    r3_datasets = [r3lee, r3data, r3nue, r3mc, r3dirt, r3ext, r3ncpi0, r3ccpi0, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi, r3lee, r3lee, r3lee, r3data_two_showers_sidebands, r3data_np_far_sidebands]
    for r3_dataset in r3_datasets:
        r3_dataset['run1'] = np.zeros(len(r3_dataset), dtype=bool)
        r3_dataset['run2'] = np.zeros(len(r3_dataset), dtype=bool)
        r3_dataset['run3'] = np.ones(len(r3_dataset), dtype=bool)
        r3_dataset['run12'] = np.zeros(len(r3_dataset), dtype=bool)
        
    uproot_v = [ur3lee,ur3mc,ur3ncpi0,ur3ccpi0,ur3ccnopi,ur3cccpi,ur3ncnopi,ur3nccpi,ur3nue,ur3ext,ur3data,ur3dirt, ur3data_two_showers_sidebands, ur3data_np_far_sidebands]
    df_v = [r3lee,r3mc,r3ncpi0,r3ccpi0,r3ccnopi,r3cccpi,r3ncnopi,r3nccpi,r3nue,r3ext,r3data,r3dirt, r3data_two_showers_sidebands, r3data_np_far_sidebands]
    for i,df in enumerate(df_v):
        up = uproot_v[i]
        trk_llr_pid_v = up.array('trk_llr_pid_score_v')
        trk_calo_energy_y_v = up.array('trk_calo_energy_y_v')
        trk_energy_proton_v = up.array('trk_energy_proton_v')
        #shr_moliere_avg_v = up.array('shr_moliere_avg_v')
        trk_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
        shr_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
        trk_llr_pid_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_llr_pid_v,trk_id)])
        trk_calo_energy_y_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_calo_energy_y_v,trk_id)])
        trk_energy_proton_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_energy_proton_v,trk_id)])
        #shr_moliere_avg_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(shr_moliere_avg_v,shr_id)])
        df['trkpid'] = trk_llr_pid_v_sel
        df['trackcaloenergy'] = trk_calo_energy_y_sel
        df['protonenergy'] = trk_energy_proton_sel
        #df['shrmoliereavg'] = shr_moliere_avg_sel
        trk_sce_start_x_v = up.array('trk_sce_start_x_v')
        trk_sce_start_y_v = up.array('trk_sce_start_y_v')
        trk_sce_start_z_v = up.array('trk_sce_start_z_v')
        trk_sce_end_x_v = up.array('trk_sce_end_x_v')
        trk_sce_end_y_v = up.array('trk_sce_end_y_v')
        trk_sce_end_z_v = up.array('trk_sce_end_z_v')
        trk_sce_start_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_x_v,shr_id)])
        trk_sce_start_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_y_v,shr_id)])
        trk_sce_start_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_z_v,shr_id)])
        trk_sce_end_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_x_v,shr_id)])
        trk_sce_end_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_y_v,shr_id)])
        trk_sce_end_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_z_v,shr_id)])
        df['shr_trk_sce_start_x'] = trk_sce_start_x_v_sel
        df['shr_trk_sce_start_y'] = trk_sce_start_y_v_sel
        df['shr_trk_sce_start_z'] = trk_sce_start_z_v_sel
        df['shr_trk_sce_end_x'] = trk_sce_end_x_v_sel
        df['shr_trk_sce_end_y'] = trk_sce_end_y_v_sel
        df['shr_trk_sce_end_z'] = trk_sce_end_z_v_sel
        #
        trk_score_v = up.array("trk_score_v")
        pfnhits_v = up.array("pfnhits")
        shr_energy_y_v = up.array("shr_energy_y_v")
        shr_theta_v = up.array("shr_theta_v")
        shr_phi_v   = up.array("shr_phi_v")
        shr_start_x_v   = up.array("shr_start_x_v")
        shr_start_y_v   = up.array("shr_start_y_v")
        shr_start_z_v   = up.array("shr_start_z_v")
        trk_start_x_v   = up.array("trk_start_x_v")
        trk_start_y_v   = up.array("trk_start_y_v")
        trk_start_z_v   = up.array("trk_start_z_v")
        shr_mask = (trk_score_v<0.5)
        df["shr1_nhits"] = awkward.fromiter([vec[vec.argsort()[-1]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr2_nhits"] = awkward.fromiter([vec[vec.argsort()[-2]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr1_energy"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_energy"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_theta"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_theta"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_phi"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_phi"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_x"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_y"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_z"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["trk_start_x"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_x_v,trk_id)])
        df["trk_start_y"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_y_v,trk_id)])
        df["trk_start_z"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_z_v,trk_id)])

    if (USEBDT == True):
        train_r3ccpi0, r3ccpi0 = train_test_split(r3ccpi0, test_size=0.5, random_state=1990)

    r1nue = ur1nue.pandas.df(variables + WEIGHTS, flatten=False)
    r1mc = ur1mc.pandas.df(variables + WEIGHTS + MCFVARS, flatten=False)
    r1ncpi0 = ur1ncpi0.pandas.df(variables + WEIGHTS, flatten=False)
    r1ccpi0 = ur1ccpi0.pandas.df(variables + WEIGHTS, flatten=False)
    r1ccnopi = ur1ccnopi.pandas.df(variables + WEIGHTS, flatten=False)
    r1cccpi = ur1cccpi.pandas.df(variables + WEIGHTS, flatten=False)
    r1ncnopi = ur1ncnopi.pandas.df(variables + WEIGHTS, flatten=False)
    r1nccpi = ur1nccpi.pandas.df(variables + WEIGHTS, flatten=False)
    r1data = ur1data.pandas.df(variables, flatten=False)
    r1ext = ur1ext.pandas.df(variables, flatten=False)
    r1dirt = ur1dirt.pandas.df(variables + WEIGHTS, flatten=False)
    r1lee = ur1lee.pandas.df(variables + WEIGHTSLEE, flatten=False)


    r1data_two_showers_sidebands = ur1data_two_showers_sidebands.pandas.df(variables, flatten=False)
    r1data_np_far_sidebands = ur1data_np_far_sidebands.pandas.df(variables, flatten=False)
    r123data_two_showers_sidebands = ur123data_two_showers_sidebands.pandas.df(variables, flatten=False)

    r1lee["is_signal"] = r1lee["category"] == 11
    r1data["is_signal"] = r1data["category"] == 11
    r1nue["is_signal"] = r1nue["category"] == 11
    r1mc["is_signal"] = r1mc["category"] == 11
    r1dirt["is_signal"] = r1dirt["category"] == 11
    r1ext["is_signal"] = r1ext["category"] == 11
    r1ncpi0["is_signal"] = r1ncpi0["category"] == 11
    r1ccpi0["is_signal"] = r1ccpi0["category"] == 11
    r1ccnopi["is_signal"] = r1ccnopi["category"] == 11
    r1cccpi["is_signal"] = r1cccpi["category"] == 11
    r1ncnopi["is_signal"] = r1ncnopi["category"] == 11
    r1nccpi["is_signal"] = r1nccpi["category"] == 11
    r1lee.loc[r1lee['category'] == 1, 'category'] = 111
    r1lee.loc[r1lee['category'] == 10, 'category'] = 111
    r1lee.loc[r1lee['category'] == 11, 'category'] = 111

    r123data_two_showers_sidebands["is_signal"] = r123data_two_showers_sidebands["category"] == 11
    r1data_two_showers_sidebands["is_signal"] = r1data_two_showers_sidebands["category"] == 11
    r1data_np_far_sidebands["is_signal"] = r1data_np_far_sidebands["category"] == 11
    
    r1_datasets = [r1lee, r1data, r1nue, r1mc, r1dirt, r1ext, r1ncpi0, r1ccpi0, r1ccnopi, r1cccpi, r1ncnopi, r1nccpi, r1lee, r123data_two_showers_sidebands, r1data_two_showers_sidebands, r1data_np_far_sidebands]
    for r1_dataset in r1_datasets:
        r1_dataset['run1'] = np.ones(len(r1_dataset), dtype=bool)
        r1_dataset['run2'] = np.zeros(len(r1_dataset), dtype=bool)
        r1_dataset['run3'] = np.zeros(len(r1_dataset), dtype=bool)
        r1_dataset['run12'] = np.ones(len(r1_dataset), dtype=bool)
    
    uproot_v = [ur1lee,ur1mc,ur1ncpi0,ur1ccpi0,ur1ccnopi,ur1cccpi,ur1ncnopi,ur1nccpi,ur1nue,ur1ext,ur1data,ur1dirt, ur123data_two_showers_sidebands, ur1data_two_showers_sidebands, ur1data_np_far_sidebands]
    df_v = [r1lee,r1mc,r1ncpi0,r1ccpi0,r1ccnopi,r1cccpi,r1ncnopi,r1nccpi,r1nue,r1ext,r1data,r1dirt, r123data_two_showers_sidebands, r1data_two_showers_sidebands, r1data_np_far_sidebands]
    for i,df in enumerate(df_v):
        up = uproot_v[i]
        trk_llr_pid_v = up.array('trk_llr_pid_score_v')
        trk_calo_energy_y_v = up.array('trk_calo_energy_y_v')
        trk_energy_proton_v = up.array('trk_energy_proton_v')
        #shr_moliere_avg_v = up.array('shr_moliere_avg_v')
        trk_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
        shr_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
        trk_llr_pid_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_llr_pid_v,trk_id)])
        trk_calo_energy_y_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_calo_energy_y_v,trk_id)])
        trk_energy_proton_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_energy_proton_v,trk_id)])
        #shr_moliere_avg_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(shr_moliere_avg_v,shr_id)])
        df['trkpid'] = trk_llr_pid_v_sel
        df['trackcaloenergy'] = trk_calo_energy_y_sel
        df['protonenergy'] = trk_energy_proton_sel
        #df['shrmoliereavg'] = shr_moliere_avg_sel
        trk_sce_start_x_v = up.array('trk_sce_start_x_v')
        trk_sce_start_y_v = up.array('trk_sce_start_y_v')
        trk_sce_start_z_v = up.array('trk_sce_start_z_v')
        trk_sce_end_x_v = up.array('trk_sce_end_x_v')
        trk_sce_end_y_v = up.array('trk_sce_end_y_v')
        trk_sce_end_z_v = up.array('trk_sce_end_z_v')
        trk_sce_start_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_x_v,shr_id)])
        trk_sce_start_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_y_v,shr_id)])
        trk_sce_start_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_z_v,shr_id)])
        trk_sce_end_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_x_v,shr_id)])
        trk_sce_end_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_y_v,shr_id)])
        trk_sce_end_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_z_v,shr_id)])
        df['shr_trk_sce_start_x'] = trk_sce_start_x_v_sel
        df['shr_trk_sce_start_y'] = trk_sce_start_y_v_sel
        df['shr_trk_sce_start_z'] = trk_sce_start_z_v_sel
        df['shr_trk_sce_end_x'] = trk_sce_end_x_v_sel
        df['shr_trk_sce_end_y'] = trk_sce_end_y_v_sel
        df['shr_trk_sce_end_z'] = trk_sce_end_z_v_sel
        #
        trk_score_v = up.array("trk_score_v")
        pfnhits_v = up.array("pfnhits")
        shr_energy_y_v = up.array("shr_energy_y_v")
        shr_theta_v = up.array("shr_theta_v")
        shr_phi_v   = up.array("shr_phi_v")
        shr_start_x_v   = up.array("shr_start_x_v")
        shr_start_y_v   = up.array("shr_start_y_v")
        shr_start_z_v   = up.array("shr_start_z_v")
        trk_start_x_v   = up.array("trk_start_x_v")
        trk_start_y_v   = up.array("trk_start_y_v")
        trk_start_z_v   = up.array("trk_start_z_v")
        shr_mask = (trk_score_v<0.5)
        df["shr1_nhits"] = awkward.fromiter([vec[vec.argsort()[-1]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr2_nhits"] = awkward.fromiter([vec[vec.argsort()[-2]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr1_energy"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_energy"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_theta"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_theta"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_phi"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_phi"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_x"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_y"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_z"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["trk_start_x"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_x_v,trk_id)])
        df["trk_start_y"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_y_v,trk_id)])
        df["trk_start_z"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_z_v,trk_id)])


    r2nue = ur2nue.pandas.df(variables + WEIGHTS, flatten=False)
    r2mc = ur2mc.pandas.df(variables + WEIGHTS + MCFVARS, flatten=False)
    r2ext = ur2ext.pandas.df(variables, flatten=False)
    r2lee = ur2lee.pandas.df(variables + WEIGHTSLEE, flatten=False)
    
    r2data_two_showers_sidebands = ur2data_two_showers_sidebands.pandas.df(variables, flatten=False)
    r2data_np_far_sidebands = ur2data_np_far_sidebands.pandas.df(variables, flatten=False)
    
    r2lee["is_signal"] = r2lee["category"] == 11
    r2nue["is_signal"] = r2nue["category"] == 11
    r2mc["is_signal"] = r2mc["category"] == 11
    r2ext["is_signal"] = r2ext["category"] == 11
    r2lee.loc[r2lee['category'] == 1, 'category'] = 111
    r2lee.loc[r2lee['category'] == 10, 'category'] = 111
    r2lee.loc[r2lee['category'] == 11, 'category'] = 111
    
    r2data_two_showers_sidebands["is_signal"] = r2data_two_showers_sidebands["category"] == 11
    r2data_np_far_sidebands["is_signal"] = r2data_np_far_sidebands["category"] == 11
    
    r2_datasets = [r2lee, r2nue, r2mc, r2ext, r2data_two_showers_sidebands, r2data_np_far_sidebands]
    for r2_dataset in r2_datasets:
        r2_dataset['run1'] = np.zeros(len(r2_dataset), dtype=bool)
        r2_dataset['run2'] = np.ones(len(r2_dataset), dtype=bool)
        r2_dataset['run3'] = np.zeros(len(r2_dataset), dtype=bool)
        r2_dataset['run12'] = np.ones(len(r2_dataset), dtype=bool)
    
    for r_dataset in [r1ncpi0, r1ccpi0, r1ccnopi, r1cccpi, r1ncnopi, r1nccpi, r3ncpi0, r3ccpi0, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]:
        r_dataset['run2'] = np.ones(len(r_dataset), dtype=bool)
    
    uproot_v = [ur2lee,ur2mc,ur2nue, ur2ext, ur2data_two_showers_sidebands, ur2data_np_far_sidebands]
    df_v = [r2lee,r2mc,r2nue, r2ext, r2data_two_showers_sidebands, r2data_np_far_sidebands]
    for i,df in enumerate(df_v):
        up = uproot_v[i]
        trk_llr_pid_v = up.array('trk_llr_pid_score_v')
        trk_calo_energy_y_v = up.array('trk_calo_energy_y_v')
        trk_energy_proton_v = up.array('trk_energy_proton_v')
        #shr_moliere_avg_v = up.array('shr_moliere_avg_v')
        trk_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
        shr_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
        trk_llr_pid_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_llr_pid_v,trk_id)])
        trk_calo_energy_y_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_calo_energy_y_v,trk_id)])
        trk_energy_proton_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_energy_proton_v,trk_id)])
        #shr_moliere_avg_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(shr_moliere_avg_v,shr_id)])
        df['trkpid'] = trk_llr_pid_v_sel
        df['trackcaloenergy'] = trk_calo_energy_y_sel
        df['protonenergy'] = trk_energy_proton_sel
        #df['shrmoliereavg'] = shr_moliere_avg_sel
        trk_sce_start_x_v = up.array('trk_sce_start_x_v')
        trk_sce_start_y_v = up.array('trk_sce_start_y_v')
        trk_sce_start_z_v = up.array('trk_sce_start_z_v')
        trk_sce_end_x_v = up.array('trk_sce_end_x_v')
        trk_sce_end_y_v = up.array('trk_sce_end_y_v')
        trk_sce_end_z_v = up.array('trk_sce_end_z_v')
        trk_sce_start_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_x_v,shr_id)])
        trk_sce_start_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_y_v,shr_id)])
        trk_sce_start_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_start_z_v,shr_id)])
        trk_sce_end_x_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_x_v,shr_id)])
        trk_sce_end_y_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_y_v,shr_id)])
        trk_sce_end_z_v_sel = awkward.fromiter([pidv[tid] if tid<len(pidv) else -9999. for pidv,tid in zip(trk_sce_end_z_v,shr_id)])
        df['shr_trk_sce_start_x'] = trk_sce_start_x_v_sel
        df['shr_trk_sce_start_y'] = trk_sce_start_y_v_sel
        df['shr_trk_sce_start_z'] = trk_sce_start_z_v_sel
        df['shr_trk_sce_end_x'] = trk_sce_end_x_v_sel
        df['shr_trk_sce_end_y'] = trk_sce_end_y_v_sel
        df['shr_trk_sce_end_z'] = trk_sce_end_z_v_sel
        #
        trk_score_v = up.array("trk_score_v")
        pfnhits_v = up.array("pfnhits")
        shr_energy_y_v = up.array("shr_energy_y_v")
        shr_theta_v = up.array("shr_theta_v")
        shr_phi_v   = up.array("shr_phi_v")
        shr_start_x_v   = up.array("shr_start_x_v")
        shr_start_y_v   = up.array("shr_start_y_v")
        shr_start_z_v   = up.array("shr_start_z_v")
        trk_start_x_v   = up.array("trk_start_x_v")
        trk_start_y_v   = up.array("trk_start_y_v")
        trk_start_z_v   = up.array("trk_start_z_v")
        shr_mask = (trk_score_v<0.5)
        df["shr1_nhits"] = awkward.fromiter([vec[vec.argsort()[-1]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr2_nhits"] = awkward.fromiter([vec[vec.argsort()[-2]] if len(vec)>1 else -9999. for vec in pfnhits_v[shr_mask]])
        df["shr1_energy"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_energy"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_energy_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_theta"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_theta"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_theta_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_phi"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_phi"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_phi_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_x"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_x_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_y"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_y_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr1_start_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["shr2_start_z"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(shr_start_z_v[shr_mask],pfnhits_v[shr_mask])])
        df["trk_start_x"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_x_v,trk_id)])
        df["trk_start_y"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_y_v,trk_id)])
        df["trk_start_z"] = awkward.fromiter([pidv[tid] if tid<len(pidv) else 9999. for pidv,tid in zip(trk_start_z_v,trk_id)])

    
    #set weights
#     r1nue["pot_scale"] = 2.74E-03/3.32e-03
#     r2nue["pot_scale"] = 4.20E-03/3.32e-03
#     r3nue["pot_scale"] = 2.94E-03/3.32e-03
    
#     r1lee["pot_scale"] = r1nue["pot_scale"]
#     r2lee["pot_scale"] = r2nue["pot_scale"]
#     r3lee["pot_scale"] = r3nue["pot_scale"]
    
#     r1mc["pot_scale"] = 1.11E-01/1.61e-01
#     r2mc["pot_scale"] = 2.54E-01/1.61e-01
#     r3mc["pot_scale"] = 1.39E-01/1.61e-01
    
#     r1ext["pot_scale"] = 4.97E-01/5.01e-01 
#     r2ext["pot_scale"] = 4.98E-01/5.01e-01 
#     r3ext["pot_scale"] = 5.09E-01/5.01e-01 

    r1nue["pot_scale"] = 1
    r2nue["pot_scale"] = 1
    r3nue["pot_scale"] = 1
    
    r1lee["pot_scale"] = 1
    r2lee["pot_scale"] = 1
    r3lee["pot_scale"] = 1
    
    r1mc["pot_scale"] = 1
    r2mc["pot_scale"] = 1
    r3mc["pot_scale"] = 1
    
    r1ext["pot_scale"] = 1 
    r2ext["pot_scale"] = 1 
    r3ext["pot_scale"] = 1 

    nue = pd.concat([r1nue,r2nue,r3nue],ignore_index=True)
    #nue = pd.concat([r3nue,r1nue],ignore_index=True)
    mc = pd.concat([r3mc,r2mc,r1mc],ignore_index=True)
    #mc = pd.concat([r3mc,r1mc],ignore_index=True)
    ncpi0 = pd.concat([r3ncpi0,r1ncpi0],ignore_index=True)
    ccpi0 = pd.concat([r3ccpi0,r1ccpi0],ignore_index=True)
    ccnopi = pd.concat([r3ccnopi,r1ccnopi],ignore_index=True)
    cccpi = pd.concat([r3cccpi,r1cccpi],ignore_index=True)
    ncnopi = pd.concat([r3ncnopi,r1ncnopi],ignore_index=True)
    nccpi = pd.concat([r3nccpi,r1nccpi],ignore_index=True)
    # data = pd.concat([r3data,r1data],ignore_index=True)
#     data = pd.concat([r123data_two_showers_sidebands],ignore_index=True)
    if which_sideband == '2plus_showers':
        data = pd.concat([r1data_two_showers_sidebands, r2data_two_showers_sidebands, r3data_two_showers_sidebands],ignore_index=True)
    elif which_sideband == 'np_far':
        data = pd.concat([r1data_np_far_sidebands, r2data_np_far_sidebands, r3data_np_far_sidebands],ignore_index=True)
    ext = pd.concat([r3ext, r2ext, r1ext],ignore_index=True)
    dirt = pd.concat([r3dirt,r1dirt],ignore_index=True)
    lee = pd.concat([r1lee,r2lee,r3lee],ignore_index=True)
    #lee = pd.concat([r3lee,r1lee],ignore_index=True)
    
    df_v = [lee,mc,ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi,nue,dirt]

    for i,df in enumerate(df_v):

        df.loc[ df['weightTune'] <= 0, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] == np.inf, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] > 100, 'weightTune' ] = 1.
        df.loc[ np.isnan(df['weightTune']) == True, 'weightTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
        df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
        if pi0scaling == 1:
            df.loc[ df['npi0'] > 0, 'weightSplineTimesTune' ] = df['weightSpline'] * df['weightTune'] * 0.759
        elif pi0scaling == 2:
            df.loc[ df['pi0_e'] > 0.1, 'weightSplineTimesTune' ] = df['weightSplineTimesTune']*(1.-0.35*df['pi0_e'])
        
    df_v = [lee,mc,ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi,nue,ext,data,dirt]

    for i,df in enumerate(df_v):
        df['subcluster'] = df['shrsubclusters0'] + df['shrsubclusters1'] + df['shrsubclusters2']
        df['trkfit'] = df['shr_tkfit_npointsvalid'] / df['shr_tkfit_npoints']
        # and the 2d angle difference
        df['anglediff_Y'] = np.abs(df['secondshower_Y_dir']-df['shrclusdir2'])
        df['anglediff_V'] = np.abs(df['secondshower_V_dir']-df['shrclusdir1'])
        df['anglediff_U'] = np.abs(df['secondshower_U_dir']-df['shrclusdir0'])
        #
        df["hitratio_shr12"] = (df["shr2_nhits"]/df["shr1_nhits"])
        df["hitratio_mod_shr12"] = (df["shr2_nhits"]/(df["shr1_nhits"]*np.sqrt(df["shr1_nhits"])))
        df["cos_shr12"] = np.sin(df["shr1_theta"])*np.cos(df["shr1_phi"])*np.sin(df["shr2_theta"])*np.cos(df["shr2_phi"])\
                        + np.sin(df["shr1_theta"])*np.sin(df["shr1_phi"])*np.sin(df["shr2_theta"])*np.sin(df["shr2_phi"])\
                        + np.cos(df["shr1_theta"])*np.cos(df["shr2_theta"])
        df["tksh1_dist"] = np.sqrt( (df["shr1_start_x"]-df["trk_start_x"])**2 + (df["shr1_start_y"]-df["trk_start_y"])**2 + (df["shr1_start_z"]-df["trk_start_z"])**2)
        df["tksh2_dist"] = np.sqrt( (df["shr2_start_x"]-df["trk_start_x"])**2 + (df["shr2_start_y"]-df["trk_start_y"])**2 + (df["shr2_start_z"]-df["trk_start_z"])**2)
        df["min_tksh_dist"] = np.minimum(df["tksh1_dist"],df["tksh2_dist"])
        df["max_tksh_dist"] = np.maximum(df["tksh1_dist"],df["tksh2_dist"])

    for i,df in enumerate(df_v):
        df["ptOverP"] = df["pt"]/df["p"]
        df["phi1MinusPhi2"] = df["shr_phi"]-df["trk_phi"]
        df["theta1PlusTheta2"] = df["shr_theta"]+df["trk_theta"]

    df_v = [lee,mc,ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi,nue,ext,data,dirt]
    for i,df in enumerate(df_v):
        df['shr_tkfit_nhits_tot'] = (df['shr_tkfit_nhits_Y']+df['shr_tkfit_nhits_U']+df['shr_tkfit_nhits_V'])
        df['shr_tkfit_dedx_avg'] = (df['shr_tkfit_nhits_Y']*df['shr_tkfit_dedx_Y'] + df['shr_tkfit_nhits_U']*df['shr_tkfit_dedx_U'] + df['shr_tkfit_nhits_V']*df['shr_tkfit_dedx_V'])/df['shr_tkfit_nhits_tot']
        df['shr_tkfit_2cm_nhits_tot'] = (df['shr_tkfit_2cm_nhits_Y']+df['shr_tkfit_2cm_nhits_U']+df['shr_tkfit_2cm_nhits_V'])
        df['shr_tkfit_2cm_dedx_avg'] = (df['shr_tkfit_2cm_nhits_Y']*df['shr_tkfit_2cm_dedx_Y'] + df['shr_tkfit_2cm_nhits_U']*df['shr_tkfit_2cm_dedx_U'] + df['shr_tkfit_2cm_nhits_V']*df['shr_tkfit_2cm_dedx_V'])/df['shr_tkfit_2cm_nhits_tot']
        df['shr_tkfit_gap10_nhits_tot'] = (df['shr_tkfit_gap10_nhits_Y']+df['shr_tkfit_gap10_nhits_U']+df['shr_tkfit_gap10_nhits_V'])
        df['shr_tkfit_gap10_dedx_avg'] = (df['shr_tkfit_gap10_nhits_Y']*df['shr_tkfit_gap10_dedx_Y'] + df['shr_tkfit_gap10_nhits_U']*df['shr_tkfit_gap10_dedx_U'] + df['shr_tkfit_gap10_nhits_V']*df['shr_tkfit_gap10_dedx_V'])/df['shr_tkfit_gap10_nhits_tot']
        df.loc[:,'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_Y']
        df.loc[(df['shr_tkfit_nhits_U']>df['shr_tkfit_nhits_Y']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_U']
        df.loc[(df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_Y']) & (df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_U']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_V']
        
    
    INTERCEPT = 0.0
    SLOPE = 0.83

    # define some energy-related variables
    for i,df in enumerate(df_v):
        df["reco_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE + df["trk_energy_tot"]
        df["reco_e_qe"] = 0.938*((df["shr_energy"]+INTERCEPT)/SLOPE)/(0.938 - ((df["shr_energy"]+INTERCEPT)/SLOPE)*(1-np.cos(df["shr_theta"])))
        df["reco_e_rqe"] = df["reco_e_qe"]/df["reco_e"]

    # and a way to filter out data
    for i,df in enumerate(df_v):
        df["bnbdata"] = np.zeros_like(df["shr_energy"])
        df["extdata"] = np.zeros_like(df["shr_energy"])
    data["bnbdata"] = np.ones_like(data["shr_energy"])
    ext["extdata"] = np.ones_like(ext["shr_energy"])
    
    
    # avoid double-counting of events out of FV in the NC/CC pi0 samples
    # not needed anymore since we improved matching with filtered samples
    #ncpi0 = ncpi0.query('category != 5')
    #ccpi0 = ccpi0.query('category != 5')
    #ccnopi = ccnopi.query('category != 5')
    #nccpi = nccpi.query('category != 5')
    #ncnopi = ncnopi.query('category != 5')

    ## avoid recycling unbiased ext events (i.e. selecting a slice with little nu content from these samples)
    ccnopi = ccnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
    cccpi = cccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
    ncnopi = ncnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
    nccpi = nccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')

    # add back the cosmic category, for background only
    df_v = [lee,mc,ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi,nue,ext,data,dirt]
    for i,df in enumerate(df_v):
        df.loc[(df['category']!=1)&(df['category']!=10)&(df['category']!=11)&(df['category']!=111)&(df['slnunhits']/df['slnhits']<0.2), 'category'] = 4
        
        
    # Np BDT

    TRAINVAR = ["shr_score","tksh_distance","tksh_angle",
                "shr_tkfit_dedx_max",
                "trkfit","trkpid",
                "subcluster","shrmoliereavg",
                "trkshrhitdist2","hits_ratio",
                "secondshower_Y_nhit","secondshower_Y_vtxdist","secondshower_Y_dot","anglediff_Y",
                "CosmicIPAll3D","CosmicDirAll3D"]
    
    LABELS =  ['pi0','nonpi0']
    
    if (USEBDT == True):
        for label, bkg_query in zip(LABELS, nue_booster.bkg_queries):
            with open(ls.pickle_path+'booster_%s_0304_extnumi.pickle' % label, 'rb') as booster_file:
                booster = pickle.load(booster_file)
                mc[label+"_score"] = booster.predict(
                    xgb.DMatrix(mc[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                nue[label+"_score"] = booster.predict(
                    xgb.DMatrix(nue[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                ext[label+"_score"] = booster.predict(
                    xgb.DMatrix(ext[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                data[label+"_score"] = booster.predict(
                    xgb.DMatrix(data[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                dirt[label+"_score"] = booster.predict(
                    xgb.DMatrix(dirt[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                lee[label+"_score"] = booster.predict(
                    xgb.DMatrix(lee[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                ncpi0[label+"_score"] = booster.predict(
                    xgb.DMatrix(ncpi0[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                ccpi0[label+"_score"] = booster.predict(
                    xgb.DMatrix(ccpi0[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                ccnopi[label+"_score"] = booster.predict(
                    xgb.DMatrix(ccnopi[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                cccpi[label+"_score"] = booster.predict(
                    xgb.DMatrix(cccpi[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                ncnopi[label+"_score"] = booster.predict(
                    xgb.DMatrix(ncnopi[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                nccpi[label+"_score"] = booster.predict(
                    xgb.DMatrix(nccpi[TRAINVAR]),
                    ntree_limit=booster.best_iteration)
                
    # 0p BDT

    TRAINVARZP = ['shrmoliereavg','shr_score', "trkfit","subcluster",
                  "CosmicIPAll3D","CosmicDirAll3D",
                  'secondshower_Y_nhit','secondshower_Y_vtxdist','secondshower_Y_dot','anglediff_Y',
                  'secondshower_V_nhit','secondshower_V_vtxdist','secondshower_V_dot','anglediff_V',
                  'secondshower_U_nhit','secondshower_U_vtxdist','secondshower_U_dot','anglediff_U',
                  "shr_tkfit_2cm_dedx_U", "shr_tkfit_2cm_dedx_V", "shr_tkfit_2cm_dedx_Y",
                  "shr_tkfit_gap10_dedx_U", "shr_tkfit_gap10_dedx_V", "shr_tkfit_gap10_dedx_Y",
                  "shrMCSMom","DeltaRMS2h","shrPCA1CMed_5cm","CylFrac2h_1cm"]

    LABELSZP = ['bkg']

    if (USEBDT == True):
        for label, bkg_query in zip(LABELSZP, nue_booster.bkg_queries):
            with open(ls.pickle_path+'booster_%s_0304_extnumi_vx.pickle' % label, 'rb') as booster_file:
                booster = pickle.load(booster_file)
                mc[label+"_score"] = booster.predict(
                    xgb.DMatrix(mc[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                nue[label+"_score"] = booster.predict(
                    xgb.DMatrix(nue[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                ext[label+"_score"] = booster.predict(
                    xgb.DMatrix(ext[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                data[label+"_score"] = booster.predict(
                    xgb.DMatrix(data[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                dirt[label+"_score"] = booster.predict(
                    xgb.DMatrix(dirt[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                lee[label+"_score"] = booster.predict(
                    xgb.DMatrix(lee[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                ncpi0[label+"_score"] = booster.predict(
                    xgb.DMatrix(ncpi0[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                ccpi0[label+"_score"] = booster.predict(
                    xgb.DMatrix(ccpi0[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                ccnopi[label+"_score"] = booster.predict(
                    xgb.DMatrix(ccnopi[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                cccpi[label+"_score"] = booster.predict(
                    xgb.DMatrix(cccpi[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                ncnopi[label+"_score"] = booster.predict(
                    xgb.DMatrix(ncnopi[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                nccpi[label+"_score"] = booster.predict(
                    xgb.DMatrix(nccpi[TRAINVARZP]),
                    ntree_limit=booster.best_iteration)
                
    data = data.drop_duplicates(subset=['reco_e'],keep='first')

    samples = {
    "mc": mc,
    "nue": nue,
    "data": data,
    "ext": ext,
    "dirt": dirt,
    "ncpi0": ncpi0,
    "ccpi0": ccpi0,
    "ccnopi": ccnopi,
    "cccpi": cccpi,
    "ncnopi": ncnopi,
    "nccpi": nccpi,
    "lee": lee
    }
    
    if return_plotter is True:
        scaling = 1

        SPLIT = 1.0
        if (USEBDT == True):
            SPLIT = 1.48

        #''' 0304
        weights = {
            "mc": 1.61e-01 * scaling, 
            "ext": 5.01e-01 * scaling, 
            "nue": 3.32e-03 * scaling,
            "lee": 3.32e-03 * scaling,
            "dirt": 9.09e-01 * scaling,
            "ncpi0": 1.19e-01 * scaling,
            "ccpi0": 5.92e-02 * SPLIT * scaling,
            "ncnopi": 5.60e-02 * scaling,
            "nccpi": 2.58e-02 * scaling,
            "ccnopi": 6.48e-02 * scaling,
            "cccpi": 5.18e-02 * scaling,
        }
        pot = 5.88e20*scaling

        my_plotter = plotter.Plotter(samples, weights, pot=pot)
        return my_plotter
    else:
        return samples
    
pot_data_unblinded = {
    1: (1.45E+20, 32139256),
    2: (2.58E+20, 60909877),
    3: (1.86E+20, 44266555),
}

pot_mc_samples = {}

pot_mc_samples[1] = {
    'mc': 1.31E+21,
    'nue': 5.27E+22,
    'lee': 5.27E+22,
    'ncpi0': 2.67E+21,
    'ccpi0': 3.4873E+21,
    'dirt': 3.22E+20,
    'ncnopi': 3.64E+21,
    'nccpi': 8.95E+21,
    'ccnopi': 4.63E+21,
    'cccpi': 6.05E+21,
    'ext': 64672423,
}

pot_mc_samples[2] = {
    'mc': 1.02E+21,
    'nue': 6.14E+22,
    'lee': 6.14E+22,
    'ext': 122320769,
}

pot_mc_samples[3] = {
    'mc': 1.34E+21,
    'nue': 6.32E+22,
    'lee': 6.32E+22,
    'ncpi0': 2.29E+21,
    'ccpi0': 3.2293E+21,
    'dirt': 3.25E+20,
    'ncnopi': 6.87E+21,
    'nccpi': 1.39E+22,
    'ccnopi': 4.45E+21,
    'cccpi': 5.30E+21,
    'ext': 86991453,
}

def get_weights(run):
    assert run in [1, 2, 3, 123, 12]
    weights_out = {}
    if run in [1, 2, 3]:
        pot_on, n_trig_on = pot_data_unblinded[run]
        for sample, pot in pot_mc_samples[run].items():
            if sample == 'ext':
                weights_out[sample] = n_trig_on/pot
            else:
                weights_out[sample] = pot_on/pot
        if run == 2:
            for sample in ['ncpi0', 'ccpi0', 'dirt', 'ncnopi', 'nccpi', 'ccnopi', 'cccpi']:
                weights_out[sample] = pot_on/(pot_mc_samples[1][sample] + pot_mc_samples[3][sample])
        pot_out = pot_on
    elif run == 123:
        total_pot_on = 0
        total_n_trig_on = 0
        for run in [1, 2, 3]:
            pot_on, n_trig_on = pot_data_unblinded[run]
            total_pot_on += pot_on
            total_n_trig_on += n_trig_on
        for sample in pot_mc_samples[1].keys():
            this_sample_pot = 0
            for run in [1, 2, 3]:
                if sample in pot_mc_samples[run].keys():
                    this_sample_pot += pot_mc_samples[run][sample]
            if sample == 'ext':
                weights_out[sample] = total_n_trig_on/this_sample_pot
            else:
                weights_out[sample] = total_pot_on/this_sample_pot
        pot_out = total_pot_on
    elif run == 12:
        total_pot_on = 0
        total_n_trig_on = 0
        for run in [1, 2]:
            pot_on, n_trig_on = pot_data_unblinded[run]
            total_pot_on += pot_on
            total_n_trig_on += n_trig_on

        for sample in pot_mc_samples[1].keys():
            this_sample_pot = 0
            for run in [1, 2]:
                if sample in pot_mc_samples[run].keys():
                    this_sample_pot += pot_mc_samples[run][sample]
            if sample == 'ext':
                weights_out[sample] = total_n_trig_on/this_sample_pot
            else:
                weights_out[sample] = total_pot_on/this_sample_pot
        pot_out = total_pot_on
        
    return weights_out, pot_out
                    
                
                    
            