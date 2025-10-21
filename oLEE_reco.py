import numpy as np
import math
from numu_tki import tki_calculators
import sys
sys.path.append("../")

# Functions for determining a particle's velocity, and if that velocity passes the critical velocity to produce Cherenkov light in a MiniBooNE-like detector using reco Monte Carlo variables
# Author: J.B. Tyler & M. Moudgalya

################################################################################
# Remove dead zone

DEAD_Z_MIN = 675
DEAD_Z_MAX = 775

def reco_not_dead_region(reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z):
    
    return not (reco_nu_vtx_sce_z > DEAD_Z_MIN and reco_nu_vtx_sce_z < DEAD_Z_MAX)

################################################################################

#MUON_TRACK_SCORE_CUT = 0.8
#MUON_VTX_DISTANCE_CUT = 4.
#MUON_LENGTH_CUT = 10.
#MUON_PID_CUT = 0.2

#def find_muon_candidate(pfp_generation_v,trk_score_v,trk_distance_v,trk_len_v,trk_llr_pid_score_v):
   
#    index=-1
#    highest_score=-1000  
#    for i in range(0,len(pfp_generation_v)):
#        if pfp_generation_v[i] != 2 or trk_score_v[i] < MUON_TRACK_SCORE_CUT or\
#           trk_distance_v[i] > MUON_VTX_DISTANCE_CUT or trk_len_v[i] < MUON_LENGTH_CUT or\
#           trk_llr_pid_score_v[i] < MUON_PID_CUT: continue   
#        if trk_llr_pid_score_v[i] > highest_score:
#            index=i
#            highest_score=trk_llr_pid_score_v[i] 

#    return index

################################################################################
# Make a list of the indices of the protons candidates

#DEFAULT_PROTON_PID_CUT = 0.2
#TRACK_SCORE_CUT = 0.5

#def find_proton_candidates(MuonCandidateIdx,pfp_generation_v,trk_score_v,trk_len_v,trk_llr_pid_score_v):

#    proton_candidate_idx=[]
#    for i in range(0,len(pfp_generation_v)):
#        if i == MuonCandidateIdx or\
#           trk_score_v[i] < TRACK_SCORE_CUT or trk_len_v[i] < 0.0 or\
#           trk_llr_pid_score_v[i] > DEFAULT_PROTON_PID_CUT: continue
        
#        proton_candidate_idx.append(i) 

#    return proton_candidate_idx

################################################################################
# Get momentum of reconstructed protons. Returns either a vector or 

#MASS_PROTON = 0.93827

#def get_reco_proton_mom_v(ProtonCandidateIdx,trk_energy_proton_v):

#    recopmom = []
#    for i in range(0,len(ProtonCandidateIdx)):
#        ke = trk_energy_proton_v[ProtonCandidateIdx[i]]
#        recomom.append(math.sqrt(ke*ke + 2*MASS_PROTON*ke))

#    return recopmom

################################################################################
# Get momentum of reconstructed protons. Returns either a vector or 

#MASS_PROTON = 0.93827

#def get_reco_proton_E_v(RecoProtonMomentum):

#    recoEp = []
#    for i in range(0,len(RecoProtonMomentum)):
#        recoEp.append(math.sqrt(RecoProtonMomentum[i]**2 + MASS_PROTON**2)) 

#    return recoEp

################################################################################
# Determine visible-to-mB showers

def get_nshr_visible(shr_energy_cali): #, shr_energy_second_cali):

    E_mB_shr_cut = 0.20 #GeV
    corr_shr_energy_cali = shr_energy_cali/0.83
#    corr_shr_energy_second_cali = shr_energy_cali/0.83

    if corr_shr_energy_cali > E_mB_shr_cut:
        return 1

#    if corr_shr_energy_cali > E_mB_shr_cut and corr_shr_energy_second_cali > E_mB_shr_cut:
#        return 2
    
#    elif corr_shr_energy_cali > E_mB_shr_cut and corr_shr_energy_second_cali < E_mB_shr_cut:
#        return 1
    
    else:
        return 0

################################################################################
# Calculate reco velocity of reco showers

#def get_reco_vel_proton(shr_energy_cali, shr_px, shr_py, shr_pz, shr_energy):
        
#    corr_shr_energy_cali = shr_energy_cali/0.83
#    corr_shr_px = shr_px*shr_energy_cali/shr_energy/0.83
#    corr_shr_py = shr_py*shr_energy_cali/shr_energy/0.83
#    corr_shr_pz = shr_pz*shr_energy_cali/shr_energy/0.83
#    reco_shr_p = np.sqrt(corr_shr_px**2 + corr_shr_py**2 + corr_shr_pz**2)
#    v = reco_shr_p/(shr_energy_cali*0.83) #units of 1/c 
        
#    return v

################################################################################
# Determine if shower velocity passes critical velocity test

#c = 1 #check later -> this is fine
#n = 1.4620 # Index of refraction for the Marcol 7 mineral oil used by MiniBooNE

#crit_vel = c/n

#def reco_crit_vel_test_vector(reco_vel_shr):
    
#    pass_reco_crit_vel = [] # Velocity is greater than critical velocity
    
#    for i in range(len(reco_vel_shr)):
#        if reco_vel_shr > crit_vel:
#            test = True
#        else:
#            test = False
#        pass_reco_crit_vel.append(test)
        
#    return pass_reco_crit_vel

###############################################################################
# Determine if slice contains any shower that passed critical velocity test

#def reco_crit_vel_slice_bool(pass_reco_crit_vel, not_dead_region):
    
#    if True in pass_reco_crit_vel and not_dead_region == True:
#        return True
#    else:
#        return False

###############################################################################
# Calculate reco KE of mc reco shower

def get_reco_KE_shr(shr_energy_cali, shr_px, shr_py, shr_pz, shr_energy, nshr_visible):
    
    corr_shr_energy_cali = shr_energy_cali/0.83
#    corr_shr_energy_second_cali = shr_energy_second_cali/0.83
#    corr_shr_second_px = shr_second_px*corr_shr_energy_second_cali/shr_energy_second
#    corr_shr_second_py = shr_second_py*corr_shr_energy_second_cali/shr_energy_second
#    corr_shr_second_pz = shr_second_pz*corr_shr_energy_second_cali/shr_energy_second
    
    if nshr_visible > 0:
        corr_shr_px = shr_px*corr_shr_energy_cali/shr_energy
        corr_shr_py = shr_py*corr_shr_energy_cali/shr_energy
        corr_shr_pz = shr_pz*corr_shr_energy_cali/shr_energy
        reco_shr_p = np.sqrt(corr_shr_px**2 + corr_shr_py**2 + corr_shr_pz**2)
        ke = (corr_shr_energy_cali) - np.sqrt((corr_shr_energy_cali)**2 - reco_shr_p**2)
        return ke
        
#    elif nshr_visible > 1:
#        reco_shr_p = np.sqrt(corr_shr_px**2 + corr_shr_py**2 + corr_shr_pz**2)
#        reco_shr_second_p = np.sqrt(corr_shr_second_px**2 + corr_shr_second_py**2 + corr_shr_second_pz**2)
#        ke1 = (corr_shr_energy_cali) - np.sqrt((corr_shr_energy_cali)**2 - reco_shr_p**2)
#        ke1 = (corr_shr_energy_second_cali) - np.sqrt((corr_shr_energy_second_cali)**2 - reco_shr_second_p**2)
#        ke = ke1 + ke2
#        return ke

##############################################################################
# Calculating lost energy (found energy: remove "not" -> AboveChrnkv_E)

#def get_reco_SubChrnkv_E(reco_crit_vel_test_vector, reco_KE_shr_vector):
    
#    iv = 0.0
    
#    for i in range(len(reco_crit_vel_test_vector)):
#        if not reco_crit_vel_test_vector[i]:
#            iv = iv + reco_KE_shr_vector[i]
            
#    return iv            
    
##############################################################################
# Calculating found energy

#def get_reco_AboveChrnkv_E(reco_crit_vel_test_vector, reco_KE_shr_vector):
    
#    iv = 0.0
    
#    for i in range(len(reco_crit_vel_test_vector)):
#        if reco_crit_vel_test_vector[i]:
#            iv = iv + reco_KE_shr_vector[i]
            
#    return iv            
    

##############################################################################
# Calculate Quasi-Elastic Neutrino Energy Electron Scalar & Vector Versions

def get_reco_EvQEle(shr_energy_cali, shr_pz, shr_energy, nshr_visible):
    
    Me = 0.511e-3
    Mp = 0.9383
    Mn = 0.9396
    Eb = 0.0285
    
    corr_shr_energy_cali = shr_energy_cali/0.83
#    corr_shr_energy_second_cali = shr_energy_second_cali/0.83
#    corr_shr_second_pz = shr_second_pz*corr_shr_energy_second_cali/shr_energy_second
    
    if nshr_visible > 0:
        corr_shr_pz = shr_pz*corr_shr_energy_cali/shr_energy
        recoEvQEle = ((corr_shr_energy_cali * (Mn - Eb)) + (0.5 * (Mp**2 - (Mn - Eb)**2 - Me**2)))/((Mn - Eb) + corr_shr_pz - corr_shr_energy_cali)
        return recoEvQEle

#    elif nshr_visible > 1:
#        recoEvQEle1 = ((corr_shr_energy_cali * (Mn - Eb)) + (0.5 * (Mp**2 - (Mn - Eb)**2 - Me**2)))/((Mn - Eb) + corr_shr_pz - corr_shr_energy_cali)
#        recoEvQEle2 = ((corr_shr_energy_second_cali * (Mn - Eb)) + (0.5 * (Mp**2 - (Mn - Eb)**2 - Me**2)))/((Mn - Eb) + corr_shr_second_pz - corr_shr_energy_second_cali)
#        recoEvQEle = recoEvQEle1 + recoEvQEle2
#        return recoEvQEle


##############################################################################
# Add a column to the dataframe indicating whether event passed the 1muNp selection

def process_oLEE_reco(up,df):

    # Load any branches not already loaded
    #df["pfp_generation_v"] = up.array("pfp_generation_v")
    #df["trk_score_v"] = up.array("trk_score_v")
    #df["trk_distance_v"] = up.array("trk_distance_v")
    #df["trk_len_v"] = up.array("trk_len_v")
    #df["trk_llr_pid_score_v"] = up.array("trk_llr_pid_score_v")
    df["trk_sce_start_x_v"] = up.array("trk_sce_start_x_v")
    df["trk_sce_start_y_v"] = up.array("trk_sce_start_y_v")
    df["trk_sce_start_z_v"] = up.array("trk_sce_start_z_v")
    df["trk_sce_end_x_v"] = up.array("trk_sce_end_x_v")
    df["trk_sce_end_y_v"] = up.array("trk_sce_end_y_v")
    df["trk_sce_end_z_v"] = up.array("trk_sce_end_z_v")
    #df["trk_range_muon_mom_v"] = up.array("trk_range_muon_mom_v")
    #df["trk_mcs_muon_mom_v"] = up.array("trk_mcs_muon_mom_v")
    #df["trk_energy_proton_v"] = up.array("trk_energy_proton_v")
    #df["trk_dir_x_v"] = up.array("trk_dir_x_v")
    #df["trk_dir_y_v"] = up.array("trk_dir_y_v")
    #df["trk_dir_z_v"] = up.array("trk_dir_z_v")
    #df["backtracked_pdg"] = up.array("backtracked_pdg")
    
    df["reco_not_dead_region"] = df.apply(lambda x: (reco_not_dead_region (x["reco_nu_vtx_sce_x"], x["reco_nu_vtx_sce_y"], x["reco_nu_vtx_sce_z"])), axis=1)
    
    #df["MuonCandidateIdx"] = df.apply(lambda x: (find_muon_candidate(x["pfp_generation_v"],x["trk_score_v"],x["trk_distance_v"],x["trk_len_v"],x["trk_llr_pid_score_v"])),axis=1)
    
    #df["ProtonCandidateIdx"] = df.apply(lambda x: (find_proton_candidates(x["MuonCandidateIdx"],x["pfp_generation_v"],x["trk_score_v"],x["trk_len_v"],x["trk_llr_pid_score_v"])),axis=1)
    
    #df["RecoProtonMomentum"] = df.apply(lambda x: (get_reco_proton_mom_v(x["ProtonCandidateIdx"],x["trk_energy_proton_v"])),axis=1)
    
    #df["RecoProtonE"] = df.apply(lambda x: (get_reco_proton_E_v(x["RecoProtonMomentum"])),axis=1)
    
    df["nshr_visible"] = df.apply(lambda x: (get_nshr_visible(x["shr_energy_cali"])),axis=1) #,x["shr_energy_second_cali"])),axis=1)
    
    #df["reco_vel_shr"] = df.apply(lambda x: (get_reco_vel_shr(x["shr_energy_cali"],x["shr_px"],x["shr_py"],x["shr_pz"],x["shr_energy"])),axis=1)
    
    #df["reco_crit_vel_test_vector"] = df.apply(lambda x: (reco_crit_vel_test_vector(x["reco_vel_shr_vector"])),axis=1)

    #df["reco_crit_vel_slice_bool"] = df.apply(lambda x: (reco_crit_vel_slice_bool(x["reco_crit_vel_test_vector"],x["reco_not_dead_region"])),axis=1)
    
    df["reco_KE_shr"] = df.apply(lambda x: (get_reco_KE_shr(x["shr_energy_cali"],x["shr_px"],x["shr_py"],x["shr_pz"],x["shr_energy"],x["nshr_visible"])), axis=1)
    
    #df["reco_SubChrnkv_E"] = df.apply(lambda x: (get_reco_SubChrnkv_E(x["reco_crit_vel_test_vector"],x["reco_KE_shr_vector"])),axis=1)
    
    #df["reco_AboveChrnkv_E"] = df.apply(lambda x: (get_reco_AboveChrnkv_E(x["reco_crit_vel_test_vector"],x["reco_KE_shr_vector"])),axis=1)
    
    df["reco_EvQEle"] = df.apply(lambda x: (get_reco_EvQEle(x["shr_energy_cali"],x["shr_pz"],x["shr_energy"],x["nshr_visible"])),axis=1)
    
    #Drop temporary data from dataframes
    
    return df
    
    
    
    
    
    
    
    
    
    
    
