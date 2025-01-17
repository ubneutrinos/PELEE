import numpy as np
import math
from numu_tki import tki_calculators
import sys
sys.path.append("../")
from microfit import selections as sel

# Functions for setting the signal definition selection in reco variables and adding useful variables for the CC1e1P selection
# (original framework in Root/C++ developed by S Gardiner, re-written into the python PeLEE framework by C Thorpe)
# Author: M Moudgalya

################################################################################
# Using the existing selections framework to make the boolean variable for the 
# reco signal events, to be used in Gardiner's unfolding framework later on

selection = "OnePL_new"
preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

################################################################################
# Fiducial volume cut is already applied by preselection filter - repeated here
# for completeness

# Definitions from SG's code
'''
FV_X_MIN = 21.5
FV_X_MAX = 234.85
FV_Y_MIN = -95.0
FV_Y_MAX = 95.0
FV_Z_MIN = 21.5
FV_Z_MAX = 966.8
DEAD_Z_MIN = 10000 # SG's code does not cut dead region 
DEAD_Z_MAX = 10000
'''

# Definitions in PeLEE technote and 'selected' variable
# https://github.com/ubneutrinos/searchingfornues/blob/0489ac5457335a553a3bab54ee5d7ba91734adf0/Selection/SelectionTools/CC0piNpSelection_tool.cc#L97-L102
# https://github.com/ubneutrinos/searchingfornues/blob/889002e5ec93b567265c3af8c178172363200490/Selection/SelectionTools/CC0piNpSelection_tool.cc#L415-L420
FV_X_MIN =   10.0
FV_X_MAX =  246.4
FV_Y_MIN = -101.5
FV_Y_MAX =  101.5
FV_Z_MIN =   10.0
FV_Z_MAX =  986.8
DEAD_Z_MIN = 675 # SG's code does not cut dead region 
DEAD_Z_MAX = 775

def sel_reco_vertex_in_FV(reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z):
    
    return reco_nu_vtx_sce_x > FV_X_MIN and reco_nu_vtx_sce_x < FV_X_MAX and\
           reco_nu_vtx_sce_y > FV_Y_MIN and reco_nu_vtx_sce_y < FV_Y_MAX and\
           reco_nu_vtx_sce_z > FV_Z_MIN and reco_nu_vtx_sce_z < FV_Z_MAX and\
           not (reco_nu_vtx_sce_z > DEAD_Z_MIN and reco_nu_vtx_sce_z < DEAD_Z_MAX)

################################################################################
# Point located in proton containment volume (PCV) - same as Fiducial volume for now
# Helper function

PCV_X_MIN =   10.0
PCV_X_MAX =  246.4
PCV_Y_MIN = -101.5
PCV_Y_MAX =  101.5
PCV_Z_MIN =   10.0
PCV_Z_MAX = 986.8

def in_proton_containment_vol(x,y,z):
    
    return x > PCV_X_MIN and x < PCV_X_MAX and\
           y > PCV_Y_MIN and y < PCV_Y_MAX and\
           z > PCV_Z_MIN and z < PCV_Z_MAX

################################################################################
# Make vector indicating if tracks/showers are in the containment volume (i.e. checking end is contained)
# Helper function

def is_contained_v(track_end_sce_x_v, track_end_sce_y_v, track_end_sce_z_v):

    contained = []
    for i in range(0,len(track_end_sce_x_v)):
         contained.append(in_proton_containment_vol(track_end_sce_x_v[i], track_end_sce_y_v[i], track_end_sce_z_v[i]))

    return contained

################################################################################
# Check track/shower is contained
# Helper function

def is_contained(Idx, IsContained_v):
    if Idx == -1:
        return False
    return IsContained_v[Idx]

################################################################################
# Returns true if all tracks/showers are contained (i.e. checking generation=2 and start is contained)
# Helper function

def pfp_starts_in_PCV_v(trk_sce_start_x_v, trk_sce_start_y_v, trk_sce_start_z_v):
    
    contained = []
    for i in range(0,len(trk_sce_start_x_v)):
         contained.append(in_proton_containment_vol(trk_sce_start_x_v[i], trk_sce_start_y_v[i], trk_sce_start_z_v[i]))

    return contained

################################################################################
# Returns whether the track/shower passes momentum cuts
# Currently not using upper momentum threshold
# Helper function

proton_p_min = 0.3 #GeV #0.239
proton_p_max = 3.0
proton_mass = 0.939
proton_E_min = np.sqrt(proton_p_min**2 + proton_mass**2)
proton_E_max = np.sqrt(proton_p_max**2 + proton_mass**2)

#def pass_mom_cut(RecoMomentum,CUT_LOW,CUT_HIGH):
def pass_mom_cut(RecoMomentum,CUT_LOW):
    if math.isnan(RecoMomentum):
        return False
    
    #return CUT_LOW < RecoMomentum < CUT_HIGH 
    return RecoMomentum > CUT_LOW

################################################################################
# Get indices of reconstructed showers with a starting point within FV

TRACK_SCORE_CUT = 0.5

def reco_showers_v(pfp_generation_v, trk_score_v, pfp_starts_in_PCV_v):
    idx = []
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] == 2 and trk_score_v[i] < TRACK_SCORE_CUT and pfp_starts_in_PCV_v[i]:
            idx.append(i)
    
    return idx

################################################################################
# Get number of reconstructed showers with a starting point within FV

def n_reco_showers(reco_showers_v):

    return len(reco_showers_v)

################################################################################
# Get (electron) shower candidate index above threshold

elec_p_min = 0. #GeV
elec_p_max = 1.2 #GeV
elec_mass = 0.511e-3 #GeV
elec_E_min = np.sqrt(elec_p_min**2 + elec_mass**2)
elec_E_max = np.sqrt(elec_p_max**2 + elec_mass**2) # Upper limit currently unused

def reco_elec_candidate_idx(reco_showers_v, shr_energy_cali):

    if len(reco_showers_v) == 1 and shr_energy_cali > elec_E_min:
    #if len(reco_showers_v) == 1 and elec_E_min < shr_energy_cali < elec_E_max:
        return reco_showers_v[0]
    
    return -1

################################################################################
# Find index of the longest track (no pid)

def find_longest_trk_len_idx(pfp_generation_v, trk_len_v):
    
    longest_idx=-1
    longest_len=-1
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] == 2 and trk_len_v[i] > longest_len:
            longest_idx = i
            longest_len = trk_len_v[i]

    return longest_idx

# ################################################################################
# # Make a list of the indices of the protons candidates fully contained in the FV

# DEFAULT_PROTON_PID_CUT = 0.02

# def find_proton_candidates(reco_elec_candidate_idx, longest_trk_len_idx, pfp_generation_v, trk_score_v, trk_len_v, trk_llr_pid_score_v, is_contained_v, pfp_starts_in_PCV_v):

#     PID_CUT = 0.015 * trk_len_v[longest_trk_len_idx] + DEFAULT_PROTON_PID_CUT  # for xsec selection as suggested by Elena
#     proton_candidate_idx_v=[]
#     for i in range(0,len(pfp_generation_v)):
#         if pfp_generation_v[i] != 2 or i == reco_elec_candidate_idx or\
#            trk_score_v[i] < TRACK_SCORE_CUT or trk_len_v[i] < 0.0 or\
#            trk_llr_pid_score_v[i] > DEFAULT_PROTON_PID_CUT or\
#            not (is_contained_v[i] and pfp_starts_in_PCV_v[i]): continue
        
#         proton_candidate_idx_v.append(i) 

#     return proton_candidate_idx_v

################################################################################
# Make a list of the indices of the protons candidates fully contained in the FV

DEFAULT_PROTON_PID_CUT = 0.02

def find_proton_candidates(reco_elec_candidate_idx, longest_trk_len_idx, pfp_generation_v, trk_score_v, trk_len_v, trk_llr_pid_score_v, is_contained_v, pfp_starts_in_PCV_v):

    if longest_trk_len_idx == -1:
        PID_CUT = np.nan
    else:
        PID_CUT = 0.015 * trk_len_v[longest_trk_len_idx] + DEFAULT_PROTON_PID_CUT  # for xsec selection as suggested by Elena
    proton_candidate_idx_v=[]
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] == 2 and i != reco_elec_candidate_idx and\
           trk_score_v[i] > TRACK_SCORE_CUT and trk_len_v[i] >= 0.0 and\
           trk_llr_pid_score_v[i] < DEFAULT_PROTON_PID_CUT and\
           is_contained_v[i] and pfp_starts_in_PCV_v[i]:
           
           proton_candidate_idx_v.append(i) 

    return proton_candidate_idx_v

################################################################################
# Count the number of proton tracks fully contained in the FV

def n_reco_protons(ProtonCandidateIdx_v):
    return len(ProtonCandidateIdx_v)

################################################################################
# Find index of the longest proton track

def find_leading_proton_candidate(ProtonCandidateIdx_v, trk_len_v):
    
    longest_idx=-1
    longest_len=-1
    for i in range(0,len(trk_len_v)):
        if i in ProtonCandidateIdx_v and trk_len_v[i] > longest_len:
            longest_idx = i
            longest_len = trk_len_v[i]

    return longest_idx

################################################################################
# Helper function to return an element from a vector given the index

def return_elm_from_vec_idx(Idx, vector_var):
    if Idx == -1:
        return np.nan
    if vector_var.size > 0:
        return vector_var[Idx]
    else:
        return np.nan

################################################################################
# Get momentum of reconstructed leading proton

MASS_PROTON = 0.939 #0.93827

def get_reco_proton_mom(LeadProtonIdx, trk_energy_proton_v):

    if LeadProtonIdx == -1:
        return np.nan
    ke = trk_energy_proton_v[LeadProtonIdx]
    return np.sqrt(ke**2 + (2*MASS_PROTON*ke))

################################################################################
# Get momentum component of the reconstructed leading proton

def get_reco_proton_mom_comp(LeadProtonIdx, trk_energy_proton_v, trk_dir_v):

    if LeadProtonIdx == -1:
        return np.nan

    ke = trk_energy_proton_v[LeadProtonIdx]
    return np.sqrt(ke**2 + (2*MASS_PROTON*ke)) * trk_dir_v[LeadProtonIdx]

################################################################################
# Get total energy of reconstructed leading proton

def get_reco_proton_E(RecoLeadProtonMomentum):

    return np.sqrt(RecoLeadProtonMomentum**2 + MASS_PROTON**2)

################################################################################
# Get kinetic energy of reconstructed leading proton

def get_reco_proton_KE(LeadProtonIdx, trk_energy_proton_v):

    if LeadProtonIdx == -1:
        return np.nan

    return trk_energy_proton_v[LeadProtonIdx]

################################################################################
# Get momentum of reconstructed protons. Returns a list

def get_reco_proton_mom_v(ProtonCandidateIdx_v, trk_energy_proton_v):

    mom_v = []
    for i in range(0,len(ProtonCandidateIdx_v)):
        ke = trk_energy_proton_v[ProtonCandidateIdx_v[i]]
        mom_v.append(np.sqrt(ke**2 + (2*MASS_PROTON*ke)))

    return mom_v

################################################################################
# Get momentum of reconstructed protons. Returns a list

def get_reco_proton_mom_comp_v(ProtonCandidateIdx_v, trk_energy_proton_v, trk_dir_v):

    mom_v = []
    for i in range(0,len(ProtonCandidateIdx_v)):
        ke = trk_energy_proton_v[ProtonCandidateIdx_v[i]]
        mom_v.append(np.sqrt(ke**2 + (2*MASS_PROTON*ke))*trk_dir_v[ProtonCandidateIdx_v[i]])

    return mom_v

################################################################################
# Get momentum of reconstructed protons. Returns a list

def get_reco_proton_E_v(RecoProtonMomentum_v):

    E_v = []
    for i in range(0,len(RecoProtonMomentum_v)):
        E_v.append(np.sqrt(RecoProtonMomentum_v[i]**2 + MASS_PROTON**2)) 

    return E_v

################################################################################
# Helper function to return the magnitude of a numpy array

def get_magnitude(arr):
    return np.linalg.norm(arr)

################################################################################
# Calculating conversion distance

def get_conversion_dist(Idx,reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z,trk_sce_start_x_v, trk_sce_start_y_v, trk_sce_start_z_v):
    
    if Idx == -1:
        return np.nan

    dx = trk_sce_start_x_v[Idx] - reco_nu_vtx_sce_x
    dy = trk_sce_start_y_v[Idx] - reco_nu_vtx_sce_y
    dz = trk_sce_start_z_v[Idx] - reco_nu_vtx_sce_z

    return np.sqrt(dx**2 + dy**2 + dz**2)

################################################################################
# Add a column indicating if the events belong to the 1e1p selection

def apply_selection_1e1p_tki(up,df):

    # Load the extra branches needed 
    df["trk_dir_x_v"] = up.array("trk_dir_x_v")
    df["trk_dir_y_v"] = up.array("trk_dir_y_v")
    df["trk_dir_z_v"] = up.array("trk_dir_z_v")
    df["trk_energy_proton_v"] = up.array("trk_energy_proton_v")
    # df['shr_energy_cali'] = up.array('shr_energy_cali')
    # df['shr_energy'] = up.array('shr_energy')
    # df['shr_px'] = up.array('shr_px')
    # df['shr_py'] = up.array('shr_py')
    # df['shr_pz'] = up.array('shr_pz')
    df["pfp_generation_v"] = up.array("pfp_generation_v")
    df["trk_score_v"] = up.array("trk_score_v")
    df["trk_len_v"] = up.array("trk_len_v")
    df["trk_llr_pid_score_v"] = up.array("trk_llr_pid_score_v")
    df["shr_llr_pid_score_v"] = up.array("shr_llr_pid_score_v")
    df["backtracked_pdg"] = up.array("backtracked_pdg")
    df["trk_sce_start_x_v"] = up.array("trk_sce_start_x_v")
    df["trk_sce_start_y_v"] = up.array("trk_sce_start_y_v")
    df["trk_sce_start_z_v"] = up.array("trk_sce_start_z_v")
    df["trk_sce_end_x_v"] = up.array("trk_sce_end_x_v")
    df["trk_sce_end_y_v"] = up.array("trk_sce_end_y_v")
    df["trk_sce_end_z_v"] = up.array("trk_sce_end_z_v")
    #df["trk_trunk_dEdx_y_v"] = up.array("trk_trunk_dEdx_y_v")
    # df["pi0_radlen1"] = up.array("pi0_radlen1")
    # df["pi0_radlen2"] = up.array("pi0_radlen2")

    df["InFV_reco"] = df.apply(lambda x: (sel_reco_vertex_in_FV(x["reco_nu_vtx_sce_x"], x["reco_nu_vtx_sce_y"], x["reco_nu_vtx_sce_z"])), axis=1)
    df["PFPStartsInPCV_v"] = df.apply(lambda x: (pfp_starts_in_PCV_v(x["trk_sce_start_x_v"], x["trk_sce_start_y_v"], x["trk_sce_start_z_v"])), axis=1)
    df["IsContained_v"] = df.apply(lambda x: (is_contained_v(x["trk_sce_end_x_v"], x["trk_sce_end_y_v"], x["trk_sce_end_z_v"])),axis=1)

    # Making corrections to the electron energy and momentum variables
    df['RecoElecMomX'] = df['shr_px'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
    df['RecoElecMomY'] = df['shr_py'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
    df['RecoElecMomZ'] = df['shr_pz'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
    df['RecoElecE'] = df['shr_energy_cali'] * 1/0.83
    df["RecoElecKE"] = df["RecoElecE"] - elec_mass
    df['RecoElecModMom'] = np.sqrt((df['RecoElecMomX'])**2 + (df['RecoElecMomY'])**2 + (df['RecoElecMomZ'])**2)

    df["RecoElecPassMomCut"] = df.apply(lambda x: (pass_mom_cut(x["RecoElecModMom"], elec_p_min)),axis=1) 

    df["RecoShowersIndices"] = df.apply(lambda x: (reco_showers_v(x["pfp_generation_v"], x["trk_score_v"], x["PFPStartsInPCV_v"])), axis=1)
    df["RecoElectronCandidateIdx"] = df.apply(lambda x: (reco_elec_candidate_idx(x["RecoShowersIndices"], x["shr_energy_cali"])), axis=1)
    df["RecoElectronCandidate_shr_pid"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoElectronCandidateIdx"], x["shr_llr_pid_score_v"])), axis=1)
    df["RecoElectronCandidate_backtracked_pdg"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoElectronCandidateIdx"], x["backtracked_pdg"])), axis=1)
    df["n_reco_showers"] = df.apply(lambda x: (n_reco_showers(x["RecoShowersIndices"])), axis=1)
    df["ElectronFullyContained"] = df.apply(lambda x: (is_contained(x["RecoElectronCandidateIdx"], x["IsContained_v"])),axis=1)
    df["RecoElectron_conversion_dist"] = df.apply(lambda x: (get_conversion_dist(x["RecoElectronCandidateIdx"], x["reco_nu_vtx_sce_x"], x["reco_nu_vtx_sce_y"], x["reco_nu_vtx_sce_z"], x["trk_sce_start_x_v"], x["trk_sce_start_y_v"], x["trk_sce_start_z_v"])), axis=1)

    df["longest_trk_len_idx"] = df.apply(lambda x: (find_longest_trk_len_idx(x["pfp_generation_v"], x["trk_len_v"])), axis=1)
    df["longest_trk_len"] = df.apply(lambda x: (return_elm_from_vec_idx(x["longest_trk_len_idx"], x["trk_len_v"])), axis=1)
    #df["longest_trk_trunk_dEdx_y"] = df.apply(lambda x: (return_elm_from_vec_idx(x["longest_trk_len_idx"], x["trk_trunk_dEdx_y_v"])), axis=1)
    df["RecoProtonIndices"] = df.apply(lambda x: (find_proton_candidates(x["RecoElectronCandidateIdx"], x["longest_trk_len_idx"], x["pfp_generation_v"], x["trk_score_v"], x["trk_len_v"], x["trk_llr_pid_score_v"], x["IsContained_v"], x["PFPStartsInPCV_v"])), axis=1)
    df["RecoLeadProtonCandidateIdx"] = df.apply(lambda x: (find_leading_proton_candidate(x["RecoProtonIndices"], x["trk_len_v"])), axis=1)
    df["RecoLeadProtonCandidate_trk_pid"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoLeadProtonCandidateIdx"], x["trk_llr_pid_score_v"])), axis=1)
    df["RecoLeadProtonCandidate_backtracked_pdg"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoLeadProtonCandidateIdx"], x["backtracked_pdg"])), axis=1)
    df["RecoLeadProton_trk_len"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoLeadProtonCandidateIdx"], x["trk_len_v"])), axis=1)
    #df["RecoLeadProton_trk_trunk_dEdx_y"] = df.apply(lambda x: (return_elm_from_vec_idx(x["RecoLeadProtonCandidateIdx"], x["trk_trunk_dEdx_y_v"])), axis=1)
    df["n_reco_tracks"] = df.apply(lambda x: (n_reco_protons(x["RecoProtonIndices"])), axis=1)
    #df["RecoLeadProton_dEdx_y_per_trklen"] = df["RecoLeadProton_trk_trunk_dEdx_y"] / df["RecoLeadProton_trk_len"]
    #df["longest_trk_dEdx_y_per_trklen"] = df["longest_trk_trunk_dEdx_y"] / df["longest_trk_len"]

    df['RecoLeadProtonModMom'] = df.apply(lambda x: (get_reco_proton_mom(x["RecoLeadProtonCandidateIdx"], x["trk_energy_proton_v"])), axis=1)
    df["RecoLeadProtonMomX"] = df.apply(lambda x: (get_reco_proton_mom_comp(x["RecoLeadProtonCandidateIdx"], x["trk_energy_proton_v"], x["trk_dir_x_v"])), axis=1)
    df["RecoLeadProtonMomY"] = df.apply(lambda x: (get_reco_proton_mom_comp(x["RecoLeadProtonCandidateIdx"], x["trk_energy_proton_v"], x["trk_dir_y_v"])), axis=1)
    df["RecoLeadProtonMomZ"] = df.apply(lambda x: (get_reco_proton_mom_comp(x["RecoLeadProtonCandidateIdx"], x["trk_energy_proton_v"], x["trk_dir_z_v"])), axis=1)
    df["RecoLeadProtonE"] = df.apply(lambda x: (get_reco_proton_E(x["RecoLeadProtonModMom"])),axis=1)
    df["RecoLeadProtonKE"] = df.apply(lambda x: (get_reco_proton_KE(x["RecoLeadProtonCandidateIdx"], x["trk_energy_proton_v"])),axis=1)

    df["RecoLeadProtonPassMomCut"] = df.apply(lambda x: (pass_mom_cut(x["RecoLeadProtonModMom"], proton_p_min)),axis=1)
    
    # Set the reco signal definition (no additional cuts)
    #nue_cc0piNp = ((df["RecoElectronCandidateIdx"] != -1) & (df["RecoLeadProtonCandidateIdx"] != -1) & (df["InFV_reco"] == True) & (df["RecoElecPassMomCut"] == True) & (df["RecoLeadProtonPassMomCut"] == True))
    nue_cc0pi1p = ((df["RecoElectronCandidateIdx"] != -1) & (df["RecoLeadProtonCandidateIdx"] != -1) & (df["InFV_reco"] == True) & (df["RecoElecPassMomCut"] == True) & (df["RecoLeadProtonPassMomCut"] == True) & (df["n_reco_tracks"] == 1) & (df["n_reco_showers"] == 1))
    
    # df.loc[nue_cc0piNp, "Signal_1eNp"] = True
    # df.loc[~nue_cc0piNp, "Signal_1eNp"] = False
    
    df.loc[nue_cc0pi1p, "Sel_1e1p"] = True
    df.loc[~nue_cc0pi1p, "Sel_1e1p"] = False

    # Making the boolean variable which incorporates selection cuts along with Sel_1e1p to be used in Gardiner's unfolding framework later on
    temp_df = df.query(query, engine='python')
    df["sel_1e1p_w_cuts"] = False
    df.loc[temp_df.index, "sel_1e1p_w_cuts"] = True

    print("Calc reco TKI variables for leading proton only")

    df["RecoDeltaPT"] = df.apply(lambda x: (tki_calculators.delta_pT(x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    #df["RecoDeltaPhiT"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df["RecoDeltaAlphaT"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df['RecoDeltaAlphaT'] = np.degrees(df['RecoDeltaAlphaT'])

    print("Calc reco GKI variables for leading proton only")
    df["RecoPN"] = df.apply(lambda x: (tki_calculators.pn(x["RecoElecE"],x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonE"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df["RecoAlpha3D"] = df.apply(lambda x: (tki_calculators.alpha_3D(x["RecoElecE"],x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonE"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df["RecoAlpha3D"] = np.degrees(df["RecoAlpha3D"])
    #df["RecoPhi3D"] = df.apply(lambda x: (tki_calculators.phi_3D(x["RecoElecE"],x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonE"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df["RecoECal"] = df.apply(lambda x: (tki_calculators.Ecal(x["RecoElecE"],x["RecoLeadProtonE"])),axis=1)
    df["Reco_vec_q"] = df.apply(lambda x: (tki_calculators.vec_q(x["RecoElecE"],x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonE"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)
    df["Reco_mag_q"] = df.apply(lambda x: (get_magnitude(x["Reco_vec_q"])), axis=1)
    df["RecoPL"] = df.apply(lambda x: (tki_calculators.pL(x["RecoElecE"],x["RecoElecMomX"],x["RecoElecMomY"],x["RecoElecMomZ"],x["RecoLeadProtonE"],x["RecoLeadProtonMomX"],x["RecoLeadProtonMomY"],x["RecoLeadProtonMomZ"])),axis=1)


    # Drop all of the temporary columns added to the dataframe to save space
    df.drop("pfp_generation_v",inplace=True,axis=1)
    df.drop("trk_score_v",inplace=True,axis=1)
    df.drop("trk_len_v",inplace=True,axis=1)
    df.drop("trk_llr_pid_score_v",inplace=True,axis=1)
    df.drop("trk_sce_start_x_v",inplace=True,axis=1)
    df.drop("trk_sce_start_y_v",inplace=True,axis=1)
    df.drop("trk_sce_start_z_v",inplace=True,axis=1)
    df.drop("trk_sce_end_x_v",inplace=True,axis=1)
    df.drop("trk_sce_end_y_v",inplace=True,axis=1)
    df.drop("trk_sce_end_z_v",inplace=True,axis=1)
    df.drop("trk_energy_proton_v",inplace=True,axis=1)
    df.drop("trk_dir_x_v",inplace=True,axis=1)
    df.drop("trk_dir_y_v",inplace=True,axis=1)
    df.drop("trk_dir_z_v",inplace=True,axis=1)
    #df.drop("trk_trunk_dEdx_y_v",inplace=True,axis=1)

    return df
