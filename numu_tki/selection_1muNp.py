import numpy as np

# Selection Algorithms for the 1muNp channel
# Re-using selection developed by Steven Gardiner
# Author: C Thorpe

TRACK_SCORE_CUT = 0.5

MUON_P_MIN_MOM_CUT = 0.100
MUON_P_MAX_MOM_CUT = 1.200

LEAD_P_MIN_MOM_CUT = 0.250
LEAD_P_MAX_MOM_CUT = 1.

################################################################################
# Fiducial volume cut is already applied by preselection filter - repeated here
# for completeness

def sel_reco_vertex_in_FV(reco_nu_vtx_sce_x,reco_nu_vtx_sce_y,reco_nu_vtx_sce_z):
    
    return reco_nu_vtx_sce_x > 5.0 and reco_nu_vtx_sce_x < 251.0 and\
           abs(reco_nu_vtx_sce_y) < 111.0 and\
           reco_nu_vtx_sce_z > 20.0 and reco_nu_vtx_sce_z < 986.0 and\
           not (reco_nu_vtx_sce_z > 675 and reco_nu_vtx_sce_z < 775) 

################################################################################
# Point located in proton containment volume

PCV_X_MIN =   10.
PCV_X_MAX =  246.35
PCV_Y_MIN = -106.5
PCV_Y_MAX =  106.5
PCV_Z_MIN =   10.
PCV_Z_MAX = 1026.8

def in_proton_containment_vol(x,y,z):
    
    return x > PCV_X_MIN and x < PCV_X_MAX and\
           y > PCV_Y_MIN and y < PCV_Y_MAX and\
           z > PCV_Z_MIN and z < PCV_Z_MAX

################################################################################
# Make vector indicating if tracks are in the containedment volume 
def is_contained(track_end_sce_x_v,track_end_sce_y_v,track_end_sce_z_v):

    contained = []
    for i in range(0,len(track_end_sce_x_v)):
         contained.append(in_proton_containment_vol(track_end_sce_x_v[i],track_end_sce_y_v[i],track_end_sce_z_v[i]))

    return contained

################################################################################
# Apply generic numu cc selection

TOPOLOGICAL_SCORE_CUT=0.1

def pass_numu_CC_selection(topological_score,pfp_generation_v,track_start_sce_x_v,track_start_sce_y_v,track_start_sce_z_v,MuonCandidateIdx1muNp):
    
    if topological_score > TOPOLOGICAL_SCORE_CUT: return False
       
    sel_pfp_starts_in_PCV = True;
    
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] != 2: continue         
        sel_pfp_starts_in_PCV = in_proton_containment_vol(track_start_sce_x_v[i],track_start_sce_y_v[i],track_start_sce_z_v[i]) and sel_pfp_starts_in_PCV                
 
    if sel_pfp_starts_in_PCV == False: return False

    return MuonCandidateIdx1muNp != -1

################################################################################

MUON_TRACK_SCORE_CUT = 0.8
MUON_VTX_DISTANCE_CUT = 4.
MUON_LENGTH_CUT = 10.
MUON_PID_CUT = 0.2

def find_muon_candidate(pfp_generation_v,trk_score_v,trk_distance_v,trk_len_v,trk_llr_pid_score_v):
   
    index=-1
    highest_score=-1000  
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] != 2 or trk_score_v[i] < MUON_TRACK_SCORE_CUT or\
           trk_distance_v[i] > MUON_VTX_DISTANCE_CUT or trk_len_v[i] < MUON_LENGTH_CUT or\
           trk_llr_pid_score_v[i] < MUON_PID_CUT: continue   
        if trk_llr_pid_score_v[i] > highest_score:
            index=i
            highest_score=trk_llr_pid_score_v[i] 

    return index

################################################################################
# Get number of reconstructed showers 

def sel_reco_showers(pfp_generation_v,trk_score_v):
    showers=0
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] == 2 and trk_score_v[i] < TRACK_SCORE_CUT: showers=showers+1
    
    return showers 

################################################################################
# Check muon track is contained

def is_muon_contianed(MuonCandidateIdx_1muNp,IsContained_1muNp):
    return IsContained_1muNp[MuonCandidateIdx_1muNp]

################################################################################
# Get momentum of reconstructed muon track

def get_reco_muon_mom(MuonCandidateIdx_1muNp,trk_range_muon_mom_v,trk_mcs_muon_mom_v,MuonContained_1muNp):

    if MuonCandidateIdx_1muNp == -1: return np.nan

    if MuonCandidateIdx_1muNp: return trk_range_muon_mom_v[MuonCandidateIdx_1muNp]
    else: return trk_mcs_muon_mom_v[MuonCandidateIdx_1muNp] 

################################################################################
# Muon track momentun cuts 

def pass_mom_cut(RecoMuonMomentum_1muNp,CUT_LOW,CUT_HIGH):
    return RecoMuonMomentum_1muNp > CUT_LOW and RecoMuonMomentum_1muNp < CUT_HIGH 

################################################################################

MUON_MOM_QUALITY_CUT = 0.25 

def pass_muon_qual_cut(MuonCandidateIdx_1muNp,trk_range_muon_mom_v,trk_mcs_muon_mom_v):
    return abs(trk_range_muon_mom_v[MuonCandidateIdx_1muNp] - trk_mcs_muon_mom_v[MuonCandidateIdx_1muNp])/trk_range_muon_mom_v[MuonCandidateIdx_1muNp] < MUON_MOM_QUALITY_CUT

################################################################################
# Make a list of the indices of the protons candidates

DEFAULT_PROTON_PID_CUT = 0.2

def find_proton_candidates(MuonCandidateIdx_1muNp,pfp_generation_v,trk_score_v,trk_len_v,trk_llr_pid_score_v,IsContained_1muNp):

    proton_candidate_idx=[]
    for i in range(0,len(pfp_generation_v)):
        if pfp_generation_v[i] != 2 or i == MuonCandidateIdx_1muNp or\
           trk_score_v[i] < TRACK_SCORE_CUT or trk_len_v[i] < 0.0 or\
           trk_llr_pid_score_v[i] > DEFAULT_PROTON_PID_CUT or\
           not IsContained_1muNp[i]: continue
        
        proton_candidate_idx.append(i) 

    return proton_candidate_idx

################################################################################

def find_leading_proton_candidate(ProtonCandidateIdx_1muNp,trk_len_v):
    
    longest_idx=-1
    longest_len=-1
    for i in range(0,len(trk_len_v)):
        if i in ProtonCandidateIdx_1muNp and trk_len_v[i] > longest_len:
            longest_idx = i
            longest_len = trk_len_v[i]

    return longest_idx

################################################################################
# Apply the whole selection

def is_sel_1muNp(PassNuMuCCSelection_1muNp,NoRecoShowers_1muNp,MuonContained_1muNp,PassMuonMomentumCut_1muNp,PassMuonQualCut_1muNp,LeadingProtonPassMomentumCut_1muNp):
    return PassNuMuCCSelection_1muNp and NoRecoShowers_1muNp and MuonContained_1muNp and\
           PassMuonMomentumCut_1muNp and PassMuonQualCut_1muNp and LeadingProtonPassMomentumCut_1muNp

################################################################################
# Add a column to the dataframe indicating whether event passed the 1muNp selection

def apply_selection_1muNp(df,filter=False):

    print("Applying SG's selection")

    df["MuonCandidateIdx_1muNp"] = df.apply(lambda x: (find_muon_candidate(x["pfp_generation_v"],x["trk_score_v"],x["trk_distance_v"],x["trk_len_v"],x["trk_llr_pid_score_v"])),axis=1)
    if filter: df = df.query("MuonCandidateIdx_1muNp != -1")
  
    df["IsContained_1muNp"] = df.apply(lambda x: (is_contained(x["trk_sce_end_x_v"],x["trk_sce_end_y_v"],x["trk_sce_end_z_v"])),axis=1)
    df["PassNuMuCCSelection_1muNp"] = df.apply(lambda x: (pass_numu_CC_selection(x["topological_score"],x["pfp_generation_v"],x["trk_sce_start_x_v"],x["trk_sce_start_y_v"],x["trk_sce_start_z_v"],x["MuonCandidateIdx_1muNp"])),axis=1)
    if filter: df = df.query("PassNuMuCCSelection_1muNp == True")    

    df["NoRecoShowers_1muNp"] = df.apply(lambda x: (sel_reco_showers(x["pfp_generation_v"],x["trk_score_v"]) == 0),axis=1)
    if filter: df = df.query("NoRecoShowers_1muNp == True")

    df["MuonContained_1muNp"] = df.apply(lambda x: (is_muon_contianed(x["MuonCandidateIdx_1muNp"],x["IsContained_1muNp"])),axis=1)
    df["RecoMuonMomentum_1muNp"] = df.apply(lambda x: (get_reco_muon_mom(x["MuonCandidateIdx_1muNp"],x["trk_range_muon_mom_v"],x["trk_mcs_muon_mom_v"],x["MuonContained_1muNp"])),axis=1)
    df["PassMuonMomentumCut_1muNp"] = df.apply(lambda x: (pass_mom_cut(x["RecoMuonMomentum_1muNp"],MUON_P_MIN_MOM_CUT,MUON_P_MAX_MOM_CUT)),axis=1) 

    if filter: df = df.query("PassMuonMomentumCut_1muNp == True")

    df["PassMuonQualCut_1muNp"] = df.apply(lambda x: (pass_muon_qual_cut(x["MuonCandidateIdx_1muNp"],x["trk_range_muon_mom_v"],x["trk_mcs_muon_mom_v"])),axis=1)
    if filter: df = df.query("PassMuonQualCut_1muNp == True")

    df["ProtonCandidateIdx_1muNp"] = df.apply(lambda x: (find_proton_candidates(x["MuonCandidateIdx_1muNp"],x["pfp_generation_v"],x["trk_score_v"],x["trk_len_v"],x["trk_llr_pid_score_v"],x["IsContained_1muNp"])),axis=1)
    df["LeadingProtonIdx_1muNp"] = df.apply(lambda x: (find_leading_proton_candidate(x["ProtonCandidateIdx_1muNp"],x["trk_len_v"])),axis=1)
    df["LeadingProtonPassMomentumCut_1muNp"] = df.apply(lambda x: (pass_mom_cut(x["RecoMuonMomentum_1muNp"],LEAD_P_MIN_MOM_CUT,LEAD_P_MAX_MOM_CUT)),axis=1)
    if filter: df = df.query("LeadingProtonPassMomentumCut_1muNp == True")  

    df["sel_CCNp0pi"] = df.apply(lambda x: (is_sel_1muNp(x["PassNuMuCCSelection_1muNp"],x["NoRecoShowers_1muNp"],x["MuonContained_1muNp"],x["PassMuonMomentumCut_1muNp"],x["PassMuonQualCut_1muNp"],x["LeadingProtonPassMomentumCut_1muNp"])),axis=1)
    if filter: df = df.query("sel_CCNp0pi == True")

    


 
    return df  
