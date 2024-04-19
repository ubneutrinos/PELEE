import numpy as np
from numu_tki import tki_calculators

# Functions for setting the signal definition and adding useful variables for the CC1e1P selection
# (original framework in Root/C++ developed by S Gardiner, re-written into the python PeLEE framework by C Thorpe)
# Author: M Moudgalya

################################################################################
# Check there is a final state muon above threshold, and return its index

muon_p_min = 0.1
muon_p_max = 1.2
muon_mass = 0.1057
muon_E_min = np.sqrt(muon_p_min**2 + muon_mass**2)
muon_E_max = np.sqrt(muon_p_max**2 + muon_mass**2) # SG's code imposes an upper limit

def true_muon_idx(mc_pdg,mc_E):

    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 13 and muon_E_min < mc_E[i] < muon_E_max:
            return i

    return -1

################################################################################
# Check there is a final state electron above threshold, and return its index

elec_p_min = 0. #GeV
elec_p_max = 5.0 #GeV
elec_mass = 0.511e-3 #GeV
elec_E_min = np.sqrt(elec_p_min**2 + elec_mass**2)
elec_E_max = np.sqrt(elec_p_max**2 + elec_mass**2) # Upper limit currently unused

def true_elec_idx(mc_pdg,mc_E):

    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 11 and mc_E[i] > elec_E_min:
        #if abs(mc_pdg[i]) == 11 and elec_E_min < mc_E[i] < elec_E_max:
            return i

    return -1

################################################################################
# Final state has one proton above threshold

proton_p_min = 0.300 #GeV
proton_p_max = 3.0
proton_mass = 0.939
proton_E_min = np.sqrt(proton_p_min**2 + proton_mass**2)
proton_E_max = np.sqrt(proton_p_max**2 + proton_mass**2) # Upper limit currently unused

def true_proton_idx(mc_pdg,mc_E):

    idx = []
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_min:
        #if abs(mc_pdg[i]) == 2212 and proton_E_min < mc_E[i] < proton_E_max:
            idx.append(i)

    return idx

################################################################################
# Number of protons above threshold

def n_proton(TrueIdx_v):

    return len(TrueIdx_v) 

################################################################################
# Index of lead proton 

def true_lead_proton_idx(mc_pdg,mc_E):

    lead_idx = -1
    lead_E = -1
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_min and mc_E[i] > lead_E:
        #if abs(mc_pdg[i]) == 2212 and proton_E_min < mc_E[i] < proton_E_max and mc_E[i] > lead_E:
            lead_E = mc_E[i]
            lead_idx = i

    return lead_idx

################################################################################
# Final state has no charged pions above thresold and no pi0

pion_thresh = 0.07 #GeV
pion_mass = 0.1396
pion_E_thresh = np.sqrt(pion_thresh**2 + pion_mass**2)

def has_no_mesons(mc_pdg,mc_E):
    for i in range(0,len(mc_pdg)):
        if (abs(mc_pdg[i]) == 211 and mc_E[i] > pion_E_thresh) or mc_pdg[i] == 111 or abs(mc_pdg[i]) == 321:
            return False 

    return True

################################################################################
# Number of charged pions above threshold

def n_fs_pion(mc_pdg,mc_E):

    n_pion = 0
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 211 and mc_E[i] > pion_E_thresh:
            n_pion += 1

    return n_pion

################################################################################
# Number of neutral pions

def n_fs_pi0(mc_pdg,mc_E):

    n_pi0 = 0
    for i in range(0,len(mc_pdg)):
        if mc_pdg[i] == 111:
            n_pi0 += 1

    return n_pi0

################################################################################
# True primary vertex is in the fiducial volume

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

def in_fiducial_volume(true_nu_vtx_x,true_nu_vtx_y,true_nu_vtx_z):

    return FV_X_MIN < true_nu_vtx_x < FV_X_MAX and\
           FV_Y_MIN < true_nu_vtx_y < FV_Y_MAX and\
           FV_Z_MIN < true_nu_vtx_z < FV_Z_MAX and\
           not (DEAD_Z_MIN < true_nu_vtx_z < DEAD_Z_MAX)

################################################################################
# Component of the momentum of a particle at index TrueIdx

def true_mom(TrueIdx,mc_p):

    if TrueIdx == -1:
        return np.nan
    else:
        return mc_p[TrueIdx]
    
################################################################################
# Component of the total momentum of a group of particles 

def true_mom_tot(TrueIdx_v,mc_p):

    if len(TrueIdx_v) == 0:
        return np.nan

    p = 0
    for i in TrueIdx_v:
        p = p + mc_p[i]    

    return p 

################################################################################
# Vector of momenta of group of particles 

def true_mom_v(TrueIdx_v,mc_p):

    #p = np.array([])
    p = []
    for i in TrueIdx_v:
        #p = np.append(p, mc_p[i])
        p.append(mc_p[i])

    return p 

################################################################################
# Add a column indicating if the events belong to the 1e1p signal
# Add a column to define the topological categories for analysis (based on the 'category' variable)

def set_Signal1e1p(up,df):

    # Load the extra branches needed 
    df["mc_pdg"] = up.array("mc_pdg")
    df["mc_E"] = up.array("mc_E")
    df["mc_px"] = up.array("mc_px")
    df["mc_py"] = up.array("mc_py")
    df["mc_pz"] = up.array("mc_pz")

    df["InFV"] = df.apply(lambda x: (in_fiducial_volume(x["true_nu_vtx_x"],x["true_nu_vtx_y"],x["true_nu_vtx_z"])),axis=1)
    
    df["TrueMuonIdx"] = df.apply(lambda x: (true_muon_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueElecIdx"] = df.apply(lambda x: (true_elec_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueLeadProtonIdx"] = df.apply(lambda x: (true_lead_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    
    df["TrueProtonIdx"] = df.apply(lambda x: (true_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueNProt"] = df.apply(lambda x: (n_proton(x["TrueProtonIdx"])),axis=1)
    df["TrueFSPions"] = df.apply(lambda x: (n_fs_pion(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueFSPi0"] = df.apply(lambda x: (n_fs_pi0(x["mc_pdg"],x["mc_E"])),axis=1)
    df["HasNoMesons"] = df.apply(lambda x: (has_no_mesons(x["mc_pdg"],x["mc_E"])),axis=1)

    df["TrueElecE"] = df.apply(lambda x: (true_mom(x["TrueElecIdx"],x["mc_E"])),axis=1)
    df["TrueElecMomX"] = df.apply(lambda x: (true_mom(x["TrueElecIdx"],x["mc_px"])),axis=1)
    df["TrueElecMomY"] = df.apply(lambda x: (true_mom(x["TrueElecIdx"],x["mc_py"])),axis=1)
    df["TrueElecMomZ"] = df.apply(lambda x: (true_mom(x["TrueElecIdx"],x["mc_pz"])),axis=1)

    df["TrueElecKE"] = df["TrueElecE"] - elec_mass
    df['TrueElecModMom'] = np.sqrt((df['TrueElecMomX'])**2 + (df['TrueElecMomY'])**2 + (df['TrueElecMomZ'])**2)

    df["TrueLeadProtonE"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx"],x["mc_E"])),axis=1)
    df["TrueLeadProtonMomX"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx"],x["mc_px"])),axis=1)
    df["TrueLeadProtonMomY"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx"],x["mc_py"])),axis=1)
    df["TrueLeadProtonMomZ"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx"],x["mc_pz"])),axis=1)

    df["TrueLeadProtonKE"] = df["TrueLeadProtonE"] - proton_mass
    df['TrueLeadProtonModMom'] = np.sqrt((df['TrueLeadProtonMomX'])**2 + (df['TrueLeadProtonMomY'])**2 + (df['TrueLeadProtonMomZ'])**2)
    
    # Set the signal definition
    nue_cc0piNp = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] != -1) & (df["TrueLeadProtonIdx"] != -1) & (df["InFV"] == True) & (df["HasNoMesons"] == True))
    nue_cc0pi1p = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] != -1) & (df["TrueLeadProtonIdx"] != -1) & (df["InFV"] == True) & (df["HasNoMesons"] == True) & (df["TrueNProt"] == 1))
    
    df.loc[nue_cc0piNp, "Signal_1eNp"] = True
    df.loc[~nue_cc0piNp, "Signal_1eNp"] = False
    
    df.loc[nue_cc0pi1p, "Signal_1e1p"] = True
    df.loc[~nue_cc0pi1p, "Signal_1e1p"] = False
    
    # Set the topological categories
    
    nue_cc0pi0p = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] != -1) & (df["TrueLeadProtonIdx"] == -1) & (df["InFV"] == True) & (df["HasNoMesons"] == True) & (df["TrueNProt"] == 0))
    nue_cc0pi2p = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] != -1) & (df["TrueLeadProtonIdx"] != -1) & (df["InFV"] == True) & (df["HasNoMesons"] == True) & (df["TrueNProt"] >= 2))
    nue_cc = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] != -1) & (df["InFV"] == True) & (df["TrueFSPions"] > 0) & (df["TrueFSPi0"] > 0) & (df["TrueNProt"] >= 0))
    nue_nc0pi0 = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] == -1) & (df["InFV"] == True) & (df["TrueFSPions"] == 0) & (df["TrueFSPi0"] == 0) & (df["TrueNProt"] == 0))
    nue_ncNpi0 = ((abs(df["nu_pdg"]) == 12) & (df["TrueElecIdx"] == -1) & (df["InFV"] == True) & (df["TrueFSPions"] == 0) & (df["TrueFSPi0"] >= 1) & (df["TrueNProt"] == 0))
    
    numu_ccNpi0 = ((abs(df["nu_pdg"]) == 14) & (df["TrueMuonIdx"] != -1) & (df["InFV"] == True) & (df["TrueFSPions"] >= 0) & (df["TrueFSPi0"] >= 1) & (df["TrueNProt"] >= 0))
    numu_cc0pi0 = ((abs(df["nu_pdg"]) == 14) & (df["TrueMuonIdx"] != -1) & (df["InFV"] == True) & (df["TrueFSPions"] >= 0) & (df["TrueFSPi0"] == 1) & (df["TrueNProt"] >= 0))
    numu_nc0pi0 = ((abs(df["nu_pdg"]) == 14) & (df["TrueMuonIdx"] == -1) & (df["InFV"] == True) & (df["TrueFSPions"] == 0) & (df["TrueFSPi0"] == 0) & (df["TrueNProt"] == 0))
    numu_ncNpi0 = ((abs(df["nu_pdg"]) == 14) & (df["TrueMuonIdx"] == -1) & (df["InFV"] == True) & (df["TrueFSPions"] == 0) & (df["TrueFSPi0"] >= 1) & (df["TrueNProt"] == 0))
    
    outFV = df["InFV"] == False
    #cosmic = ((abs(df["nu_pdg"]) != 14) & (df["nu_pdg"] != 12) & (df["InFV"] == True))
    cosmic = ((df["InFV"] == True) & ~nue_cc0piNp & ~nue_cc0pi1p & ~nue_cc0pi0p & ~nue_cc0pi2p & ~nue_cc & ~nue_nc0pi0 & ~nue_ncNpi0 & ~numu_ccNpi0 & ~numu_cc0pi0 & ~numu_nc0pi0 & ~numu_ncNpi0)
    
    df["category_1e1p_tki"] = 6  # 'other'
    df.loc[nue_cc0pi0p, "category_1e1p_tki"] = 10
    df.loc[nue_cc0pi1p, "category_1e1p_tki"] = 12  # 1e1p signal
    df.loc[nue_cc0pi2p, "category_1e1p_tki"] = 13
    df.loc[nue_cc, "category_1e1p_tki"] = 1
    df.loc[nue_nc0pi0, "category_1e1p_tki"] = 3
    df.loc[nue_ncNpi0, "category_1e1p_tki"] = 31
    df.loc[numu_cc0pi0, "category_1e1p_tki"] = 2
    df.loc[numu_ccNpi0, "category_1e1p_tki"] = 21
    df.loc[numu_nc0pi0, "category_1e1p_tki"] = 3
    df.loc[numu_ncNpi0, "category_1e1p_tki"] = 31
    df.loc[outFV, "category_1e1p_tki"] = 5
    df.loc[cosmic, "category_1e1p_tki"] = 4

    # Calculate the TKI variables for 1e1p using the leading proton

    print("Calc true TKI variables for leading proton only")

    df["TrueDeltaPT"] = df.apply(lambda x: (tki_calculators.delta_pT(x["TrueElecMomX"],x["TrueElecMomY"],x["TrueElecMomZ"],x["TrueLeadProtonMomX"],x["TrueLeadProtonMomY"],x["TrueLeadProtonMomZ"])),axis=1)
    #df["TrueDeltaPhiT"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["TrueElecMomX"],x["TrueElecMomY"],x["TrueElecMomZ"],x["TrueLeadProtonMomX"],x["TrueLeadProtonMomY"],x["TrueLeadProtonMomZ"])),axis=1)
    df["TrueDeltaAlphaT"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["TrueElecMomX"],x["TrueElecMomY"],x["TrueElecMomZ"],x["TrueLeadProtonMomX"],x["TrueLeadProtonMomY"],x["TrueLeadProtonMomZ"])),axis=1)
    df['TrueDeltaAlphaT'] = np.degrees(df['TrueDeltaAlphaT'])
    
    
    # Drop temporary data from dataframes
    df.drop("mc_pdg", inplace=True, axis=1)
    df.drop("mc_E", inplace=True, axis=1)
    df.drop("mc_px", inplace=True, axis=1)
    df.drop("mc_py", inplace=True, axis=1)
    df.drop("mc_pz", inplace=True, axis=1)
    
    return df