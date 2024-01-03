import math
import numpy as np
from numu_tki import tki_calculators

# Functions for setting the signal definition and adding useful variables for
# the CC1muNP selection developed by S Gardiner
# Author: C Thorpe

################################################################################
# Check there is a final state muon above threshold, and return its index

muon_p_min=0.1
muon_p_max=1.2
muon_mass=0.1057
muon_E_min=math.sqrt(muon_p_min*muon_p_min+muon_mass*muon_mass)
muon_E_max=math.sqrt(muon_p_max*muon_p_max+muon_mass*muon_mass) # SG's code imposes an upper limit

def true_muon_idx(mc_pdg,mc_E):

    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 13 and mc_E[i] > muon_E_min and mc_E[i] < muon_E_max:
            return i

    return -1

################################################################################
# Final state has one proton above threshold

proton_p_min=0.250
proton_p_max=1.0
proton_mass=0.939
proton_E_min=math.sqrt(proton_p_min*proton_p_min+proton_mass*proton_mass)
proton_E_max=math.sqrt(proton_p_max*proton_p_max+proton_mass*proton_mass)

def true_proton_idx(mc_pdg,mc_E):

    idx=[]
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_min and mc_E[i] < proton_E_max:
            idx.append(i)

    return idx

################################################################################
# Index of lead proton 

def true_lead_proton_idx(mc_pdg,mc_E):

    lead_idx=-1
    lead_E=-1
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_min and mc_E[i] < proton_E_max and mc_E[i] > lead_E:
            lead_E = mc_E[i]
            lead_idx = i

    return lead_idx

################################################################################
# Final state has no charged pions above thresold and no pi0

pion_thresh=0.1
pion_mass=0.1396
pion_E_thresh=math.sqrt(pion_thresh*pion_thresh+pion_mass*pion_mass)

def has_no_mesons(mc_pdg,mc_E):
    for i in range(0,len(mc_pdg)):
        if (abs(mc_pdg[i]) == 211 and mc_E[i] > pion_E_thresh) or mc_pdg[i] == 111 or abs(mc_pdg[i]) == 321:
            return False 

    return True 

################################################################################
# True primary vertex is in the fiducial volume

'''
# Definitions from SG's code
FV_X_MIN = 21.5
FV_X_MAX = 234.85
FV_Y_MIN = -95.0
FV_Y_MAX = 95.0
FV_Z_MIN = 21.5
FV_Z_MAX = 966.8
DEAD_Z_MIN = 10000 # SG's code does not cut dead region 
DEAD_Z_MAX = 10000
'''
# Definitions in other PeLEE code
FV_X_MIN =   5.0
FV_X_MAX =  251.0
FV_Y_MIN = -111.0
FV_Y_MAX =  111.0
FV_Z_MIN =   20.0
FV_Z_MAX =  986.0
DEAD_Z_MIN = 675 # SG's code does not cut dead region 
DEAD_Z_MAX = 775

def in_fiducial_volume(true_nu_vtx_x,true_nu_vtx_y,true_nu_vtx_z):

    return true_nu_vtx_x > FV_X_MIN and true_nu_vtx_x < FV_X_MAX and\
           true_nu_vtx_y > FV_Y_MIN and true_nu_vtx_y < FV_Y_MAX and\
           true_nu_vtx_z > FV_Z_MIN and true_nu_vtx_z < FV_Z_MAX and\
           not (true_nu_vtx_z > DEAD_Z_MIN and true_nu_vtx_z < DEAD_Z_MAX) 

################################################################################
# Component of the momentum of a particle at index TrueIdx_1muNp 

def true_mom(TrueIdx_1muNp,mc_p):

    if TrueIdx_1muNp == -1: return np.nan
    else: return mc_p[TrueIdx_1muNp]

################################################################################
# Component of the total momentum of a group of particles 

def true_mom_tot(TrueIdx_v,mc_p):

    if len(TrueIdx_v) == 0:
        return np.nan

    p=0
    for i in TrueIdx_v:
        p = p + mc_p[i]    

    return p 

################################################################################
# Vector of momenta of group of particles 

def true_mom_v(TrueIdx_v,mc_p):

    p=[]
    for i in TrueIdx_v:
        p.append(mc_p[i])

    return p 

################################################################################
# Number of protons 

def n_proton(TrueIdx_v):

    return len(TrueIdx_v) 

################################################################################
# Number of pions 

pion_min_p = 0.1
pion_mass = 0.1396
pion_min_E = math.sqrt(pion_min_p*pion_min_p + pion_mass*pion_mass) 

def n_fs_pion(mc_pdg,mc_E):

    n_pion=0
    for i in range(0,len(mc_pdg)):
        if mc_pdg[i] == 111: n_pion = n_pion+1
        if abs(mc_pdg[i]) == 211 and mc_E[i] > pion_min_E: n_pion = n_pion+1

    return n_pion

################################################################################
# Add a column indicating if the events belong to
# the 1muNp signal

def set_Signal1muNp(up,df):

    # Load the extra branches needed 
    df["mc_pdg"] = up.array("mc_pdg")
    df["mc_E"] = up.array("mc_E")
    df["mc_px"] = up.array("mc_px")
    df["mc_py"] = up.array("mc_py")
    df["mc_pz"] = up.array("mc_pz")

    df["InFV_1muNp"] = df.apply(lambda x: (in_fiducial_volume(x["true_nu_vtx_x"],x["true_nu_vtx_y"],x["true_nu_vtx_z"])),axis=1)
    
    df["TrueMuonIdx_1muNp"] = df.apply(lambda x: (true_muon_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueProtonIdx_1muNp"] = df.apply(lambda x: (true_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueLeadProtonIdx_1muNp"] = df.apply(lambda x: (true_lead_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueNProt_1muNp"] = df.apply(lambda x: (n_proton(x["TrueProtonIdx_1muNp"])),axis=1)
    df["TrueFSPions_1muNp"] = df.apply(lambda x: (n_fs_pion(x["mc_pdg"],x["mc_E"])),axis=1)

    df["TrueMuonE_1muNp"] = df.apply(lambda x: (true_mom(x["TrueMuonIdx_1muNp"],x["mc_E"])),axis=1)
    df["TrueMuonMomX_1muNp"] = df.apply(lambda x: (true_mom(x["TrueMuonIdx_1muNp"],x["mc_px"])),axis=1)
    df["TrueMuonMomY_1muNp"] = df.apply(lambda x: (true_mom(x["TrueMuonIdx_1muNp"],x["mc_py"])),axis=1)
    df["TrueMuonMomZ_1muNp"] = df.apply(lambda x: (true_mom(x["TrueMuonIdx_1muNp"],x["mc_pz"])),axis=1)

    df["TrueTotalProtonE_1muNp"] = df.apply(lambda x: (true_mom_tot(x["TrueProtonIdx_1muNp"],x["mc_E"])),axis=1)
    df["TrueTotalProtonMomX_1muNp"] = df.apply(lambda x: (true_mom_tot(x["TrueProtonIdx_1muNp"],x["mc_px"])),axis=1)
    df["TrueTotalProtonMomY_1muNp"] = df.apply(lambda x: (true_mom_tot(x["TrueProtonIdx_1muNp"],x["mc_py"])),axis=1)
    df["TrueTotalProtonMomZ_1muNp"] = df.apply(lambda x: (true_mom_tot(x["TrueProtonIdx_1muNp"],x["mc_pz"])),axis=1)

    df["TrueProtonE_1muNp"] = df.apply(lambda x: (true_mom_v(x["TrueProtonIdx_1muNp"],x["mc_E"])),axis=1)
    df["TrueProtonMomX_1muNp"] = df.apply(lambda x: (true_mom_v(x["TrueProtonIdx_1muNp"],x["mc_px"])),axis=1)
    df["TrueProtonMomY_1muNp"] = df.apply(lambda x: (true_mom_v(x["TrueProtonIdx_1muNp"],x["mc_py"])),axis=1)
    df["TrueProtonMomZ_1muNp"] = df.apply(lambda x: (true_mom_v(x["TrueProtonIdx_1muNp"],x["mc_pz"])),axis=1)

    df["TrueLeadProtonE_1muNp"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx_1muNp"],x["mc_E"])),axis=1)
    df["TrueLeadProtonMomX_1muNp"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx_1muNp"],x["mc_px"])),axis=1)
    df["TrueLeadProtonMomY_1muNp"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx_1muNp"],x["mc_py"])),axis=1)
    df["TrueLeadProtonMomZ_1muNp"] = df.apply(lambda x: (true_mom(x["TrueLeadProtonIdx_1muNp"],x["mc_pz"])),axis=1)

    # Set the signal definition
    df.loc[((df["TrueMuonIdx_1muNp"] != -1) & (df["TrueLeadProtonIdx_1muNp"] != -1) & (df["InFV_1muNp"] == True) & (df["TrueFSPions_1muNp"] == 0)), "Signal_1muNp"] = True 
    df.loc[((df["TrueMuonIdx_1muNp"] == -1) | (df["TrueLeadProtonIdx_1muNp"] == -1) | (df["InFV_1muNp"] != True) | (df["TrueFSPions_1muNp"] != 0)), "Signal_1muNp"] = False 
    df.loc[((df["TrueMuonIdx_1muNp"] != -1) & (df["TrueLeadProtonIdx_1muNp"] != -1) & (df["InFV_1muNp"] == True) & (df["TrueFSPions_1muNp"] == 0) & (df["TrueNProt_1muNp"] == 1)), "Signal_1mu1p"] = True 
    df.loc[((df["TrueMuonIdx_1muNp"] == -1) | (df["TrueLeadProtonIdx_1muNp"] == -1) | (df["InFV_1muNp"] != True) | (df["TrueFSPions_1muNp"] != 0) | (df["TrueNProt_1muNp"] != 1)), "Signal_1mu1p"] = False 

    print("Calc true TKI variables for leading proton only")

    df["TrueDeltaPT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)

    df["TrueDeltaPhiT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaAlphaT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueECal_1mu1p"] = df.apply(lambda x: (tki_calculators.Ecal(x["TrueMuonE_1muNp"],x["TrueLeadProtonE_1muNp"])),axis=1)
    df["TruePL_1mu1p"] = df.apply(lambda x: (tki_calculators.pL(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPL_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pL(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePN_1mu1p"] = df.apply(lambda x: (tki_calculators.pn(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueAlpha3D_1mu1p"] = df.apply(lambda x: (tki_calculators.alpha_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePhi3D_1mu1p"] = df.apply(lambda x: (tki_calculators.phi_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTX_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT_X(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTY_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT_Y(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNTX_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_TX(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNTY_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_TY(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNT_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_T(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNII_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_II(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    
    print("Calc true TKI variables with all FS protons above threshold")

    df["TrueDeltaPT_1muNp"] = df.apply(lambda x: (tki_calculators.delta_pT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPhiT_1muNp"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaAlphaT_1muNp"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueECal_1muNp"] = df.apply(lambda x: (tki_calculators.Ecal(x["TrueMuonE_1muNp"],x["TrueProtonE_1muNp"])),axis=1)
    df["TruePL_1muNp"] = df.apply(lambda x: (tki_calculators.pL(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePN_1muNp"] = df.apply(lambda x: (tki_calculators.pn(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueAlpha3D_1muNp"] = df.apply(lambda x: (tki_calculators.alpha_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePhi3D_1muNp"] = df.apply(lambda x: (tki_calculators.phi_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTX_1muNp"] = df.apply(lambda x: (tki_calculators.delta_pT_X(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTY_1muNp"] = df.apply(lambda x: (tki_calculators.delta_pT_Y(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePNTX_1muNp"] = df.apply(lambda x: (tki_calculators.pn_TX(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePNTY_1muNp"] = df.apply(lambda x: (tki_calculators.pn_TY(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePNT_1muNp"] = df.apply(lambda x: (tki_calculators.pn_T(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)
    df["TruePNII_1muNp"] = df.apply(lambda x: (tki_calculators.pn_II(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueProtonE_1muNp"],x["TrueProtonMomX_1muNp"],x["TrueProtonMomY_1muNp"],x["TrueProtonMomZ_1muNp"])),axis=1)

    # Drop temporary data from dataframes
    df.drop("mc_pdg",inplace=True,axis=1)
    df.drop("mc_E",inplace=True,axis=1)
    df.drop("mc_px",inplace=True,axis=1)
    df.drop("mc_py",inplace=True,axis=1)
    df.drop("mc_pz",inplace=True,axis=1)
    df.drop("InFV_1muNp",inplace=True,axis=1)
    df.drop("TrueMuonIdx_1muNp",inplace=True,axis=1)
    df.drop("TrueProtonIdx_1muNp",inplace=True,axis=1)
    df.drop("TrueLeadProtonIdx_1muNp",inplace=True,axis=1)
    df.drop("TrueNProt_1muNp",inplace=True,axis=1)
    df.drop("TrueTotalProtonE_1muNp",inplace=True,axis=1)
    df.drop("TrueTotalProtonMomX_1muNp",inplace=True,axis=1)
    df.drop("TrueTotalProtonMomY_1muNp",inplace=True,axis=1)
    df.drop("TrueTotalProtonMomZ_1muNp",inplace=True,axis=1)
    df.drop("TrueLeadProtonE_1muNp",inplace=True,axis=1)
    df.drop("TrueLeadProtonMomX_1muNp",inplace=True,axis=1)
    df.drop("TrueLeadProtonMomY_1muNp",inplace=True,axis=1)
    df.drop("TrueLeadProtonMomZ_1muNp",inplace=True,axis=1)
    df.drop("TrueProtonE_1muNp",inplace=True,axis=1)
    df.drop("TrueProtonMomX_1muNp",inplace=True,axis=1)
    df.drop("TrueProtonMomY_1muNp",inplace=True,axis=1)
    df.drop("TrueProtonMomZ_1muNp",inplace=True,axis=1)
    df.drop("TrueMuonE_1muNp",inplace=True,axis=1)
    df.drop("TrueMuonMomX_1muNp",inplace=True,axis=1)
    df.drop("TrueMuonMomY_1muNp",inplace=True,axis=1)
    df.drop("TrueMuonMomZ_1muNp",inplace=True,axis=1)

    return df

