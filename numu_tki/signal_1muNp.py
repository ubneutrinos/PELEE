import math
import numpy as np
from numu_tki import tki_calculators

# Functions for setting the signal definition and adding useful variables for
# the CC1muNP selection developed by S Gardiner
# Author: C Thorpe

################################################################################
# Check there is a final state muon above threshold, and return its index

muon_thresh=0.1
muon_mass=0.1057
muon_E_thresh=math.sqrt(muon_thresh*muon_thresh+muon_mass*muon_mass)

def true_muon_idx(mc_pdg,mc_E):

    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 13 and mc_E[i] > muon_E_thresh:
            return i

    return -1

################################################################################
# Final state has one proton above threshold

proton_thresh=0.3
proton_mass=0.939
proton_E_thresh=math.sqrt(proton_thresh*proton_thresh+proton_mass*proton_mass)

def true_proton_idx(mc_pdg,mc_E):

    idx=[]
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_thresh:
            idx.append(i)

    return idx

################################################################################
# Index of lead proton 

def true_lead_proton_idx(mc_pdg,mc_E):

    lead_idx=-1
    lead_E=-1
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_thresh and mc_E[i] > lead_E:
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

def in_fiducial_volume(true_nu_vtx_x,true_nu_vtx_y,true_nu_vtx_z):

    return true_nu_vtx_x > 5.0 and true_nu_vtx_x < 251.0 and\
           abs(true_nu_vtx_y) < 111.0 and\
           true_nu_vtx_z > 20.0 and true_nu_vtx_z < 986.0 and\
           not (true_nu_vtx_z > 675 and true_nu_vtx_z < 775) 

################################################################################
# Component of the momentum of a particle at index TrueIdx_1muNp 

def true_mom(TrueIdx_1muNp,mc_p):

    if TrueIdx_1muNp == -1: return np.nan
    else: return mc_p[TrueIdx_1muNp]

################################################################################
# Component of the total momentum of a group of particles 

def true_mom_tot(TrueIdx_v,mc_p):

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
# Add a column indicating if the events belong to
# the 1muNp signal

def set_Signal1muNp(df):

    df = df.head(100)

    df["InFV_1muNp"] = df.apply(lambda x: (in_fiducial_volume(x["true_nu_vtx_x"],x["true_nu_vtx_y"],x["true_nu_vtx_z"])),axis=1)
    df["TrueMuonIdx_1muNp"] = df.apply(lambda x: (true_muon_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueProtonIdx_1muNp"] = df.apply(lambda x: (true_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueLeadProtonIdx_1muNp"] = df.apply(lambda x: (true_lead_proton_idx(x["mc_pdg"],x["mc_E"])),axis=1)
    df["TrueNProt_1muNp"] = df.apply(lambda x: (n_proton(x["TrueProtonIdx_1muNp"])),axis=1)

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
    df.loc[((df["TrueMuonIdx_1muNp"] != -1) & (df["TrueLeadProtonIdx_1muNp"] != -1) & (df["InFV_1muNp"] == True)), "Signal_1muNp"] = True 
    df.loc[((df["TrueMuonIdx_1muNp"] != -1) & (df["TrueLeadProtonIdx_1muNp"] != -1) & (df["InFV_1muNp"] == True) & (df["TrueNProt_1muNp"] == 1)), "Signal_1mu1p"] = True 

    print("Calc deltapT for leading proton only")
    df["TrueDeltaPT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPhiT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaAlphaT_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueECal_1mu1p"] = df.apply(lambda x: (tki_calculators.Ecal(x["TrueMuonE_1muNp"],x["TrueLeadProtonE_1muNp"])),axis=1)
    df["TruePL_1mu1p"] = df.apply(lambda x: (tki_calculators.pL(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePN_1mu1p"] = df.apply(lambda x: (tki_calculators.pn(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueAlpha3D_1mu1p"] = df.apply(lambda x: (tki_calculators.alpha_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePhi3D_1mu1p"] = df.apply(lambda x: (tki_calculators.phi_3D(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTX_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT_X(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TrueDeltaPTY_1mu1p"] = df.apply(lambda x: (tki_calculators.delta_pT_Y(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNTX_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_TX(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNTY_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_TY(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNT_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_T(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)
    df["TruePNII_1mu1p"] = df.apply(lambda x: (tki_calculators.pn_II(x["TrueMuonE_1muNp"],x["TrueMuonMomX_1muNp"],x["TrueMuonMomY_1muNp"],x["TrueMuonMomZ_1muNp"],x["TrueLeadProtonE_1muNp"],x["TrueLeadProtonMomX_1muNp"],x["TrueLeadProtonMomY_1muNp"],x["TrueLeadProtonMomZ_1muNp"])),axis=1)

    print("Calc deltapT with all FS protons above threshold")
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

    return df   

