import math

# Functions for adding TKI variables to data frames and applying 1muNp selection
# developed by S Gardiner
# Author: C Thorpe

################################################################################
# Final state has a muon above threshold
muon_thresh=0.1
muon_mass=0.1057
muon_E_thresh=math.sqrt(muon_thresh*muon_thresh+muon_mass*muon_mass)

def has_muon(mc_pdg,mc_E):

    for i in range(0,len(mc_pdg)):
        #print(mc_pdg[i],mc_E[i]) 
        if abs(mc_pdg[i]) == 13 and mc_E[i] > muon_E_thresh:
            return True 
    return False

################################################################################
# Final state has one proton above threshold

proton_thresh=0.3
proton_mass=0.939
proton_E_thresh=math.sqrt(proton_thresh*proton_thresh+proton_mass*proton_mass)

def n_fs_protons(mc_pdg,mc_E):
    nprot=0
    for i in range(0,len(mc_pdg)):
        if abs(mc_pdg[i]) == 2212 and mc_E[i] > proton_E_thresh:
            nprot = nprot+1

    return nprot

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

def in_fiducial_volume(true_nu_vtx_x,true_nu_vtx_y,true_nu_vtx_z):

    return true_nu_vtx_x > 5.0 and true_nu_vtx_x < 251.0 and\
           abs(true_nu_vtx_y) < 111.0 and\
           true_nu_vtx_z > 20.0 and true_nu_vtx_z < 986.0 and\
           not (true_nu_vtx_z > 675 and true_nu_vtx_z < 775) 

################################################################################
# Add a column indicating if the events belong to
# the 1muNp signal

def set_1muNpSignal(df):
 
    # First check if there is a muon in the dataframe
     
    #print(df["mc_pdg"].iloc[0][0]) 
 
    #df["1mu1pSignal"] = False
    df["1mu1pSignal"] = df.apply(lambda x: (has_muon(x["mc_pdg"],x["mc_E"]) and\
                                            n_fs_protons(x["mc_pdg"],x["mc_E"]) == 1 and\
                                            has_no_mesons(x["mc_pdg"],x["mc_E"]) and\
                                            in_fiducial_volume(x["true_nu_vtx_x"],x["true_nu_vtx_y"],x["true_nu_vtx_z"]))\
                                            ,axis=1)

    print(df["1mu1pSignal"])
 
    return df    
