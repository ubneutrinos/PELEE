import numpy as np

# Functions for determining a particle's velocity, and if that velocity passes the critical velocity to produce Cherenkov light in a MiniBooNE-like detector using truth Monte Carlo variables
# Author: J.B. Tyler & M. Moudgalya

###################################################################
# True primary vertex is not in dead region

DEAD_Z_MIN = 675 # SG's code does not cut dead region
DEAD_Z_MAX = 775

def not_dead_region(true_nu_vtx_x,true_nu_vtx_y,true_nu_vtx_z):

    return not (DEAD_Z_MIN < true_nu_vtx_z < DEAD_Z_MAX)

#################################################################
# Calculate true velocity of mc particle

def get_true_vel_vector(mc_E, mc_px, mc_py, mc_pz):
    
    vel = []
    
    for i in range(len(mc_E)):
        p = np.sqrt(mc_px[i]**2 + mc_py[i]**2 + mc_pz[i]**2)
        v = p/mc_E[i] #units of 1/c
        
        vel.append(v)
        
    return vel

#################################################################
# Determine if velocity passes critical velocity test

c = 1 #check later -> this is fine
n = 1.4620 # Index of refraction for the Marcol 7 mineral oil used by MiniBooNE

crit_vel = c/n

def true_crit_vel_test_vector(true_vel_vector):
    
    pass_true_crit_vel = [] # Velocity is greater than critical velocity
    
    for i in range(len(true_vel_vector)):
        if true_vel_vector[i] > crit_vel:
            test = True
        else:
            test = False
        pass_true_crit_vel.append(test)
        
    return pass_true_crit_vel

##################################################################
# Determine if slice contains any particle that passed critical velocity test

def true_crit_vel_slice_bool(pass_true_crit_vel, not_dead_region):
    
    if True in pass_true_crit_vel and not_dead_region == True:
        return True
    else:
        return False

###################################################################
# Calculate true KE of mc particle

def get_true_KE_vector(mc_E, mc_px, mc_py, mc_pz):
    
    vec = []
    
    for i in range(len(mc_E)):
        p = np.sqrt(mc_px[i]**2 + mc_py[i]**2 + mc_pz[i]**2)
        ke = mc_E[i] - np.sqrt(mc_E[i]**2 - p**2)
        
        vec.append(ke)
        
    return vec

###################################################################
# Calculating lost energy (found energy: remove "not" -> AboveChrnkv_E)

def get_SubChrnkv_E(true_crit_vel_test_vector, true_KE_vector, mc_pdg):
    
    iv = 0.0
    
    for i in range(len(true_crit_vel_test_vector)):
        if mc_pdg[i] == 2112: #remove neutrons
            continue
        if not true_crit_vel_test_vector[i]:
            iv = iv + true_KE_vector[i]
            
    return iv            
    
###################################################################
# Calculate Quasi-Elastic Neutrino Energy Electron Scalar & Vector Versions

def get_EvQEle(mc_pdg, mc_E, mc_pz):
    
    Me = 0.511e-3
    Mp = 0.9383
    Mn = 0.9396
    Eb = 0.0285
        
    EvQEle = 0.0 #Scalar
#    EvQEle = [] #Vector
    
    for i in range(len(mc_pdg)):
        if mc_pdg[i] == 11:
            x1 = ((mc_E[i] * (Mn - Eb)) + (0.5 * (Mp**2 - (Mn - Eb)**2 - Me**2)))/((Mn - Eb) + mc_pz[i] - mc_E[i])
            if x1 > EvQEle: #Scalar
                EvQEle = x1 #Scalar
            #EvQEle.append(x1) #Vector
            
        else:
            continue
                
    return EvQEle

###################################################################
# Calculate Quasi-Elastic Neutrino Energy Muon

def get_EvQElu(mc_pdg, mc_E, mc_pz):
    
    Mu = 0.10566
    Mp = 0.9383
    Mn = 0.9396
    Eb = 0.0285
    
    EvQElu = 0.0
    
    for i in range(len(mc_pdg)):
        if mc_pdg[i] == 13:
            x2 = ((mc_E[i] * (Mn - Eb)) + (0.5 * (Mp**2 - (Mn - Eb)**2 - Mu**2)))/((Mn - Eb) + mc_pz[i] - mc_E[i])
            if x2 > EvQElu:
                EvQElu = x2
        else:
            continue
                
    return EvQElu

###################################################################
# Counting Protons Below Cherenkov Threshold



###################################################################
# Counting Protons Above Cherenkov Threshold



###################################################################
# Add a column containing critical velocity pass/fail info
# NOTE: Order of variables here MUST MATCH order of variables in function above!!!

def process_oLEE_truth(up,df):
    
    print("Yes oLEE")
    
    # Load the extra branches needed
    df["mc_pdg"] = up.array("mc_pdg")
    df["mc_E"] = up.array("mc_E")
    df["mc_px"] = up.array("mc_px")
    df["mc_py"] = up.array("mc_py")
    df["mc_pz"] = up.array("mc_pz")
    
    df["not_dead_region"] = df.apply(lambda x: (not_dead_region(x["true_nu_vtx_x"],x["true_nu_vtx_y"],x["true_nu_vtx_z"])),axis=1)

    df["true_vel_vector"] = df.apply(lambda x: (get_true_vel_vector(x["mc_E"],x["mc_px"],x["mc_py"],x["mc_pz"])),axis=1)
    
    df["true_crit_vel_test_vector"] = df.apply(lambda x: (true_crit_vel_test_vector(x["true_vel_vector"])),axis=1)

    df["true_crit_vel_slice_bool"] = df.apply(lambda x: (true_crit_vel_slice_bool(x["true_crit_vel_test_vector"],x["not_dead_region"])),axis=1)
    
    df["true_KE_vector"] = df.apply(lambda x: (get_true_KE_vector(x["mc_E"],x["mc_px"],x["mc_py"],x["mc_pz"])),axis=1)
    
    df["SubChrnkv_E"] = df.apply(lambda x: (get_SubChrnkv_E(x["true_crit_vel_test_vector"],x["true_KE_vector"],x["mc_pdg"])),axis=1)
    
    df["EvQEle"] = df.apply(lambda x: (get_EvQEle(x["mc_pdg"],x["mc_E"],x["mc_pz"])),axis=1)
    df["EvQElu"] = df.apply(lambda x: (get_EvQElu(x["mc_pdg"],x["mc_E"],x["mc_pz"])),axis=1)

    # Drop temporary data from dataframes
    df.drop("mc_px", inplace=True, axis=1)
    df.drop("mc_py", inplace=True, axis=1)
    df.drop("mc_pz", inplace=True, axis=1)
    
    return df

