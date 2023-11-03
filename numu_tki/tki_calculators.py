import math
import numpy as np

# Functions for adding TKI variables to data frames and applying 1muNp selection
# developed by S Gardiner
# Author: C Thorpe
# Do everything in components for sake for speed
# GKI Variables defined in docdb 39499

TARGET_MASS = 37.215526
NEUTRON_MASS = 0.93956541
PROTON_MASS = 0.93827208
BINDING_ENERGY = 0.02478 

################################################################################
# Vector pT

def vec_pT(Mom_x,Mom_y,Mom_z):
    pT = np.array([0,0,0])
    if isinstance(Mom_x,list): 
        for i in range(0,len(Mom_x)):
            pT = pT + vec_pT(Mom_x[i],Mom_y[i],0)
    else:    
        pT = np.array([Mom_x,Mom_y,0])

    return pT 

################################################################################
# Mag of delta pT

def delta_pT_vec(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    had_pT = vec_pT(HadMom_x,HadMom_y,HadMom_z)
    lep_pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)

    return lep_pT + had_pT 

################################################################################
# Mag of delta pT

def delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    dpT = delta_pT_vec(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z)
    return np.sqrt(dpT.dot(dpT))

################################################################################
# Delta phi T

def delta_phiT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    had_pT = vec_pT(HadMom_x,HadMom_y,HadMom_z)
    lep_pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)
    mag_lep_pT = np.sqrt(lep_pT.dot(lep_pT))
    mag_had_pT = np.sqrt(had_pT.dot(had_pT))
    return math.acos(-lep_pT.dot(had_pT)/mag_lep_pT/mag_had_pT)

################################################################################
# Delta phi T

def delta_alphaT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    dpT = delta_pT_vec(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z)
    lep_pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)
    mag_lep_pT = np.sqrt(lep_pT.dot(lep_pT))
    mag_dpT = np.sqrt(dpT.dot(dpT))
    return math.acos(-lep_pT.dot(dpT)/mag_lep_pT/mag_dpT)

