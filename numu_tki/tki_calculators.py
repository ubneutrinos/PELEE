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

def vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    had_pT = vec_pT(HadMom_x,HadMom_y,HadMom_z)
    lep_pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)

    return lep_pT + had_pT 

################################################################################
# Mag of delta pT

def delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z):

    dpT = vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z)
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

    dpT = vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z)
    lep_pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)
    mag_lep_pT = np.sqrt(lep_pT.dot(lep_pT))
    mag_dpT = np.sqrt(dpT.dot(dpT))

    # Sometimes get domain errors
    arg=-lep_pT.dot(dpT)/mag_lep_pT/mag_dpT
    if arg != np.nan:
        if arg > 1.0: return 0.0
        if arg < -1.0: return 3.1415

    return math.acos(arg)

################################################################################
# ECal 

def Ecal(Lep_E,Had_E):

    E=0
    if isinstance(Had_E,list): 
        for i in range(0,len(Had_E)):
            E = E + Had_E[i] - PROTON_MASS + BINDING_ENERGY
    else:    
        E = Had_E - PROTON_MASS + BINDING_ENERGY 
    
    return E + Lep_E

################################################################################
# pL 

def vec_pL(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):

    p = LepMom_z
    if isinstance(Had_E,list): 
        for i in range(0,len(Had_E)):
            p = p + HadMom_z[i]
    else: 
        p = p + HadMom_z

    p = p - Ecal(Lep_E,Had_E)

    return np.array([0,0,p])

################################################################################
# pL 

def pL(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    p = vec_pL(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return np.sqrt(p.dot(p))

################################################################################
# Vector q (4 momentum transfer) 

def vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    return np.array([-LepMom_x,-LepMom_y,Ecal(Lep_E,Had_E)-LepMom_z])
   
################################################################################
# Vector pn 

def vec_pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    return vec_pL(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)+\
           vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z)

################################################################################
# pn 

def pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    p = vec_pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return np.sqrt(p.dot(p))

################################################################################
# Phi 3D 

def phi_3D(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    q = vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    pp = np.array([0,0,0])
    if isinstance(HadMom_x,list): 
        for i in range(0,len(HadMom_x)):
            pp = pp + np.array([HadMom_x[i],HadMom_y[i],HadMom_z[i]])
    else: 
        pp = np.array([HadMom_x,HadMom_y,HadMom_z])
    
    return math.acos(q.dot(pp)/np.sqrt(q.dot(q))/np.sqrt(pp.dot(pp)))
    #return np.sqrt(pp.dot(pp))

################################################################################
# Alpha 3D 

def alpha_3D(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    q = vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    pn = vec_pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    print(q.dot(pn)/np.sqrt(q.dot(q))/np.sqrt(pn.dot(pn)))
    arg = q.dot(pn)/np.sqrt(q.dot(q))/np.sqrt(pn.dot(pn))
    if arg != np.nan:
        if arg > 1.0: return 0.0
        if arg < -1.0: return 3.1415
    return math.acos(arg)

################################################################################
# delta pT x 

def delta_pT_X(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    pnu = np.array([0,0,1.0])
    pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)
    
    return (np.cross(pnu,pT)).dot(vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z))
################################################################################
# delta pT y 

def delta_pT_Y(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    pT = vec_pT(LepMom_x,LepMom_y,LepMom_z)
    return -pT.dot(vec_delta_pT(LepMom_x,LepMom_y,LepMom_z,HadMom_x,HadMom_y,HadMom_z))

################################################################################
# p n (perp) x 

def pn_TX(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    q = vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    qT = np.array([q[0],q[1],0.0])
    pnu = np.array([0,0,1.0])   
    p = pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return np.cross(qT,pnu).dot(p)

################################################################################
# p n (perp) y 

def pn_TY(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    q = vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    qT = np.array([q[0],q[1],0.0])
    pnu = np.array([0,0,1.0])   
    p = pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return np.cross(q,np.cross(qT,pnu)).dot(p)

################################################################################
# p n (perp) 

def pn_T(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    pnx = pn_TX(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    pny = pn_TY(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return np.sqrt(pnx*pnx + pny*pny)

################################################################################
# p n (paral) 

def pn_II(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z):
    q = vec_q(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    p = pn(Lep_E,LepMom_x,LepMom_y,LepMom_z,Had_E,HadMom_x,HadMom_y,HadMom_z)
    return q.dot(p) 
