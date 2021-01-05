import numpy as np

# pi0 selection
LOOSE = False
if (LOOSE):
    SCORECUT = 0.8 # 0.75 #75 # max track score
    DVTX = 3.0 # 3. # distance from vertex of each shower
    VTXDOT = 0.8 # dot product between each shower's direction and the vtx -> shr start vector
    EMIN1 =  50. # leading photon min energy
    EMIN2 =  20. #20. # 20. # subleading photon min energy
    GAMMADOT = 0.94 # max dot product between showres
    DEDXCUT = 0.0 # MeV/cm cut on leading shower only
else:
    SCORECUT = 0.5 # 0.75 #75 # max track score
    DVTX = 3.0 # 3. # distance from vertex of each shower
    VTXDOT = 0.8 # dot product between each shower's direction and the vtx -> shr start vector
    EMIN1 =  60. # leading photon min energy
    EMIN2 =  40. #20. # 20. # subleading photon min energy
    GAMMADOT = 0.94 # max dot product between showres
    DEDXCUT = 1.0 # MeV/cm cut on leading shower only

CUT_VAR_V = ["nslice","pi0_shrscore1","pi0_shrscore2","pi0_dot1","pi0_dot2",\
            "pi0_radlen1","pi0_radlen2","pi0_gammadot","pi0_energy1_Y","pi0_energy2_Y",\
            "pi0_dedx1_fit_Y"]#,"n_showers_contained"]
CUT_VAL_V = [" == 1"," < %f"%SCORECUT," < %f"%SCORECUT," > %f"%VTXDOT," > %f"%VTXDOT,\
            " > %f"%DVTX," > %f"%DVTX," < %f"%GAMMADOT," > %f"%EMIN1," > %f"%EMIN2,\
            ">= %f"%DEDXCUT]#," != 0"]

def Pi0Query(APP):

    QUERY = ""
    
    for i,v in enumerate(CUT_VAR_V):
        if (i == 0):
            QUERY  += '%s_%s %s'%(v,APP,CUT_VAL_V[i])
        else:
            QUERY  += ' and %s_%s %s'%(v,APP,CUT_VAL_V[i])   
            
    return QUERY

def PRESQ(APP):

    # nue preselection
    q = 'nslice_%s == 1'%APP
    q += ' and selected_%s == 1'%APP
    q += ' and shr_energy_tot_cali_%s > 0.07'%APP

    return q

def NPPRESQ(APP):

    # 1eNp preselection
    q = PRESQ(APP)
    q += ' and n_tracks_contained_%s > 0'%APP

    return q

def NPVLCUTQ_all_showers(APP):

    # very loose box cuts
    q = NPPRESQ(APP)
    q += ' and CosmicIPAll3D_%s > 10.'%APP
    q += ' and trkpid_%s < 0.25'%APP
    q += ' and hits_ratio_%s > 0.5'%APP
    q += ' and trkfit_%s < 0.90'%APP
    q += ' and tksh_distance_%s < 10.0'%APP
    q += ' and tksh_angle_%s > -0.9'%APP

    return q
    
def NPVLCUTQ(APP):

    q = NPVLCUTQ_all_showers(APP) + ' and n_showers_contained_%s == 1'%APP

    return q


def NPLCUTQ_all_showers(APP):

    # loose box cuts
    q = NPVLCUTQ_all_showers(APP)
    q += ' and CosmicIPAll3D_%s > 10.'%APP
    q += ' and trkpid_%s < 0.02'%APP
    q += ' and hits_ratio_%s > 0.50'%APP
    q += ' and shrmoliereavg_%s < 9'%APP
    q += ' and subcluster_%s > 4'%APP
    q += ' and trkfit_%s < 0.65'%APP
    q += ' and tksh_distance_%s < 6.0'%APP
    q += ' and (shr_tkfit_nhits_tot_%s > 1 and shr_tkfit_dedx_max_%s > 0.5 and shr_tkfit_dedx_max_%s < 5.5)'%(APP,APP,APP)
    q += ' and tksh_angle_%s > -0.9'%APP
    q += ' and shr_trk_len_%s < 300.'%APP # new cut
    return q

def NPLCUTQ(APP):
    
    q = NPLCUTQ_all_showers(APP) + ' and n_showers_contained_%s == 1'%APP
    return q

def BDTCQ_all_showers(APP):
    
    q = NPLCUTQ_all_showers(APP)
    q += ' and pi0_score_%s > 0.67 and nonpi0_score_%s > 0.70'%(APP,APP)
    return q

def BDTCQ(APP):
    q = BDTCQ_all_showers(APP) + ' and n_showers_contained_%s == 1'%APP
    return q

def NPPRESQ_2pshowers(APP):
    q = NPPRESQ(APP) + ' and n_showers_contained_%s > 1'%APP
    return q

def NPLCUTQ_2pshowers(APP):
    q = NPLCUTQ_all_showers(APP) + ' and n_showers_contained_%s > 1'%APP
    return q

def BDTCQ_2pshowers(APP):
    q = BDTCQ_all_showers(APP) + ' and n_showers_contained_%s > 1'%APP
    return q

def ZPPRESQ(APP):
    # 1e0p preselection
    q = PRESQ(APP)
    q += ' and n_tracks_contained_%s == 0'%APP
    return q

def ZPLCUTQ_all_showers(APP):
    # loose box cuts
    q = ZPPRESQ(APP)
    q += ' and CosmicIPAll3D_%s > 10.'%APP
    q += ' and CosmicDirAll3D_%s > -0.9 and CosmicDirAll3D_%s < 0.9'%(APP,APP)
    q += ' and shrmoliereavg_%s < 15'%APP
    q += ' and subcluster_%s > 4'%APP
    q += ' and trkfit_%s < 0.65'%APP
    q += ' and secondshower_Y_nhit_%s < 50'%APP
    q += ' and shr_trk_sce_start_y_%s > -100 and shr_trk_sce_start_y_%s < 80'%(APP,APP)
    q += ' and shr_trk_sce_end_y_%s > -100 and shr_trk_sce_end_y_%s < 100 '%(APP,APP)
    q += ' and shr_trk_len_%s < 300.'%APP
    q += ' and (n_tracks_tot_%s == 0 or (n_tracks_tot_%s>0 and tk1sh1_angle_alltk_%s>-0.9))'%(APP,APP,APP)
    return q

def ZPLCUTQ(APP):
    q = ZPLCUTQ_all_showers(APP) + ' and n_showers_contained_%s == 1'%APP
    return q

def ZPLCUTQ_2pshowers(APP):
    q = ZPLCUTQ_all_showers(APP) + ' and n_showers_contained_%s > 1'%APP
    return q

def ZPBDTCQ(APP):
    q = ZPLCUTQ(APP)
    q += ' and bkg_score_%s >0.72'%APP
    return q

def NUMUPRESEL(APP):

    q = 'nslice_%s == 1'%APP
    q += ' and reco_nu_vtx_sce_x_%s > 5 and reco_nu_vtx_sce_x_%s < 251. '%(APP,APP)
    q += ' and reco_nu_vtx_sce_y_%s > -110 and reco_nu_vtx_sce_y_%s < 110. '%(APP,APP)
    q += ' and reco_nu_vtx_sce_z_%s > 20 and reco_nu_vtx_sce_z_%s < 986. '%(APP,APP)
    q += ' and (reco_nu_vtx_sce_z_%s < 675 or reco_nu_vtx_sce_z_%s > 775) '%(APP,APP)
    q += ' and topological_score_%s > 0.06 '%APP

    return q

def NUMUSEL(APP):


    q = NUMUPRESEL(APP)
    q += ' and (crtveto_%s != 1 or crthitpe_%s < 100) and _closestNuCosmicDist_%s > 5.'%(APP,APP,APP)
    q += ' and n_muons_tot_%s > 0'%APP

    return q


def NpQuery(APP):
    
    
    NPPRESQ_one_shower = NPPRESQ + ' and n_showers_contained_%s == 1'%APP
    NPPRESQ_one_shower_one_track = NPPRESQ_one_shower + ' and n_tracks_contained_%s == 1'%APP
    NPPRESQ_one_shower_twoplus_tracks = NPPRESQ_one_shower + ' and n_tracks_contained_%s > 1'%APP
    NPPRESQ_one_track = NPPRESQ + ' and n_tracks_contained_%s == 1'%APP
    NPPRESQ_twoplus_tracks = NPPRESQ + ' and n_tracks_contained_%s > 1'%APP
    
    # 2+ showers preselection
    PRESQ_twoplus_showers = PRESQ + ' and n_showers_contained_%s >= 2'%APP
    NPPRESQ_twoplus_showers = NPPRESQ + ' and n_showers_contained_%s >= 2'%APP
    
    
    
    # tight box cuts
    NPTCUTQ_all_showers = NPVLCUTQ_all_showers
    NPTCUTQ_all_showers += ' and CosmicIPAll3D_%s > 30.'%APP
    NPTCUTQ_all_showers += ' and CosmicDirAll3D_%s > -0.98 and CosmicDirAll3D_%s < 0.98'%(APP,APP)
    NPTCUTQ_all_showers += ' and trkpid_%s < 0.02'%APP
    NPTCUTQ_all_showers += ' and hits_ratio_%s > 0.65'%APP
    NPTCUTQ_all_showers += ' and shr_score_%s < 0.25'%APP
    NPTCUTQ_all_showers += ' and shrmoliereavg_%s > 2 and shrmoliereavg_%s < 10'%(APP,APP)
    NPTCUTQ_all_showers += ' and subcluster_%s > 7'%APP
    NPTCUTQ_all_showers += ' and trkfit_%s < 0.70'%APP
    NPTCUTQ_all_showers += ' and tksh_distance_%s < 4.0'%APP
    NPTCUTQ_all_showers += ' and trkshrhitdist2_%s < 1.5'%APP
    NPTCUTQ_all_showers += ' and (shr_tkfit_nhits_tot_%s > 1 and shr_tkfit_dedx_max_%s > 1.0 and shr_tkfit_dedx_max_%s < 3.8)'%(APP,APP,APP)
    NPTCUTQ_all_showers += ' and (secondshower_Y_nhit_%s <= 8 or secondshower_Y_dot_%s <= 0.8 or anglediff_Y_%s <= 40 or secondshower_Y_vtxdist_%s >= 100)'%(APP,APP,APP,APP)
    NPTCUTQ_all_showers += ' and tksh_angle_%s > -0.9 and tksh_angle_%s < 0.70'%(APP,APP)
    NPTCUTQ_all_showers += ' and shr_trk_len_%s < 300.'%APP
    NPTCUTQ = NPTCUTQ_all_showers + ' and n_showers_contained_%s == 1'%APP

    # BDT cuts
    # 0304 extnumi, pi0 and nonpi0
    BDTCQ_all_showers = NPLCUTQ_all_showers
    BDTCQ_all_showers += ' and pi0_score_%s > 0.67 and nonpi0_score_%s > 0.70'%(APP,APP)
    BDTCQ = BDTCQ_all_showers + ' and n_showers_contained_%s == 1'%APP

    return BDTCQ
    
    
def NpBoxCutVA(APP,IGNORE=-1):
    
    QUERY = ''
    
    # nue preselection
    PRESQ = 'nslice_%s == 1'%APP
    PRESQ += ' and selected_%s == 1'%APP
    PRESQ += ' and shr_energy_tot_cali_%s > 0.07'%APP
    PRESQ += ' and _opfilter_pe_beam_%s > 0 and _opfilter_pe_veto_%s < 20'%(APP,APP)
    
    #return PRESQ
    
    # 1eNp preselection
    NPPRESQ = PRESQ
    NPPRESQ += ' and n_tracks_contained_%s > 0'%APP
    
    # tight box cuts
    NPTCUTQ = NPPRESQ
    if (IGNORE != 0):
        NPTCUTQ += ' and CosmicIPAll3D_%s > 30.'%APP
        NPTCUTQ += ' and CosmicDirAll3D_%s > -0.98 and CosmicDirAll3D_%s < 0.98'%(APP,APP)
    if (IGNORE != 1):
        NPTCUTQ += ' and trkpid_%s < 0.02'%APP
    if (IGNORE != 2):
        NPTCUTQ += ' and hits_ratio_%s > 0.65'%APP
    if (IGNORE != 3):
        NPTCUTQ += ' and shr_score_%s < 0.25'%APP
    if (IGNORE != 4):
        NPTCUTQ += ' and shrmoliereavg_%s > 2 and shrmoliereavg_%s < 10'%(APP,APP)
    if (IGNORE != 5):
        NPTCUTQ += ' and subcluster_%s > 7'%APP
    if (IGNORE != 6):
        NPTCUTQ += ' and trkfit_%s < 0.70'%APP
    if (IGNORE != 7):
        NPTCUTQ += ' and n_showers_contained_%s == 1'%APP
    if (IGNORE != 8):
        NPTCUTQ += ' and tksh_distance_%s < 4.0'%APP
    if (IGNORE != 9):
        NPTCUTQ += ' and trkshrhitdist2_%s < 1.5'%APP
    if (IGNORE != 10):
        NPTCUTQ += ' and (shr_tkfit_nhits_tot_%s > 1 and shr_tkfit_dedx_max_%s > 1.0 and shr_tkfit_dedx_max_%s < 3.8)'%(APP,APP,APP)
    if (IGNORE != 11):
        NPTCUTQ += ' and (secondshower_Y_nhit_%s <= 8 or secondshower_Y_dot_%s <= 0.8 or anglediff_Y_%s <= 40 or secondshower_Y_vtxdist_%s >= 100)'%(APP,APP,APP,APP)
    if (IGNORE != 12):
        NPTCUTQ += ' and secondshower_Y_nhit_%s < 30'%APP
    if (IGNORE != 13):
        NPTCUTQ += ' and tksh_angle_%s > -0.9 and tksh_angle_%s < 0.70'%(APP,APP)
    
    return NPTCUTQ

IGNOREDICT = {
    #"Cosmic": 0,
    "trkpid": 1,
    "hits_ratio": 2,
    "shr_score": 3,
    "shrmoliereavg":4,
    "subcluster":5,
    "trkfit":6,
    "n_showers_contained":7,
    "tksh_distance":8,
    "trkshrhitdist2":9,
    "shr_tkfit_dedx_max":10,
    #"secondshower":11,
    "secondshower_Y_nhit":12,
    "tksh_angle":13
}

IGNOREBINS = {
    #"Cosmic": 0,
    "trkpid": np.array([-1,0.02,1]),
    "hits_ratio": np.array([0,0.65,1]),
    "shr_score": np.array([0,0.25,1.0]),
    "shrmoliereavg":np.array([0,2,10,20]),
    "subcluster":np.array([0,7,20]),
    "trkfit":np.array([0,0.7,1.0]),
    "n_showers_contained":np.array([0,1,2,3,4]),
    "tksh_distance":np.array([0,4,10]),
    "trkshrhitdist2":np.array([0,1.5,10]),
    "shr_tkfit_dedx_max":np.array([0,1,3.8,10]),
    #"secondshower":np.array([]),
    "secondshower_Y_nhit":np.array([0,30,100]),
    "tksh_angle":np.array([-1.0,-0.9,0.7,1.0])
}


def ZpQuery(APP):

    # nue preselection
    PRESQ = 'nslice_%s == 1'%APP
    PRESQ += ' and selected_%s == 1'%APP
    PRESQ += ' and shr_energy_tot_cali_%s > 0.07'%APP
    
    #1e0p selection
    ZPPRESEL_all_tracks = PRESQ
    ZPPRESEL_onep_track = ZPPRESEL_all_tracks + ' and n_tracks_contained_%s > 0'%APP
    ZPPRESEL = ZPPRESEL_all_tracks + ' and n_tracks_contained_%s == 0'%APP
    
    ZPLOOSESEL_all_tracks = ZPPRESEL_all_tracks
    ZPLOOSESEL_all_tracks += ' and n_showers_contained_%s == 1'%APP
    ZPLOOSESEL_all_tracks += ' and CosmicIPAll3D_%s > 10.'%APP
    ZPLOOSESEL_all_tracks += ' and CosmicDirAll3D_%s > -0.9 and CosmicDirAll3D_%s < 0.9'%(APP,APP)
    ZPLOOSESEL_all_tracks += ' and shrmoliereavg_%s < 15'%APP
    ZPLOOSESEL_all_tracks += ' and subcluster_%s > 4'%APP
    ZPLOOSESEL_all_tracks += ' and trkfit_%s < 0.65'%APP
    ZPLOOSESEL_all_tracks += ' and secondshower_Y_nhit_%s < 50'%APP
    ZPLOOSESEL_all_tracks += ' and shr_trk_sce_start_y_%s > -100 and shr_trk_sce_start_y_%s < 100'%(APP,APP)
    ZPLOOSESEL_all_tracks += ' and shr_trk_sce_end_y_%s > -100 and shr_trk_sce_end_y_%s < 100 '%(APP,APP)
    ZPLOOSESEL_all_tracks += ' and shr_trk_len_%s < 300.'%APP
    ZPLOOSESEL_onep_track = ZPLOOSESEL_all_tracks + ' and n_tracks_contained_%s > 0'%APP
    ZPLOOSESEL = ZPLOOSESEL_all_tracks + ' and n_tracks_contained_%s == 0'%APP
    
    ZPBDT_all_tracks = ZPLOOSESEL_all_tracks
    ZPBDT_all_tracks += ' and bkg_score_%s > 0.85'%APP
    ZPBDT = ZPBDT_all_tracks + ' and n_tracks_contained_%s == 0'%APP

    return ZPBDT
    
# option:
# 0 -> no query
# 1 -> far-PID
# 2 -> near-PID
# 3 -> signal-PID
# 4 -> signal energy
# 5 -> near energy
# 6 -> far energy
# 7 -> two-shower
# 8 -> high energy + loose box cuts
def NpSidebandQuery(APP,OPTION=0):
    QUERY = 'nslice_%s == 1'%APP
    QUERY += ' and selected_%s == 1'%APP
    QUERY += ' and shr_energy_tot_cali_%s > 0.07'%APP
    QUERY += ' and (_opfilter_pe_beam_%s > 0 and _opfilter_pe_veto_%s < 20)'%(APP,APP)
    
    # 2 shower-cut
    if (OPTION == 7):
        QUERY += ' and n_showers_contained_%s > 1'%APP
        return QUERY
    #return QUERY
    
    #QUERY += ' and n_tracks_contained_%s> 0'%APP
    QUERY += ' and n_showers_contained_%s == 1'%APP
    
    if (OPTION == 1):
        QUERY += ' and (0.0 < pi0_score_%s < 1.0)'%APP
        QUERY += ' and (0.0 < nonpi0_score_%s < 1.0)'%APP
        QUERY += ' and ~((pi0_score_%s > 0.1) and (nonpi0_score_%s > 0.1))'%(APP,APP)
        return QUERY
    if (OPTION == 2):
        QUERY += ' and (0.1 < pi0_score_%s < 1.0)'%APP
        QUERY += ' and (0.1 < nonpi0_score_%s < 1.0)'%APP
        QUERY += ' and ~((pi0_score_%s > 0.67) and (nonpi0_score_%s > 0.70))'%(APP,APP)
        return QUERY
    if (OPTION == 3):
        QUERY += ' and (0.67 < pi0_score_%s < 1.0)'%APP
        QUERY += ' and (0.70 < nonpi0_score_%s < 1.0)'%APP
        QUERY += ' and ~((pi0_score_%s > 1.0) and (nonpi0_score_%s > 1.0))'%(APP,APP)
        # loose box cuts
        QUERY += ' and CosmicIPAll3D_%s > 10.'%APP
        QUERY += ' and trkpid_%s < 0.02'%APP
        QUERY += ' and hits_ratio_%s > 0.50'%APP
        QUERY += ' and shrmoliereavg_%s < 9'%APP
        QUERY += ' and subcluster_%s > 4'%APP
        QUERY += ' and trkfit_%s < 0.65'%APP
        QUERY += ' and n_showers_contained_%s == 1'%APP
        QUERY += ' and tksh_distance_%s < 6.0'%APP
        QUERY += ' and (shr_tkfit_nhits_tot_%s > 1 and shr_tkfit_dedx_max_%s > 0.5 and shr_tkfit_dedx_max_%s < 5.5)'%(APP,APP,APP)
        #QUERY += ' and secondshower_Y_nhit_%s < 50'%APP
        QUERY += ' and tksh_angle_%s > -0.9'%APP
    if (OPTION == 4):
        QUERY += ' and (0.05 < reco_e_%s < 0.75)'%(APP)
    if (OPTION == 5):
        QUERY += ' and (0.75 < reco_e_%s < 1.05)'%(APP)
    if (OPTION == 6):
        QUERY += ' and (1.05 < reco_e_%s < 2.05)'%(APP)
    if (OPTION == 8):
        # loose box cuts
        # very loose box cuts
        QUERY += ' and CosmicIPAll3D_%s > 10.'%(APP)
        QUERY += ' and trkpid_%s < 0.25'%(APP)
        QUERY += ' and hits_ratio_%s > 0.5'%(APP)
        QUERY += ' and trkfit_%s < 0.90'%(APP)
        QUERY += ' and n_showers_contained_%s == 1'%(APP)
        QUERY += ' and tksh_distance_%s < 10.0'%(APP)
        QUERY += ' and tksh_angle_%s > -0.9'%(APP)
        QUERY += ' and (1.05 < reco_e_%s < 2.05)'%(APP)
    return QUERY


# option:
# 0 -> no query
# 1 -> far-PID
# 2 -> near-PID
# 3 -> signal-PID
# 4 -> far energy
# 5 -> far-energy with loose selection
def ZpSidebandQuery(APP,OPTION=0):
    QUERY = 'nslice_%s == 1'%(APP)
    QUERY += ' and selected_%s == 1'%(APP)
    QUERY += ' and contained_fraction_%s > 0.9'%(APP)
    QUERY += ' and shr_energy_tot_cali_%s > 0.07'%(APP)
    QUERY += ' and n_tracks_contained_%s == 0'%(APP)
    QUERY += ' and n_showers_contained_%s == 1'%(APP)
    
    QUERY += ' and CosmicIPAll3D_%s > 10. '%APP
    QUERY += ' and (CosmicDirAll3D_%s <0.9 and CosmicDirAll3D_%s >-0.9)'%(APP,APP)
    
    if (OPTION == 1):
        QUERY += ' and (0.0 < bkg_score_%s < 0.4)'%APP
        return QUERY
    if (OPTION == 2):
        QUERY += ' and (0.4 < bkg_score_%s < 0.72)'%APP
        return QUERY
    if (OPTION == 3):
        QUERY += ' and (0.72 < bkg_score_%s < 1.0)'%APP
    if (OPTION == 4):
        QUERY += ' and reco_e_%s > 0.9'%APP
    if (OPTION == 5):
        QUERY += ' and n_showers_contained_%s == 1'%APP
        QUERY += ' and CosmicIPAll3D_%s > 10.'%APP
        QUERY += ' and CosmicDirAll3D_%s > -0.9 and CosmicDirAll3D_%s < 0.9'%(APP,APP)
        QUERY += ' and shrmoliereavg_%s < 15'%APP
        QUERY += ' and subcluster_%s > 4'%APP
        QUERY += ' and trkfit_%s < 0.65'%APP
        QUERY += ' and secondshower_Y_nhit_%s < 50'%APP
        QUERY += ' and reco_e_%s > 0.9'%APP 
        QUERY += ' and 0.4 < bkg_score_%s'%APP
        
    return QUERY


####################################
# Cuts for NUMU constraint selection
####################################
# returns QUERY, track_cuts for given appendix ('CV' OR 'VAR')
#updated for SCE
#will swap numerical vals at end
FVx = [10,246]#[5,251]
FVy = [-110,110]
FVz = [20,986]

def NUMUQuery(APPEND,verbose=False):
    query = 'nslice_{} == 1'.format(APPEND)
    query += ' and topological_score_{} > 0.06'.format(APPEND)
    query += ' and reco_nu_vtx_sce_x_{} > FVx[0] and reco_nu_vtx_sce_x_{} < FVx[1]'.format(APPEND,APPEND)
    query += ' and reco_nu_vtx_sce_y_{} > FVy[0] and reco_nu_vtx_sce_y_{} < FVy[1]'.format(APPEND,APPEND)
    query += ' and reco_nu_vtx_sce_z_{} > FVz[0] and reco_nu_vtx_sce_z_{} < FVz[1]'.format(APPEND,APPEND)
    query += ' and ( (reco_nu_vtx_sce_z_{} < 675) or (reco_nu_vtx_sce_z_{} > 775) )'.format(APPEND,APPEND) #avoid dead wire region    
    if USECRT: query += ' and (crtveto_{}!=1 or crthitpe_{} < 100.) and (_closestNuCosmicDist_{} > 20.)'.format(APPEND,APPEND,APPEND)

    query = query.replace('FVx[0]',str(FVx[0]))
    query = query.replace('FVy[0]',str(FVy[0]))
    query = query.replace('FVz[0]',str(FVz[0]))
    query = query.replace('FVx[1]',str(FVx[1]))
    query = query.replace('FVy[1]',str(FVy[1]))
    query = query.replace('FVz[1]',str(FVz[1])) 

    if verbose: print ("QUERY: \n",query)

    track_cuts = [
        ('trk_sce_start_x_v_{}'.format(APPEND), '>', FVx[0]),
        ('trk_sce_start_x_v_{}'.format(APPEND), '<', FVx[1]),
        ('trk_sce_start_y_v_{}'.format(APPEND), '>', FVy[0]),
        ('trk_sce_start_y_v_{}'.format(APPEND), '<', FVy[1]),
        ('trk_sce_start_z_v_{}'.format(APPEND), '>', FVz[0]),
        ('trk_sce_start_z_v_{}'.format(APPEND), '<', FVz[1]),
        ('trk_sce_end_x_v_{}'.format(APPEND), '>', FVx[0]),
        ('trk_sce_end_x_v_{}'.format(APPEND), '<', FVx[1]),
        ('trk_sce_end_y_v_{}'.format(APPEND), '>', FVy[0]),
        ('trk_sce_end_y_v_{}'.format(APPEND), '<', FVy[1]),
        ('trk_sce_end_z_v_{}'.format(APPEND), '>', FVz[0]),
        ('trk_sce_end_z_v_{}'.format(APPEND), '<', FVz[1]),
        ('trk_p_quality_v_{}'.format(APPEND), '>', -0.5),
        ('trk_p_quality_v_{}'.format(APPEND), '<', 0.5),
        ('trk_llr_pid_score_v_{}'.format(APPEND), '>', 0.2),
        ('trk_score_v_{}'.format(APPEND), '>', 0.8),
        ('trk_len_v_{}'.format(APPEND), '>', 10),
        ('pfp_generation_v_{}'.format(APPEND), '==', 2),
        ('trk_distance_v_{}'.format(APPEND), '<', 4)
    ]

    if verbose: 
        print("\ntrack cuts:")
        cut_string = ""
        for c,cut in enumerate(track_cuts):
            if c > 0: cut_string += " and "
            if type(cut[1]) == list: cut_string += "( ({} {} {}) or ({} {} {}) )".format(cut[0],cut[1][0],cut[2][0],cut[0],cut[1][1],cut[2][1])
            else: cut_string += "{} {} {}".format(cut[0],cut[1],cut[2])
        print(cut_string)
    
    return query, track_cuts

