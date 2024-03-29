import sys
import localSettings as ls
main_path = ls.main_path
sys.path.append(main_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import nue_booster 

import uproot
import awkward
import plotter

USEBDT = True

QUERY = "nslice == 1 and reco_nu_vtx_sce_x > 5 and reco_nu_vtx_sce_x < 251.  and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 110.  and reco_nu_vtx_sce_z > 20 and reco_nu_vtx_sce_z < 986.  and (reco_nu_vtx_sce_z < 675 or reco_nu_vtx_sce_z > 775)  and topological_score > 0.06  and nslice == 1 and reco_nu_vtx_sce_x > 5 and reco_nu_vtx_sce_x < 251.  and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 110.  and reco_nu_vtx_sce_z > 20 and reco_nu_vtx_sce_z < 986.  and (reco_nu_vtx_sce_z < 675 or reco_nu_vtx_sce_z > 775)  and topological_score > 0.06 & trk2_energy > 0.3" 

# returns the element in a vector at a given index (if out of range, the element is set to defval)
def get_elm_from_vec_idx(myvec,idx,defval=9999.):
    #print ('vector check....')
    #print (idx)
    #print (len(pidv))
    return awkward.fromiter([pidv[tid] if ( (tid<len(pidv) ) & (tid>=0)) else defval for pidv,tid in zip(myvec,idx)])

# this function returns the index in a vector at position argidx after it has been masked and sorted
# the returned index refers to the original (unsorted and unmaksed) vector
def get_idx_from_vec_sort(argidx,vecsort,mask):
    vid = vecsort[mask]
    sizecheck = argidx if argidx>=0 else abs(argidx)-1
    # find the position in the array after masking
    mskd_pos = [v.argsort()[argidx] if len(v)>sizecheck else -1 for v in vid]
    # go back to the corresponding position in the origin array before masking
    result = [[i for i, n in enumerate(m) if n == 1][p] if (p)>=0 else -1 for m,p in zip(mask,mskd_pos)]
    return result

# this function sums all elements in a vector after applying a mask
def sum_elements_from_mask(vector,mask):
    vid = vector[mask]
    result = vid.sum()
    return result

def distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

def cosAngleTwoVecs(vx1,vy1,vz1,vx2,vy2,vz2):
    return (vx1*vx2 + vy1*vy2 + vz1*vz2)/(np.sqrt(vx1**2+vy1**2+vz1**2) * np.sqrt(vx2**2+vy2**2+vz2**2))

def mgg(e1,e2,px1,px2,py1,py2,pz1,pz2):
    return np.sqrt(2*e1*e2*(1-px1*px2-py1*py2-pz1*pz2))

def combs(args):
    res = []
    for i in range(0,len(args)):
        for j in range(i+1,len(args)):
            res.append( (args[i],args[j]) )
    return res

def all_comb_mgg(ev,pxv,pyv,pzv,combs):
    res = []
    for i,j in combs:
        res.append(np.nan_to_num(mgg(ev[i],ev[j],pxv[i],pxv[j],pyv[i],pyv[j],pzv[i],pzv[j])))
    return res

def unique_combs(combs, combs_argsort):
    res = []
    usedargs = []
    for arg in combs_argsort:
        i,j = combs[arg]
        if i in usedargs or j in usedargs: continue
        usedargs.append(i)
        usedargs.append(j)
        res.append(arg)
    return res

def pick_closest_shower(up,df):
    #
    trk1_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
    shr1_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
    #
    # do the best we can to get the right shr2_id
    #
    trk_score_v = up.array("trk_score_v")
    shr_mask = (trk_score_v<0.5)
    pfnhits_v = up.array("pfnhits")
    shr2_id_corr = up.array('shr2_id')-1 # I think we need this -1 to get the right result
    shr2_id_appr = get_idx_from_vec_sort(-2,pfnhits_v,shr_mask)
    shr2_id = np.where((shr2_id_corr>=0)&(shr2_id_corr<df['n_showers_tot']),shr2_id_corr,shr2_id_appr)
    #
    shr_start_x_v   = up.array("shr_start_x_v")
    shr_start_y_v   = up.array("shr_start_y_v")
    shr_start_z_v   = up.array("shr_start_z_v")
    trk_start_x_v   = up.array("trk_start_x_v")
    trk_start_y_v   = up.array("trk_start_y_v")
    trk_start_z_v   = up.array("trk_start_z_v")
    #
    df["shr1_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr1_id,-9999.)
    df["shr1_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr1_id,-9999.)
    df["shr1_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr1_id,-9999.)
    df["shr2_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr2_id,-9999.)
    df["shr2_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr2_id,-9999.)
    df["shr2_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr2_id,-9999.)
    df["trk1_start_x"] = get_elm_from_vec_idx(trk_start_x_v,trk1_id,-9999.)
    df["trk1_start_y"] = get_elm_from_vec_idx(trk_start_y_v,trk1_id,-9999.)
    df["trk1_start_z"] = get_elm_from_vec_idx(trk_start_z_v,trk1_id,-9999.)
    #
    df['tk1sh1_distance'] = np.where((df['n_showers_contained']>0)&(df['n_tracks_contained']>0),\
                                     distance(df['shr1_start_x'],df['shr1_start_y'],df['shr1_start_z'],\
                                              df['trk1_start_x'],df['trk1_start_y'],df['trk1_start_z']),\
                                     9999.)
    df['tk1sh2_distance'] = np.where((df['n_showers_contained']>1)&(df['n_tracks_contained']>0),\
                                     distance(df['shr2_start_x'],df['shr2_start_y'],df['shr2_start_z'],\
                                              df['trk1_start_x'],df['trk1_start_y'],df['trk1_start_z']),\
                                     9999.)
    # set the shr_id
    df['shr_id'] = shr1_id
    df["is_shr2clsr"] = np.zeros_like(df["n_tracks_contained"])
    shr2clsr = (df['n_showers_contained']>1)&(df['n_tracks_contained']>0)&(df['tk1sh2_distance']<df['tk1sh1_distance'])
    df.loc[shr2clsr, 'is_shr2clsr' ] = 1
    #
    # now redefine shower selection variables
    # shr_score
    df["shr2_score"] = get_elm_from_vec_idx(trk_score_v,shr2_id,-9999.)
    df.loc[shr2clsr,"shr_score"] = df["shr2_score"]
    # tksh_distance
    df.loc[shr2clsr,"tksh_distance"] = df['tk1sh2_distance']
    # tksh_angle
    shr_px_v = up.array("shr_px_v")
    shr_py_v = up.array("shr_py_v")
    shr_pz_v = up.array("shr_pz_v")
    df["shr2_px"] = get_elm_from_vec_idx(shr_px_v,shr2_id,-9999.)
    df["shr2_py"] = get_elm_from_vec_idx(shr_py_v,shr2_id,-9999.)
    df["shr2_pz"] = get_elm_from_vec_idx(shr_pz_v,shr2_id,-9999.)
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    df["trk1_dir_x"] = get_elm_from_vec_idx(trk_dir_x_v,trk1_id,-9999.)
    df["trk1_dir_y"] = get_elm_from_vec_idx(trk_dir_y_v,trk1_id,-9999.)
    df["trk1_dir_z"] = get_elm_from_vec_idx(trk_dir_z_v,trk1_id,-9999.)
    df["tk1sh2_angle"] = cosAngleTwoVecs(df["trk1_dir_x"],df["trk1_dir_y"],df["trk1_dir_z"],\
                                         df["shr2_px"],    df["shr2_py"],    df["shr2_pz"])
    df.loc[shr2clsr,"tksh_angle"] = df['tk1sh2_angle']
    # shr_tkfit_dedx_max
    shr_tkfit_dedx_u_v = up.array("shr_tkfit_dedx_u_v")
    shr_tkfit_dedx_v_v = up.array("shr_tkfit_dedx_v_v")
    shr_tkfit_dedx_y_v = up.array("shr_tkfit_dedx_y_v")
    shr_tkfit_nhits_u_v = up.array("shr_tkfit_dedx_nhits_u_v")
    shr_tkfit_nhits_v_v = up.array("shr_tkfit_dedx_nhits_v_v")
    shr_tkfit_nhits_y_v = up.array("shr_tkfit_dedx_nhits_y_v")
    df["shr2_tkfit_dedx_u"] = get_elm_from_vec_idx(shr_tkfit_dedx_u_v,shr2_id,-9999.)
    df["shr2_tkfit_dedx_v"] = get_elm_from_vec_idx(shr_tkfit_dedx_v_v,shr2_id,-9999.)
    df["shr2_tkfit_dedx_y"] = get_elm_from_vec_idx(shr_tkfit_dedx_y_v,shr2_id,-9999.)
    df["shr2_tkfit_nhits_u"] = get_elm_from_vec_idx(shr_tkfit_nhits_u_v,shr2_id,0)
    df["shr2_tkfit_nhits_v"] = get_elm_from_vec_idx(shr_tkfit_nhits_v_v,shr2_id,0)
    df["shr2_tkfit_nhits_y"] = get_elm_from_vec_idx(shr_tkfit_nhits_y_v,shr2_id,0)
    df.loc[shr2clsr, 'shr_tkfit_dedx_U' ] = df["shr2_tkfit_dedx_u"]
    df.loc[shr2clsr, 'shr_tkfit_dedx_V' ] = df["shr2_tkfit_dedx_v"]
    df.loc[shr2clsr, 'shr_tkfit_dedx_Y' ] = df["shr2_tkfit_dedx_y"]
    df.loc[shr2clsr, 'shr_tkfit_nhits_U' ] = df['shr2_tkfit_nhits_u']
    df.loc[shr2clsr, 'shr_tkfit_nhits_V' ] = df['shr2_tkfit_nhits_v']
    df.loc[shr2clsr, 'shr_tkfit_nhits_Y' ] = df['shr2_tkfit_nhits_y']
    # trkfit
    shr_tkfit_nhits_v = up.array("shr_tkfit_nhits_v")
    df["shr2_tkfit_npointsvalid"] = get_elm_from_vec_idx(shr_tkfit_nhits_v,shr2_id,-9999.)
    df["shr2_tkfit_npoints"] = get_elm_from_vec_idx(pfnhits_v,shr2_id,-9999.)
    df.loc[shr2clsr,"shr_tkfit_npointsvalid"] = df["shr2_tkfit_npointsvalid"]
    df.loc[shr2clsr, 'shr_tkfit_npoints' ] = df["shr2_tkfit_npoints"]
    # subcluster
    pfpplanesubclusters_U_v = up.array("pfpplanesubclusters_U")
    pfpplanesubclusters_V_v = up.array("pfpplanesubclusters_V")
    pfpplanesubclusters_Y_v = up.array("pfpplanesubclusters_Y")
    df["shr2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v,shr2_id,0)
    df["shr2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v,shr2_id,0)
    df["shr2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v,shr2_id,0)
    df.loc[shr2clsr,"shrsubclusters0"] = df["shr2subclusters0"]
    df.loc[shr2clsr,"shrsubclusters1"] = df["shr2subclusters1"]
    df.loc[shr2clsr,"shrsubclusters2"] = df["shr2subclusters2"]
    # shrmoliereavg
    shr_moliere_avg_v = up.array("shr_moliere_avg_v")
    df["shr2_moliere_avg"] = get_elm_from_vec_idx(shr_moliere_avg_v,shr2_id,-9999.)
    df.loc[shr2clsr,"shrmoliereavg"] = df['shr2_moliere_avg']
    # trkshrhitdist2
    df.loc[shr2clsr,"trkshrhitdist2"] = df['tksh_distance']
    #
    return

def process_uproot(up,df):
    #
    trk_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
    shr_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
    #
    trk_llr_pid_v = up.array('trk_llr_pid_score_v')
    trk_calo_energy_y_v = up.array('trk_calo_energy_y_v')
    trk_energy_proton_v = up.array('trk_energy_proton_v')
    #
    trk_llr_pid_v_sel = get_elm_from_vec_idx(trk_llr_pid_v,trk_id, np.nan)
    trk_calo_energy_y_sel = get_elm_from_vec_idx(trk_calo_energy_y_v,trk_id)
    trk_energy_proton_sel = get_elm_from_vec_idx(trk_energy_proton_v,trk_id)
    df['trkpid'] = trk_llr_pid_v_sel
    df['trackcaloenergy'] = trk_calo_energy_y_sel
    df['protonenergy'] = trk_energy_proton_sel
    trk_sce_start_x_v = up.array('trk_sce_start_x_v')
    trk_sce_start_y_v = up.array('trk_sce_start_y_v')
    trk_sce_start_z_v = up.array('trk_sce_start_z_v')
    trk_sce_end_x_v = up.array('trk_sce_end_x_v')
    trk_sce_end_y_v = up.array('trk_sce_end_y_v')
    trk_sce_end_z_v = up.array('trk_sce_end_z_v')
    df['shr_trk_sce_start_x'] = get_elm_from_vec_idx(trk_sce_start_x_v,shr_id)
    df['shr_trk_sce_start_y'] = get_elm_from_vec_idx(trk_sce_start_y_v,shr_id)
    df['shr_trk_sce_start_z'] = get_elm_from_vec_idx(trk_sce_start_z_v,shr_id)
    df['shr_trk_sce_end_x'] = get_elm_from_vec_idx(trk_sce_end_x_v,shr_id)
    df['shr_trk_sce_end_y'] = get_elm_from_vec_idx(trk_sce_end_y_v,shr_id)
    df['shr_trk_sce_end_z'] = get_elm_from_vec_idx(trk_sce_end_z_v,shr_id)
    df['shr_trk_len'] = distance(df['shr_trk_sce_start_x'],df['shr_trk_sce_start_y'],df['shr_trk_sce_start_z'], \
                                 df['shr_trk_sce_end_x'],  df['shr_trk_sce_end_y'],  df['shr_trk_sce_end_z'])
    df['mevcm'] = 1000 * df['shr_energy_tot_cali'] / df['shr_trk_len']
    #
    df["slclnhits"] = up.array("pfnhits").sum()
    df["slclnunhits"] = up.array("pfnunhits").sum()
    #
    pfp_pdg_v = up.array('backtracked_pdg')
    trk_pdg = get_elm_from_vec_idx(pfp_pdg_v,trk_id)
    df['trk_pdg'] = trk_pdg
    pfp_pur_v = up.array('backtracked_purity')
    trk_pur = get_elm_from_vec_idx(pfp_pur_v,trk_id)
    df['trk_pur'] = trk_pur
    pfp_cmp_v = up.array('backtracked_completeness')
    trk_cmp = get_elm_from_vec_idx(pfp_cmp_v,trk_id)
    df['trk_cmp'] = trk_cmp
    #
    # fix elec_pz for positrons
    nu_pdg = up.array('nu_pdg')
    ccnc = up.array('ccnc')
    mc_pdg = up.array('mc_pdg')
    mc_E = up.array('mc_E')
    mc_px = up.array('mc_px')
    mc_py = up.array('mc_py')
    mc_pz = up.array('mc_pz')
    elec_pz = up.array('elec_pz')
    positr_mask = (mc_pdg==-11)
    mostEpositrIdx = get_idx_from_vec_sort(-1,mc_E,positr_mask)
    mc_E_posi = get_elm_from_vec_idx(mc_E,mostEpositrIdx,-9999.)
    mc_px_posi = get_elm_from_vec_idx(mc_px,mostEpositrIdx,-9999.)
    mc_py_posi = get_elm_from_vec_idx(mc_py,mostEpositrIdx,-9999.)
    mc_pz_posi = get_elm_from_vec_idx(mc_pz,mostEpositrIdx,-9999.)
    mc_p_posi = np.sqrt(mc_px_posi*mc_px_posi + mc_py_posi*mc_py_posi + mc_pz_posi*mc_pz_posi)
    df['positr_px'] = np.where((mc_E_posi>0),mc_px_posi/mc_p_posi,-9999.)
    df['positr_py'] = np.where((mc_E_posi>0),mc_py_posi/mc_p_posi,-9999.)
    df['positr_pz'] = np.where((mc_E_posi>0),mc_pz_posi/mc_p_posi,-9999.)
    df.loc[(nu_pdg==-12)&(ccnc==0)&(elec_pz<-2), 'elec_px' ] = df['positr_px']
    df.loc[(nu_pdg==-12)&(ccnc==0)&(elec_pz<-2), 'elec_py' ] = df['positr_py']
    df.loc[(nu_pdg==-12)&(ccnc==0)&(elec_pz<-2), 'elec_pz' ] = df['positr_pz']
    #
    #get true proton angle
    prot_mask = (mc_pdg==2212)
    mostEprotIdx = get_idx_from_vec_sort(-1,mc_E,prot_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E,mostEprotIdx,-9999.)
    mc_px_prot = get_elm_from_vec_idx(mc_px,mostEprotIdx,-9999.)
    mc_py_prot = get_elm_from_vec_idx(mc_py,mostEprotIdx,-9999.)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz,mostEprotIdx,-9999.)
    mc_p_prot = np.sqrt(mc_px_prot*mc_px_prot + mc_py_prot*mc_py_prot + mc_pz_prot*mc_pz_prot)
    df['proton_pz'] = np.where((mc_E_prot>0),mc_pz_prot/mc_p_prot,-9999.)
    #
    # true proton length (assuming straight line)
    mc_vx = up.array('mc_vx')
    mc_vy = up.array('mc_vy')
    mc_vz = up.array('mc_vz')
    mc_endx = up.array('mc_endx')
    mc_endy = up.array('mc_endy')
    mc_endz = up.array('mc_endz')
    p_vx = get_elm_from_vec_idx(mc_vx,mostEprotIdx,-9999.)
    p_vy = get_elm_from_vec_idx(mc_vy,mostEprotIdx,-9999.)
    p_vz = get_elm_from_vec_idx(mc_vz,mostEprotIdx,-9999.)
    p_endx = get_elm_from_vec_idx(mc_endx,mostEprotIdx,-9999.)
    p_endy = get_elm_from_vec_idx(mc_endy,mostEprotIdx,-9999.)
    p_endz = get_elm_from_vec_idx(mc_endz,mostEprotIdx,-9999.)
    df['proton_len'] = np.sqrt( (p_endx-p_vx)*(p_endx-p_vx) + (p_endy-p_vy)*(p_endy-p_vy) + (p_endz-p_vz)*(p_endz-p_vz) )
    #
    trk_score_v = up.array("trk_score_v")
    shr_mask = (trk_score_v<0.5)
    trk_mask = (trk_score_v>0.5)
    df['n_tracks_tot'] = trk_mask.sum()
    df['n_showers_tot'] = shr_mask.sum()
    trk_len_v = up.array("trk_len_v")
    df["n_trks_gt10cm"] = (trk_len_v[trk_mask>=0.5]>10).sum()
    df["n_trks_gt25cm"] = (trk_len_v[trk_mask>=0.5]>25).sum()
    trk_distance_v = up.array("trk_distance_v")
    df["n_tracks_attach"] = (trk_distance_v[trk_mask>=0.5]<3).sum()
    df["n_protons_attach"] = ((trk_distance_v[trk_mask>=0.5]<3)&(trk_llr_pid_v[trk_mask>=0.5]<0.02)).sum()
    #
    pfnhits_v = up.array("pfnhits")
    trk_id_all = get_idx_from_vec_sort(-1,pfnhits_v,trk_mask) # this includes also uncontained tracks
    #
    shr_start_x_v   = up.array("shr_start_x_v")
    shr_start_y_v   = up.array("shr_start_y_v")
    shr_start_z_v   = up.array("shr_start_z_v")
    df["shr_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr_id)
    df["shr_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr_id)
    df["shr_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr_id)
    trk_start_x_v   = up.array("trk_start_x_v")
    trk_start_y_v   = up.array("trk_start_y_v")
    trk_start_z_v   = up.array("trk_start_z_v")
    df["trk1_start_x_alltk"] = get_elm_from_vec_idx(trk_start_x_v,trk_id_all)
    df["trk1_start_y_alltk"] = get_elm_from_vec_idx(trk_start_y_v,trk_id_all)
    df["trk1_start_z_alltk"] = get_elm_from_vec_idx(trk_start_z_v,trk_id_all)
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    df["trk1_dir_x_alltk"] = get_elm_from_vec_idx(trk_dir_x_v,trk_id_all)
    df["trk1_dir_y_alltk"] = get_elm_from_vec_idx(trk_dir_y_v,trk_id_all)
    df["trk1_dir_z_alltk"] = get_elm_from_vec_idx(trk_dir_z_v,trk_id_all)
    #
    # tksh_distance and tksh_angle for track with most hits, regardless of containment
    #
    if np.shape(df['shr_start_x']) != np.shape(df['trk1_start_x_alltk']): return
    df['tk1sh1_distance_alltk'] = np.where(df['n_tracks_tot']==0,99999,
                                           distance(df['shr_start_x'],       df['shr_start_y'],       df['shr_start_z'],\
                                                    df['trk1_start_x_alltk'],df['trk1_start_y_alltk'],df['trk1_start_z_alltk']))
    df["tk1sh1_angle_alltk"] = np.where(df['n_tracks_tot']==0,99999,
                                  cosAngleTwoVecs(df["trk1_dir_x_alltk"],df["trk1_dir_y_alltk"],df["trk1_dir_z_alltk"],\
                                                  df["shr_px"],          df["shr_py"],          df["shr_pz"]))
    #
    df['shr_ptot'] = np.sqrt( df['shr_px']**2 + df['shr_py']**2 + df['shr_pz']**2)
    df['shr_px_unit'] = df['shr_px'] / df['shr_ptot']
    df['shr_py_unit'] = df['shr_py'] / df['shr_ptot']
    df['shr_pz_unit'] = df['shr_pz'] / df['shr_ptot']
    
    # return # DAVIDC
    
    #
    # fix the 'subcluster' bug (in case of more than one shower, it comes from the one with least hits, not the one with most)
    # so we overwrite the dataframe column taking the correct value from the corrsponding vector branches
    #
    pfpplanesubclusters_U_v = up.array("pfpplanesubclusters_U")
    pfpplanesubclusters_V_v = up.array("pfpplanesubclusters_V")
    pfpplanesubclusters_Y_v = up.array("pfpplanesubclusters_Y")
    df["shrsubclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v,shr_id,0)
    df["shrsubclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v,shr_id,0)
    df["shrsubclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v,shr_id,0)
    #
    # do the best we can to get the right shr2_id
    #
    shr2_id_corr = up.array('shr2_id')-1 # I think we need this -1 to get the right result
    shr2_id_appr = get_idx_from_vec_sort(-2,pfnhits_v,shr_mask)
    shr2_id = np.where((shr2_id_corr>=0)&(shr2_id_corr<df['n_showers_tot']),shr2_id_corr,shr2_id_appr)
    #
    df["shr2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v,shr2_id,0)
    df["shr2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v,shr2_id,0)
    df["shr2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v,shr2_id,0)
    df['subcluster2tmp'] = df['shr2subclusters0'] + df['shr2subclusters1'] + df['shr2subclusters2']
    #
    df["shr2_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr2_id,-9999.)
    df["shr2_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr2_id,-9999.)
    df["shr2_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr2_id,-9999.)
    df["trk1_start_x"] = get_elm_from_vec_idx(trk_start_x_v,trk_id,-9999.)
    df["trk1_start_y"] = get_elm_from_vec_idx(trk_start_y_v,trk_id,-9999.)
    df["trk1_start_z"] = get_elm_from_vec_idx(trk_start_z_v,trk_id,-9999.)
    df['tk1sh2_distance'] = np.where((df['n_showers_contained']>1)&(df['n_tracks_contained']>0),\
                                     distance(df['shr2_start_x'], df['shr2_start_y'], df['shr2_start_z'],\
                                     df['trk1_start_x'],df['trk1_start_y'],df['trk1_start_z']),\
                                     9999.)
    #
    #df['sh1sh2_distance'] = np.where(df['n_showers_contained']>1,\
    #                                 distance(df['shr2_start_x'], df['shr2_start_y'], df['shr2_start_z'],\
    #                                 df['shr_start_x'],df['shr_start_y'],df['shr_start_z']),\
    #                                 9999.)
    #
    df['shr2pid'] = get_elm_from_vec_idx(trk_llr_pid_v,shr2_id,9999.)
    df['shr2_score'] = get_elm_from_vec_idx(trk_score_v,shr2_id,9999.)
    shr_moliere_avg_v = up.array("shr_moliere_avg_v")
    df["shr2_moliereavg"] = get_elm_from_vec_idx(shr_moliere_avg_v,shr2_id,-9999.)
    #
    #df.drop(columns=['shr_start_x', 'shr_start_y', 'shr_start_z'])
    #df.drop(columns=['trk1_start_x_alltk', 'trk1_start_y_alltk', 'trk1_start_z_alltk'])
    #df.drop(columns=['trk1_dir_x_alltk', 'trk1_dir_y_alltk', 'trk1_dir_z_alltk'])
    #df.drop(columns=['shr2subclusters0', 'shr2subclusters1', 'shr2subclusters2'])
    #
    #pick_closest_shower(up,df)
    #
    #
    return

def process_uproot_recoveryvars(up,df):
 
    #
    # data events where recovery matters should have shr2_id and trk2_id properly set
    #
    trk_id = up.array('trk_id')-1 # I think we need this -1 to get the right result
    shr_id = up.array('shr_id')-1 # I think we need this -1 to get the right result
    trk2_id = up.array('trk2_id')-1 # I think we need this -1 to get the right result
    shr2_id = up.array('shr2_id')-1 # I think we need this -1 to get the right result
    #
    shr_energy_y_v = up.array("shr_energy_y_v")

    # Fix of the 9999 bug
    #df["trk2_energy"] = get_elm_from_vec_idx(shr_energy_y_v, trk2_id, 0.0)
    #df["shr2_energy"] = get_elm_from_vec_idx(shr_energy_y_v, shr2_id, 0.0)
    # Old code
    df["trk2_energy"] = get_elm_from_vec_idx(shr_energy_y_v,trk2_id,-9999.)
    df["shr2_energy"] = get_elm_from_vec_idx(shr_energy_y_v,shr2_id,-9999.)

    #
    shr_start_x_v   = up.array("shr_start_x_v")
    shr_start_y_v   = up.array("shr_start_y_v")
    shr_start_z_v   = up.array("shr_start_z_v")
    df["shr1_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr_id,-9999.)
    df["shr2_start_x"] = get_elm_from_vec_idx(shr_start_x_v,shr2_id,-9999.)
    df["shr1_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr_id,-9999.)
    df["shr2_start_y"] = get_elm_from_vec_idx(shr_start_y_v,shr2_id,-9999.)
    df["shr1_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr_id,-9999.)
    df["shr2_start_z"] = get_elm_from_vec_idx(shr_start_z_v,shr2_id,-9999.)
    #
    df["shr12_start_dx"] = df["shr2_start_x"]-df["shr1_start_x"]
    df["shr12_start_dy"] = df["shr2_start_y"]-df["shr1_start_y"]
    df["shr12_start_dz"] = df["shr2_start_z"]-df["shr1_start_z"]
    #
    df["shr12_cos_p1_dstart"] = np.where((df['n_showers_contained']<2)|(df["shr2_energy"]<0)|(df["shr12_start_dx"]==0),-9999.,
                                   cosAngleTwoVecs(df["shr12_start_dx"],df["shr12_start_dy"],df["shr12_start_dz"],\
                                                   df["shr_px"],        df["shr_py"],        df["shr_pz"]))
    #
    trk_len_v = up.array("trk_len_v")
    df["trk1_len"] = get_elm_from_vec_idx(trk_len_v,trk_id,-9999.)
    df["trk2_len"] = get_elm_from_vec_idx(trk_len_v,trk2_id,-9999.)
    #
    trk_distance_v = up.array("trk_distance_v")
    df["trk1_distance"] = get_elm_from_vec_idx(trk_distance_v,trk_id,-9999.)
    df["trk2_distance"] = get_elm_from_vec_idx(trk_distance_v,trk2_id,-9999.)
    #
    trk_llr_pid_v = up.array('trk_llr_pid_score_v')
    df["trk1_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v,trk_id,np.nan)
    df["trk2_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v,trk2_id,np.nan)
    #
    pfnhits_v = up.array("pfnhits")
    df["trk1_nhits"] = get_elm_from_vec_idx(pfnhits_v,trk_id,-9999.)
    df["trk2_nhits"] = get_elm_from_vec_idx(pfnhits_v,trk2_id,-9999.)
    df["shr1_nhits"] = get_elm_from_vec_idx(pfnhits_v,shr_id,-9999.)
    df["shr2_nhits"] = get_elm_from_vec_idx(pfnhits_v,shr2_id,-9999.)
    #
    trk_start_x_v   = up.array("trk_start_x_v")
    trk_start_y_v   = up.array("trk_start_y_v")
    trk_start_z_v   = up.array("trk_start_z_v")
    df["trk1_start_x"] = get_elm_from_vec_idx(trk_start_x_v,trk_id,-9999.)
    df["trk2_start_x"] = get_elm_from_vec_idx(trk_start_x_v,trk2_id,-9999.)
    df["trk1_start_y"] = get_elm_from_vec_idx(trk_start_y_v,trk_id,-9999.)
    df["trk2_start_y"] = get_elm_from_vec_idx(trk_start_y_v,trk2_id,-9999.)
    df["trk1_start_z"] = get_elm_from_vec_idx(trk_start_z_v,trk_id,-9999.)
    df["trk2_start_z"] = get_elm_from_vec_idx(trk_start_z_v,trk2_id,-9999.)
    df['tk1sh1_distance'] = np.where((df['n_showers_contained']>0)&(df['n_tracks_contained']>0),\
                                     distance(df['shr_start_x'], df['shr_start_y'], df['shr_start_z'],\
                                              df['trk1_start_x'],df['trk1_start_y'],df['trk1_start_z']),\
                                     9999.)
    df['tk2sh1_distance'] = np.where((df['n_showers_contained']>0)&(df['n_tracks_contained']>1),\
                                     distance(df['shr_start_x'], df['shr_start_y'], df['shr_start_z'],\
                                     df['trk2_start_x'],df['trk2_start_y'],df['trk2_start_z']),\
                                     9999.)
    df['tk1tk2_distance'] = np.where(df['n_tracks_contained']>1,\
                                     distance(df['trk1_start_x'],df['trk1_start_y'],df['trk1_start_z'],\
                                     df['trk2_start_x'],df['trk2_start_y'],df['trk2_start_z']),\
                                     9999.)
    #
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    df["trk1_dir_x"] = get_elm_from_vec_idx(trk_dir_x_v,trk_id,-9999.)
    df["trk2_dir_x"] = get_elm_from_vec_idx(trk_dir_x_v,trk2_id,-9999.)
    df["trk1_dir_y"] = get_elm_from_vec_idx(trk_dir_y_v,trk_id,-9999.)
    df["trk2_dir_y"] = get_elm_from_vec_idx(trk_dir_y_v,trk2_id,-9999.)
    df["trk1_dir_z"] = get_elm_from_vec_idx(trk_dir_z_v,trk_id,-9999.)
    df["trk2_dir_z"] = get_elm_from_vec_idx(trk_dir_z_v,trk2_id,-9999.)

    df["tk1sh1_angle"] = np.where((df['n_tracks_contained']<1)|(df['n_showers_contained']<1),-9999.,
                            cosAngleTwoVecs(df["trk1_dir_x"],df["trk1_dir_y"],df["trk1_dir_z"],\
                                            df["shr_px"],    df["shr_py"],    df["shr_pz"]))
    df["tk2sh1_angle"] = np.where((df['n_tracks_contained']<2)|(df['n_showers_contained']<1),-9999.,
                            cosAngleTwoVecs(df["trk2_dir_x"],df["trk2_dir_y"],df["trk2_dir_z"],\
                                            df["shr_px"],    df["shr_py"],    df["shr_pz"]))
    df["tk1tk2_angle"] = np.where(df['n_tracks_contained']<2,-9999.,
                            cosAngleTwoVecs(df["trk1_dir_x"],df["trk1_dir_y"],df["trk1_dir_z"],\
                                            df["trk2_dir_x"],df["trk2_dir_y"],df["trk2_dir_z"]))
    #
    # todo: update also other variables, not used in the selection
    #
    # try to recover cases where the 2nd shower is split from the main one
    # note: we do not remake the shower pfp, so we ignore differences on
    # shr_score, shr_tkfit_dedx_max, trkfit since they are negligible
    # note2: in principle this can be done for 0p as well, but we focus only on Np for now
    #
    df["is_shr2splt"] = np.zeros_like(df["n_tracks_contained"])
    shr2splt = ((df["n_tracks_contained"]>0) & (df["n_showers_contained"]>1) &\
                (df['shr12_cos_p1_dstart'] > 0.95) & (df['tk1sh2_distance'] > 60) &\
                (df['shr_score']<0.1) & ((df["shrsubclusters0"]+df["shrsubclusters1"]+df["shrsubclusters2"])>3))
    df.loc[shr2splt, 'is_shr2splt' ] = 1
    df.loc[shr2splt, 'n_showers_contained' ] = 1 #assume this happens to nues only! previously: = df["n_showers_contained"]-1
    pfpplanesubclusters_U_v = up.array("pfpplanesubclusters_U")
    pfpplanesubclusters_V_v = up.array("pfpplanesubclusters_V")
    pfpplanesubclusters_Y_v = up.array("pfpplanesubclusters_Y")
    df["shr2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v,shr2_id,0)
    df["shr2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v,shr2_id,0)
    df["shr2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v,shr2_id,0)
    df.loc[shr2splt, 'shrsubclusters0' ] = df["shrsubclusters0"] + df["shr2subclusters0"]
    df.loc[shr2splt, 'shrsubclusters1' ] = df["shrsubclusters1"] + df["shr2subclusters1"]
    df.loc[shr2splt, 'shrsubclusters2' ] = df["shrsubclusters2"] + df["shr2subclusters2"]
    df.loc[shr2splt & (df["shr1shr2moliereavg"]>0), 'shrmoliereavg' ] = df["shr1shr2moliereavg"]
    #
    # try to recover cases where the leading track is spurious (more than 30 cm away from nu vtx)
    # note: we redefine all track-related variables from trk2 (except pt and p for now),
    # and remove the contribution of trk1 from hit counting and energy calculation
    #
    df["is_trk1bad"] = np.zeros_like(df["n_tracks_contained"])
    trk1bad = ((df["n_tracks_contained"]>1) & (df['trk_distance'] > 30.) & (df["is_shr2splt"]==0))
    df.loc[trk1bad, 'is_trk1bad' ] = 1
    df.loc[trk1bad, 'trkpid' ] = df["trk2_llr_pid"]
    df.loc[trk1bad, 'tksh_distance' ] = df["tk2sh1_distance"]
    df.loc[trk1bad, 'tksh_angle' ] = df["tk2sh1_angle"]
    df.loc[trk1bad, 'hits_ratio' ] = df["shr_hits_tot"]/(df["shr_hits_tot"]+df["trk_hits_tot"]-df["trk1_nhits"])
    df.loc[trk1bad, 'trk_len' ] = df["trk2_len"]
    df.loc[trk1bad, 'trk_distance' ] = df["trk2_distance"]
    trk_score_v = up.array("trk_score_v")
    df["trk2_score"] = get_elm_from_vec_idx(trk_score_v,trk2_id,-9999.)
    trk_energy_proton_v = up.array('trk_energy_proton_v')
    df["trk2_protonenergy"] = get_elm_from_vec_idx(trk_energy_proton_v,trk2_id,-9999.)
    trk_theta_v = up.array("trk_theta_v")
    df["trk2_theta"] = get_elm_from_vec_idx(trk_theta_v,trk2_id,-9999.)
    trk_phi_v = up.array("trk_phi_v")
    df["trk2_phi"] = get_elm_from_vec_idx(trk_phi_v,trk2_id,-9999.)
    df.loc[trk1bad, 'trk_score' ] = df["trk2_score"]
    df.loc[trk1bad, 'protonenergy' ] = df["trk2_protonenergy"]
    df.loc[trk1bad, 'trk_theta' ] = df["trk2_theta"]
    df.loc[trk1bad, 'trk_phi' ] = df["trk2_phi"]
    df.loc[trk1bad, 'trkshrhitdist2' ] = df["trk2shrhitdist2"]
    df.loc[trk1bad, 'n_tracks_contained' ] = df["n_tracks_contained"]-1


    df.loc[trk1bad, 'trk_energy_tot'] = df["trk_energy_tot"]-df["trk_energy"]

    # note: we should redefine also pt, p
    #
    # try to recover cases where the 2nd track is actually the start of the shower
    # we need to redefine almost all shower variables (including dedx, which is tricky)
    #
    df["is_trk2srtshr"] = np.zeros_like(df["n_tracks_contained"])
    trk2srtshr = ((df["n_tracks_contained"]>1) & (df['tk2sh1_angle']>0.98) & (df['tk1tk2_distance']<df['tksh_distance']) & \
                  (df['shr_score']<0.1) & (df["is_shr2splt"]==0) & (df["is_trk1bad"]==0))
    df.loc[trk2srtshr, 'is_trk2srtshr' ] = 1
    #
    shr_tkfit_dedx_u_v = up.array("shr_tkfit_dedx_u_v")
    shr_tkfit_dedx_v_v = up.array("shr_tkfit_dedx_v_v")
    shr_tkfit_dedx_y_v = up.array("shr_tkfit_dedx_y_v")
    shr_tkfit_nhits_u_v = up.array("shr_tkfit_dedx_nhits_u_v")
    shr_tkfit_nhits_v_v = up.array("shr_tkfit_dedx_nhits_v_v")
    shr_tkfit_nhits_y_v = up.array("shr_tkfit_dedx_nhits_y_v")
    df["trk2_tkfit_dedx_u"] = get_elm_from_vec_idx(shr_tkfit_dedx_u_v,trk2_id,-9999.)
    df["trk2_tkfit_dedx_v"] = get_elm_from_vec_idx(shr_tkfit_dedx_v_v,trk2_id,-9999.)
    df["trk2_tkfit_dedx_y"] = get_elm_from_vec_idx(shr_tkfit_dedx_y_v,trk2_id,-9999.)
    df["trk2_tkfit_nhits_u"] = get_elm_from_vec_idx(shr_tkfit_nhits_u_v,trk2_id,0)
    df["trk2_tkfit_nhits_v"] = get_elm_from_vec_idx(shr_tkfit_nhits_v_v,trk2_id,0)
    df["trk2_tkfit_nhits_y"] = get_elm_from_vec_idx(shr_tkfit_nhits_y_v,trk2_id,0)
    df["trk2_tkfit_nhits_tot"] = df["trk2_tkfit_nhits_u"]+df["trk2_tkfit_nhits_v"]+df["trk2_tkfit_nhits_y"]    
    df["trk2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v,trk2_id,0)
    df["trk2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v,trk2_id,0)
    df["trk2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v,trk2_id,0)
    #
    df.loc[trk2srtshr, 'tksh_distance' ] = df["tk1tk2_distance"]
    df.loc[trk2srtshr, 'tksh_angle' ] = df["tk1tk2_angle"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_dedx_U' ] = df["trk2_tkfit_dedx_u"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_dedx_V' ] = df["trk2_tkfit_dedx_v"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_dedx_Y' ] = df["trk2_tkfit_dedx_y"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_nhits_U' ] = df['trk2_tkfit_nhits_u']
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_nhits_V' ] = df['trk2_tkfit_nhits_v']
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"]>0), 'shr_tkfit_nhits_Y' ] = df['trk2_tkfit_nhits_y']
    df.loc[trk2srtshr, 'hits_ratio' ] = (df["shr_hits_tot"]+df["trk2_nhits"])/(df["shr_hits_tot"]+df["trk_hits_tot"])
    #
    df.loc[trk2srtshr, 'shr_tkfit_npointsvalid' ] = df["shr_tkfit_npointsvalid"] + df["trk2_nhits"] #patched!
    # other option... taking the track fit npoints for both (results do not change)
    #shr_tkfit_nhits_v = up.array("shr_tkfit_nhits_v")
    #df["trk2_tkfit_npointsvalid"] = get_elm_from_vec_idx(shr_tkfit_nhits_v,trk2_id,-9999.)
    #df.loc[trk2srtshr, 'shr_tkfit_npointsvalid' ] = df["shr_tkfit_npointsvalid"] + df["trk2_tkfit_npointsvalid"]
    #df.loc[trk2srtshr, 'shr_tkfit_npoints' ] = df["shr_tkfit_npoints"] + df["trk2_nhits"]
    #
    df.loc[trk2srtshr & (df["trk1trk2hitdist2"]>0) & (df["trkshrhitdist2"]>0) & (df["trk1trk2hitdist2"]<df["trkshrhitdist2"]), 'trkshrhitdist2' ] = df["trk1trk2hitdist2"]
    df.loc[trk2srtshr & (df["trk1trk2hitdist2"]>0) & (df["trkshrhitdist2"]<0), 'trkshrhitdist2' ] = df["trk1trk2hitdist2"]
    df.loc[trk2srtshr, 'shrsubclusters0' ] = df["shrsubclusters0"] + df["trk2subclusters0"]
    df.loc[trk2srtshr, 'shrsubclusters1' ] = df["shrsubclusters1"] + df["trk2subclusters1"]
    df.loc[trk2srtshr, 'shrsubclusters2' ] = df["shrsubclusters2"] + df["trk2subclusters2"]
    df.loc[trk2srtshr & (df["shr1trk2moliereavg"]>0), 'shrmoliereavg' ] = df["shr1trk2moliereavg"]
    df.loc[trk2srtshr, 'n_tracks_contained' ] = df["n_tracks_contained"]-1
    df.loc[trk2srtshr, 'trk_energy_tot'] = df["trk_energy_tot"]-df["trk2_protonenergy"]
    df.loc[df["trk2_energy"]<0., "trk2_energy"] = 0.
    df["trk2_energy_cali"] = 0.001 * df["trk2_energy"] * df["shr_energy_tot_cali"] / df["shr_energy_tot"]
    df.loc[trk2srtshr, 'shr_energy_tot_cali'] = df["shr_energy_tot_cali"]+df["trk2_energy_cali"]
    #
    # try to recover cases where the 2nd shower is actually an attached proton
    #
    df["is_shr2prtn"] = np.zeros_like(df["n_tracks_contained"])
    shr2prtn = ((df["n_showers_contained"]>1) & (df['tk1sh2_distance'] < 6.0) & (df["subcluster2tmp"]<=4) & (df["shr2pid"]<0.02))
    df.loc[shr2prtn, 'is_shr2prtn' ] = 1
    df.loc[shr2prtn, 'n_showers_contained' ] = df["n_showers_contained"]-1
    df.loc[shr2prtn, 'n_tracks_contained' ] = df["n_tracks_contained"]+1
    df["shr2_protonenergy"] = get_elm_from_vec_idx(trk_energy_proton_v,shr2_id,-9999.)
    df.loc[shr2prtn, 'trk_energy_tot'] = df["trk_energy_tot"]+df["shr2_protonenergy"]
    df.loc[df["shr2_energy"]<0., "shr2_energy"] = 0.
    df["shr2_energy_cali"] = 0.001 * df["shr2_energy"] * df["shr_energy_tot_cali"] / df["shr_energy_tot"]

    df.loc[shr2prtn, 'shr_energy_tot_cali'] = df["shr_energy_tot_cali"]-df["trk2_energy_cali"]
    #
    # try to recover cases where the leading track is embedded in the shower
    # todo: check that the two overlap, i.e. the shower is not downstream the track
    # todo: use distance/angle/dedx from the object closest to trk2
    # todo: in principle we should update also moliere angle and subcluster
    # FOR NOW WE JUST TAG THEM AND DO NOT TRY TO RECOVER
    #
    df["is_trk1embd"] = np.zeros_like(df["n_tracks_contained"])
    trk1embd = ((df["n_tracks_contained"]>1) & (df['tksh_angle'] > 0.99) & (df["is_trk1bad"]==0))
    df.loc[trk1embd, 'is_trk1embd' ] = 1
    #df.loc[trk1embd, 'trkpid' ] = df["trk2_llr_pid"]
    #df.loc[trk1embd, 'tksh_distance' ] = df["tk2sh1_distance"]
    #df.loc[trk1embd, 'tksh_angle' ] = df["tk2sh1_angle"]
    #df.loc[trk1embd, 'hits_ratio' ] = (df["shr_hits_tot"]+df["trk1_nhits"])/(df["shr_hits_tot"]+df["trk_hits_tot"])
    #df.loc[trk1embd, 'trkshrhitdist2' ] = df["tk2sh1_distance"] #patched!
    #df.loc[trk1embd, 'n_tracks_contained' ] = df["n_tracks_contained"]-1
    #
    # Let's save memory by dropping some stuff we just used and won't use anymore
    #
    df.drop(columns=['shr1_start_x', 'shr1_start_y', 'shr1_start_z'])
    df.drop(columns=['shr2_start_x', 'shr2_start_y', 'shr2_start_z'])
    df.drop(columns=['shr12_start_dx', 'shr12_start_dy', 'shr12_start_dz'])
    #df.drop(columns=['shr2_energy'])
    df.drop(columns=['trk1_len', 'trk2_len'])
    df.drop(columns=['trk1_distance', 'trk2_distance'])
    df.drop(columns=['trk1_llr_pid', 'trk2_llr_pid'])
    df.drop(columns=['trk1_nhits', 'trk2_nhits'])
    df.drop(columns=['trk1_start_x', 'trk1_start_y', 'trk1_start_z'])
    df.drop(columns=['trk2_start_x', 'trk2_start_y', 'trk2_start_z'])
    df.drop(columns=['trk1_dir_x', 'trk1_dir_y', 'trk1_dir_z'])
    df.drop(columns=['trk2_dir_x', 'trk2_dir_y', 'trk2_dir_z'])
    df.drop(columns=['shr2subclusters0', 'shr2subclusters1', 'shr2subclusters2'])
    df.drop(columns=['trk2_score', 'trk2_protonenergy'])
    df.drop(columns=['trk2_theta', 'trk2_phi'])
    df.drop(columns=['trk2_tkfit_dedx_u', 'trk2_tkfit_dedx_v', 'trk2_tkfit_dedx_y'])
    df.drop(columns=['trk2_tkfit_nhits_u', 'trk2_tkfit_nhits_v', 'trk2_tkfit_nhits_y'])
    df.drop(columns=['trk2_tkfit_nhits_tot'])
    df.drop(columns=['trk2subclusters0', 'trk2subclusters1', 'trk2subclusters2'])
    df.drop(columns=['trk2_energy', 'trk2_energy_cali'])
    #
    return


def process_uproot_numu(up,df):
    #
    trk_llr_pid_v = up.array("trk_llr_pid_score_v")
    trk_score_v = up.array("trk_score_v")
    trk_len_v   = up.array('trk_len_v')
    trk_end_x_v = up.array('trk_sce_end_x_v')
    trk_end_y_v = up.array('trk_sce_end_y_v')
    trk_end_z_v = up.array('trk_sce_end_z_v')
    trk_start_x_v = up.array('trk_sce_start_x_v')
    trk_start_y_v = up.array('trk_sce_start_y_v')
    trk_start_z_v = up.array('trk_sce_start_z_v')
    trk_energy_proton_v = up.array('trk_energy_proton_v') # range-based proton kinetic energy
    trk_range_muon_mom_v   = up.array('trk_range_muon_mom_v')  # range-based muon momentum
    trk_mcs_muon_mom_v     = up.array('trk_mcs_muon_mom_v')
    trk_theta_v        = up.array('trk_theta_v')
    trk_phi_v        = up.array('trk_phi_v')
    pfp_generation_v = up.array('pfp_generation_v')
    trk_distance_v  = up.array('trk_distance_v')
    trk_calo_energy_y_v = up.array('trk_calo_energy_y_v')
    trk_pfp_id_v = up.array('trk_pfp_id_v')
    pfp_pdg_v = up.array('backtracked_pdg')
    
    #trk_dir_x_v = up.array('trk_dir_x_v')
    #trk_dir_y_v = up.array('trk_dir_y_v')
    #trk_dir_z_v = up.array('trk_dir_z_v')
    
    trk_mask = (trk_score_v>0.0)
    proton_mask = ((trk_score_v > 0.5) & (trk_llr_pid_v < 0.))

    df["proton_range_energy"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else 0. for vec,vid in zip(trk_energy_proton_v[proton_mask],trk_len_v[proton_mask])])

    '''
    df["trk1_score"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_score_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_x_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_z_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_x_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_z_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_len"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_len_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_pid"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_llr_pid_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_range_proton"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_energy_proton_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_calo"]    = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_calo_energy_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_range_muon"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_range_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_mcs_muon"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_mcs_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_theta"] = np.cos(awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_theta_v[trk_mask],trk_len_v[trk_mask])]))
    df["trk1_phi"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_phi_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_pfp_idx"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_pfp_id_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_pfp_idx"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_pfp_id_v[trk_mask],trk_len_v[trk_mask])])
    trk1_pfp_idx = df["trk1_pfp_idx"].values
    #print ('trk1_pfp_idx : ', trk1_pfp_idx)
    #print ('pfp_pdg_v : ', pfp_pdg_v)
    df["trk1_backtracked_pdg"] = awkward.fromiter([vec[int(vid)-1] if ( (len(vec) > 0) and (vid >= 0) and (vid < 10000) and (vid != -9999.) ) else -9999. for vec,vid in zip(pfp_pdg_v,trk1_pfp_idx)])
    #df["trk1_backtracked_pdg"] = awkward.fromiter([vec1[vec2[vid.argsort()[-1]]] if ( (len(vid)>0) and (len(vec1) >= len(vec2)) ) else -9999. for vec1,vec2,vid in zip(pfp_pdg_v,trk_pfp_id_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_backtracked_pdg"] = get_elm_from_vec_idx(pfp_pdg_v,trk1_pfp_idx)
    '''

    '''
    # 2nd longest track
    df["trk2_len"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_len_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_pid"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_llr_pid_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_range_proton"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_energy_proton_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_calo"]    = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_calo_energy_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_range_muon"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_range_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_mcs_muon"]   = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_mcs_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_theta"] = np.cos(awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_theta_v[trk_mask],trk_len_v[trk_mask])]))
    df["trk2_phi"]   = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_phi_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_pfp_idx"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_pfp_id_v[trk_mask],trk_len_v[trk_mask])])
    trk2_pfp_idx = df["trk2_pfp_idx"].values
    #print ('trk1_pfp_idx : ', trk1_pfp_idx)                                                                                                                   
    #print ('pfp_pdg_v : ', pfp_pdg_v)                                                                                                                     
    df["trk2_backtracked_pdg"] = awkward.fromiter([vec[int(vid)-1] if ( (len(vec) > 0) and (vid >= 0) and (vid < 10000) and (vid != -9999.) ) else -9999. for vec,vid in zip(pfp_pdg_v,trk2_pfp_idx)])
    '''
    
    # get element-wise reconstructed neutrino energy (for each index the value will be the neutrino energy assuming the track at that index is the muon)
    df['trk_energy_tot'] = trk_energy_proton_v.sum()
    muon_energy_correction_v = np.sqrt(trk_range_muon_mom_v**2 + 0.105**2) - trk_energy_proton_v
    # get element-wise MCS consistency
    muon_mcs_consistency_v    = ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v )
    muon_calo_consistency_v   = ( (trk_calo_energy_y_v - trk_range_muon_mom_v) / trk_range_muon_mom_v )
    proton_calo_consistency_v = ( (trk_calo_energy_y_v * 0.001 - trk_energy_proton_v) / trk_energy_proton_v )

    shr_mask = (trk_score_v<0.5)
    trk_mask = (trk_score_v>0.5)
    
    muon_candidate_idx = get_idx_from_vec_sort(-1,trk_len_v,trk_mask)

    '''
    df["muon_candidate_length"]  = get_elm_from_vec_idx(trk_len_v,muon_candidate_idx)
    df["muon_candidate_score"]   = get_elm_from_vec_idx(trk_score_v,muon_candidate_idx)
    df["muon_candidate_pid"]     = get_elm_from_vec_idx(trk_llr_pid_v,muon_candidate_idx)
    df["muon_candidate_distance"]= get_elm_from_vec_idx(trk_distance_v,muon_candidate_idx)
    df["muon_candidate_gen"]     = get_elm_from_vec_idx(pfp_generation_v,muon_candidate_idx)
    df["muon_candidate_mcs"]     = get_elm_from_vec_idx(muon_mcs_consistency_v,muon_candidate_idx)
    df["muon_candidate_start_x"] = get_elm_from_vec_idx(trk_start_x_v,muon_candidate_idx)
    df["muon_candidate_start_y"] = get_elm_from_vec_idx(trk_start_y_v,muon_candidate_idx)
    df["muon_candidate_start_z"] = get_elm_from_vec_idx(trk_start_z_v,muon_candidate_idx)
    df["muon_candidate_end_x"]   = get_elm_from_vec_idx(trk_end_x_v,muon_candidate_idx)
    df["muon_candidate_end_y"]   = get_elm_from_vec_idx(trk_end_y_v,muon_candidate_idx)
    df["muon_candidate_end_z"]   = get_elm_from_vec_idx(trk_end_z_v,muon_candidate_idx)
    '''

    '''
    df["trk1_muon_mcs_consistency"]    = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(muon_mcs_consistency_v[trk_mask] ,trk_len_v[trk_mask])])
    df["trk1_muon_calo_consistency"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(muon_calo_consistency_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_proton_calo_consistency"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(proton_calo_consistency_v[trk_mask],trk_len_v[trk_mask])])
    '''

    # apply numu selection as defined by Ryan
    trk_score_v = up.array("trk_score_v")
    #'''
    muon_mask = (trk_score_v>0.8) & (trk_llr_pid_v > 0.2) \
                & (trk_start_x_v > 5.) & (trk_start_x_v < 251.) & (trk_end_x_v > 5.) & (trk_end_x_v < 251.) \
                & (trk_start_y_v > -110.) & (trk_start_y_v < 110.) & (trk_end_y_v > -110.) & (trk_end_y_v < 110.) \
                & (trk_start_z_v > 20.) & (trk_start_z_v < 986.) & (trk_end_z_v > 20.) & (trk_end_z_v < 986.) \
                & (trk_len_v > 10) & (trk_distance_v < 4.) & (pfp_generation_v == 2) \
                & ( ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v ) > -0.5 ) \
                & ( ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v ) < 0.5 )
    '''
    muon_mask = (trk_score_v>0.8) & (trk_llr_pid_v > -1.0) \
                & (trk_start_x_v > 5.) & (trk_start_x_v < 251.) & (trk_end_x_v > 5.) & (trk_end_x_v < 251.) \
                & (trk_start_y_v > -110.) & (trk_start_y_v < 110.) & (trk_end_y_v > -110.) & (trk_end_y_v < 110.) \
                & (trk_start_z_v > 20.) & (trk_start_z_v < 986.) & (trk_end_z_v > 20.) & (trk_end_z_v < 986.) \
                & (trk_len_v > 10) & (trk_distance_v < 4.) & (pfp_generation_v == 2)
    '''

    '''
    contained_track_mask = (trk_start_x_v > 5.) & (trk_start_x_v < 251.) & (trk_end_x_v > 5.) & (trk_end_x_v < 251.) \
                           & (trk_start_y_v > -110.) & (trk_start_y_v < 110.) & (trk_end_y_v > -110.) & (trk_end_y_v < 110.) \
                           & (trk_start_z_v > 20.) & (trk_start_z_v < 986.) & (trk_end_z_v > 20.) & (trk_end_z_v < 986.) \
                           & (trk_score_v>0.5)
    '''

    #p_v = up.array("pfnhits")
    muon_idx = get_idx_from_vec_sort(-1,trk_len_v,muon_mask)

    df["muon_length"] = get_elm_from_vec_idx(trk_len_v,muon_idx)
    df["muon_momentum"] = get_elm_from_vec_idx(trk_range_muon_mom_v,muon_idx)
    df['muon_phi']    = get_elm_from_vec_idx(trk_phi_v,muon_idx)
    df['muon_theta']  = get_elm_from_vec_idx(np.cos(trk_theta_v),muon_idx)
    df['muon_proton_energy'] = get_elm_from_vec_idx(np.cos(trk_energy_proton_v),muon_idx) 
    df['muon_energy'] = np.sqrt( df['muon_momentum']**2 + 0.105**2 )
    #df['neutrino_energy'] = df['trk_energy_tot'] + df['muon_energy'] - df['muon_proton_energy']
    df['neutrino_energy'] = df['trk_energy_tot'] + get_elm_from_vec_idx(muon_energy_correction_v,muon_idx)
    df['muon_mcs_consistency'] = get_elm_from_vec_idx(muon_mcs_consistency_v,muon_idx)
    
    trk_score_v = up.array("trk_score_v")
    df['n_muons_tot'] = muon_mask.sum()
    df['n_tracks_tot'] = trk_mask.sum()
    #df['n_tracks_contained'] = contained_track_mask.sum()
    df['n_protons_tot'] = proton_mask.sum()
    df['n_showers_tot'] = shr_mask.sum()    
    
    #df["trk1_dir_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_x_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_dir_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_y_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_dir_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_z_v[trk_mask],trk_len_v[trk_mask])])
    return

def process_uproot_eta(up,df):
    #

    trk_score_v = up.array("trk_score_v")
    trk_calo_energy_y_v = up.array("trk_calo_energy_y_v")
    trk_distance_v  = up.array('trk_distance_v')
    trk_sce_end_x_v = up.array('trk_sce_end_x_v')
    trk_sce_end_y_v = up.array('trk_sce_end_y_v')
    trk_sce_end_z_v = up.array('trk_sce_end_z_v')
    trk_llr_pid_v = up.array("trk_llr_pid_score_v")
    trk_len_v = up.array('trk_len_v')

    protonidcut = 0.5 #fimxe (original was 0)                                                              
    trk_dir_x_v = up.array('trk_dir_x_v')
    trk_dir_y_v = up.array('trk_dir_y_v')
    trk_dir_z_v = up.array('trk_dir_z_v')
    trk_energy_proton_v = up.array('trk_energy_proton_v')
    trk_score_v = up.array('trk_score_v')
    pi0_energy1_Y = 0.001*up.array('pi0_energy1_Y')/0.83
    pi0_dir1_x = up.array('pi0_dir1_x')
    pi0_dir1_y = up.array('pi0_dir1_y')
    pi0_dir1_z = up.array('pi0_dir1_z')
    pi0_energy2_Y = 0.001*up.array('pi0_energy2_Y')/0.83
    pi0_dir2_x = up.array('pi0_dir2_x')
    pi0_dir2_y = up.array('pi0_dir2_y')
    pi0_dir2_z = up.array('pi0_dir2_z')
    proton_mask = (trk_llr_pid_v<protonidcut)&(trk_score_v>0.5)
    leadProtonIdx = get_idx_from_vec_sort(-1,trk_energy_proton_v,proton_mask)
    leadP_KE = get_elm_from_vec_idx(trk_energy_proton_v,leadProtonIdx,0.)
    leadP_dirx = get_elm_from_vec_idx(trk_dir_x_v,leadProtonIdx,0.)
    leadP_diry = get_elm_from_vec_idx(trk_dir_y_v,leadProtonIdx,0.)
    leadP_dirz = get_elm_from_vec_idx(trk_dir_z_v,leadProtonIdx,0.)
    leadP_E = leadP_KE + 0.938
    leadP_P = np.sqrt(leadP_E*leadP_E - 0.938*0.938)
    reco_E_hadsum = leadP_E + pi0_energy1_Y + pi0_energy2_Y
    reco_px_hadsum = leadP_P*leadP_dirx + pi0_energy1_Y*pi0_dir1_x + pi0_energy2_Y*pi0_dir2_x
    reco_py_hadsum = leadP_P*leadP_diry + pi0_energy1_Y*pi0_dir1_y + pi0_energy2_Y*pi0_dir2_y
    reco_pz_hadsum = leadP_P*leadP_dirz + pi0_energy1_Y*pi0_dir1_z + pi0_energy2_Y*pi0_dir2_z
    reco_M_had = np.sqrt(reco_E_hadsum*reco_E_hadsum - reco_px_hadsum*reco_px_hadsum - reco_py_hadsum*reco_py_hadsum - reco_pz_hadsum*reco_pz_hadsum)
    df['reco_W'] = reco_M_had

    shr_mask = (trk_score_v<0.5)
    trk_mask = (trk_score_v > 0.5)
    proton_mask = ( (trk_mask) & (trk_llr_pid_v < 0.) )
    mip_mask = ( (trk_mask) & (trk_llr_pid_v >= 0.) )
    gap_mask = (trk_distance_v>2)
    df['n_pfp_tot'] = trk_score_v.sum()
    df['n_showers_tot'] = shr_mask.sum()
    df['n_tracks_tot'] = trk_mask.sum()
    df['n_protons_tot'] = proton_mask.sum()
    df['n_mip_tot'] = mip_mask.sum()
    shr_mask_025 = ( (trk_score_v<0.5) & (trk_calo_energy_y_v > 25.) )
    shr_mask_050 = ( (trk_score_v<0.5) & (trk_calo_energy_y_v > 50.) )
    shr_mask_100 = ( (trk_score_v<0.5) & (trk_calo_energy_y_v > 100.) )
    df['n_showers_025_tot'] = shr_mask_025.sum()
    df['n_showers_050_tot'] = shr_mask_050.sum()
    df['n_showers_100_tot'] = shr_mask_100.sum()
    df['n_gap_tot'] = gap_mask.sum()

    trk_dir_z_v = up.array("trk_dir_z_v")
    df["pfp_1_dir_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_z_v,trk_calo_energy_y_v)])


def process_uproot_ccncpi0vars(up,df):
    mc_pdg = up.array('mc_pdg')
    # compute hadronic mass
    #hadron_mask = (mc_pdg!=11)&(mc_pdg!=-11)&(mc_pdg!=13)&(mc_pdg!=-13)&(mc_pdg!=15)&(mc_pdg!=-15)&\
    #              (mc_pdg!=12)&(mc_pdg!=-12)&(mc_pdg!=14)&(mc_pdg!=-14)&(mc_pdg!=16)&(mc_pdg!=-16)&\
    #              (mc_pdg!=22)
    # compute hadronic mass, consider the leading nucleon only
    #nucleon_mask = (mc_pdg==2112)|(mc_pdg==2212)
    #other_mask = (mc_pdg!=11)&(mc_pdg!=-11)&(mc_pdg!=13)&(mc_pdg!=-13)&(mc_pdg!=15)&(mc_pdg!=-15)&\
    #             (mc_pdg!=12)&(mc_pdg!=-12)&(mc_pdg!=14)&(mc_pdg!=-14)&(mc_pdg!=16)&(mc_pdg!=-16)&\
    #             (mc_pdg!=22)&(mc_pdg!=2112)&(mc_pdg!=2212)
    mc_E = up.array('mc_E')
    mc_px = up.array('mc_px')
    mc_py = up.array('mc_py')
    mc_pz = up.array('mc_pz')
    #mc_E_hadsum = sum_elements_from_mask(mc_E,hadron_mask)
    #mc_px_hadsum = sum_elements_from_mask(mc_px,hadron_mask)
    #mc_py_hadsum = sum_elements_from_mask(mc_py,hadron_mask)
    #mc_pz_hadsum = sum_elements_from_mask(mc_pz,hadron_mask)
    #mc_E_other = sum_elements_from_mask(mc_E,other_mask)
    #mc_px_other = sum_elements_from_mask(mc_px,other_mask)
    #mc_py_other = sum_elements_from_mask(mc_py,other_mask)
    #mc_pz_other = sum_elements_from_mask(mc_pz,other_mask)
    #mostEnuclIdx = get_idx_from_vec_sort(-1,mc_E,nucleon_mask)
    #mc_E_nucl = get_elm_from_vec_idx(mc_E,mostEnuclIdx,0.)
    #mc_px_nucl = get_elm_from_vec_idx(mc_px,mostEnuclIdx,0.)
    #mc_py_nucl = get_elm_from_vec_idx(mc_py,mostEnuclIdx,0.)
    #mc_pz_nucl = get_elm_from_vec_idx(mc_pz,mostEnuclIdx,0.)
    #mc_E_hadsum = mc_E_other+mc_E_nucl
    #mc_px_hadsum = mc_px_other+mc_px_nucl
    #mc_py_hadsum = mc_py_other+mc_py_nucl
    #mc_pz_hadsum = mc_pz_other+mc_pz_nucl
    #mc_M_had = np.sqrt(mc_E_hadsum*mc_E_hadsum - mc_px_hadsum*mc_px_hadsum -\
    #                   mc_py_hadsum*mc_py_hadsum - mc_pz_hadsum*mc_pz_hadsum)
    proton_mask = (mc_pdg==2212)
    pi0_mask = (mc_pdg==111)
    pipm_mask = (mc_pdg==211)|(mc_pdg==-211)
    pi_mask = (pi0_mask|pipm_mask)
    mostEprotIdx = get_idx_from_vec_sort(-1,mc_E,proton_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E,mostEprotIdx,0.)
    mc_px_prot = get_elm_from_vec_idx(mc_px,mostEprotIdx,0.)
    mc_py_prot = get_elm_from_vec_idx(mc_py,mostEprotIdx,0.)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz,mostEprotIdx,0.)
    mc_E_pi = sum_elements_from_mask(mc_E,pi_mask)
    mc_px_pi = sum_elements_from_mask(mc_px,pi_mask)
    mc_py_pi = sum_elements_from_mask(mc_py,pi_mask)
    mc_pz_pi = sum_elements_from_mask(mc_pz,pi_mask)
    mc_E_hadsum = mc_E_prot+mc_E_pi
    mc_px_hadsum = mc_px_prot+mc_px_pi
    mc_py_hadsum = mc_py_prot+mc_py_pi
    mc_pz_hadsum = mc_pz_prot+mc_pz_pi
    mc_M_had = np.sqrt(mc_E_hadsum*mc_E_hadsum - mc_px_hadsum*mc_px_hadsum -\
                       mc_py_hadsum*mc_py_hadsum - mc_pz_hadsum*mc_pz_hadsum)
    df['mc_W'] = mc_M_had
    #
    # compute hadronic mass, consider the leading proton and leading pi0 only
    proton_mask = (mc_pdg==2212)
    pi0_mask = (mc_pdg==111)
    mostEprotIdx = get_idx_from_vec_sort(-1,mc_E,proton_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E,mostEprotIdx,0.)
    mc_px_prot = get_elm_from_vec_idx(mc_px,mostEprotIdx,0.)
    mc_py_prot = get_elm_from_vec_idx(mc_py,mostEprotIdx,0.)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz,mostEprotIdx,0.)
    mostEpi0Idx = get_idx_from_vec_sort(-1,mc_E,pi0_mask)
    mc_E_pi0 = get_elm_from_vec_idx(mc_E,mostEpi0Idx,0.)
    mc_px_pi0 = get_elm_from_vec_idx(mc_px,mostEpi0Idx,0.)
    mc_py_pi0 = get_elm_from_vec_idx(mc_py,mostEpi0Idx,0.)
    mc_pz_pi0 = get_elm_from_vec_idx(mc_pz,mostEpi0Idx,0.)
    mc_E_ppi0 = mc_E_pi0+mc_E_prot
    mc_px_ppi0 = mc_px_pi0+mc_px_prot
    mc_py_ppi0 = mc_py_pi0+mc_py_prot
    mc_pz_ppi0 = mc_pz_pi0+mc_pz_prot
    mc_M_ppi0 = np.sqrt(mc_E_ppi0*mc_E_ppi0 - mc_px_ppi0*mc_px_ppi0 - mc_py_ppi0*mc_py_ppi0 - mc_pz_ppi0*mc_pz_ppi0)
    df['mc_W_ppi0'] = mc_M_ppi0
    # compute momentum transfer
    nu_e = up.array('nu_e')
    lepton_mask = (mc_pdg==11)|(mc_pdg==-11)|(mc_pdg==13)|(mc_pdg==-13)|(mc_pdg==15)|(mc_pdg==-15)|\
                  (mc_pdg==12)|(mc_pdg==-12)|(mc_pdg==14)|(mc_pdg==-14)|(mc_pdg==16)|(mc_pdg==-16)
    mc_E_lep = sum_elements_from_mask(mc_E,lepton_mask)
    mc_px_lep = sum_elements_from_mask(mc_px,lepton_mask)
    mc_py_lep = sum_elements_from_mask(mc_py,lepton_mask)
    mc_pz_lep = sum_elements_from_mask(mc_pz,lepton_mask)
    mc_q_E = nu_e - mc_E_lep
    mc_q_px = -1 * mc_px_lep
    mc_q_py = -1 * mc_py_lep
    mc_q_pz = nu_e - mc_pz_lep
    mc_Q2 = -1*(mc_q_E*mc_q_E - mc_q_px*mc_q_px - mc_q_py*mc_q_py - mc_q_pz*mc_q_pz)
    df['mc_Q2'] = mc_Q2
    #
    #
    protonidcut = 0.5 #fimxe (original was 0)
    trk_dir_x_v = up.array('trk_dir_x_v')
    trk_dir_y_v = up.array('trk_dir_y_v')
    trk_dir_z_v = up.array('trk_dir_z_v')
    trk_energy_proton_v = up.array('trk_energy_proton_v')
    trk_llr_pid_score_v = up.array('trk_llr_pid_score_v')
    trk_score_v = up.array('trk_score_v')
    pi0_energy1_Y = 0.001*up.array('pi0_energy1_Y')/0.83
    pi0_dir1_x = up.array('pi0_dir1_x')
    pi0_dir1_y = up.array('pi0_dir1_y')
    pi0_dir1_z = up.array('pi0_dir1_z')
    pi0_energy2_Y = 0.001*up.array('pi0_energy2_Y')/0.83
    pi0_dir2_x = up.array('pi0_dir2_x')
    pi0_dir2_y = up.array('pi0_dir2_y')
    pi0_dir2_z = up.array('pi0_dir2_z')
    proton_mask = (trk_llr_pid_score_v<protonidcut)&(trk_score_v>0.5)
    leadProtonIdx = get_idx_from_vec_sort(-1,trk_energy_proton_v,proton_mask)
    leadP_KE = get_elm_from_vec_idx(trk_energy_proton_v,leadProtonIdx,0.)
    leadP_dirx = get_elm_from_vec_idx(trk_dir_x_v,leadProtonIdx,0.)
    leadP_diry = get_elm_from_vec_idx(trk_dir_y_v,leadProtonIdx,0.)
    leadP_dirz = get_elm_from_vec_idx(trk_dir_z_v,leadProtonIdx,0.)
    leadP_E = leadP_KE + 0.938
    leadP_P = np.sqrt(leadP_E*leadP_E - 0.938*0.938)
    reco_E_hadsum = leadP_E + pi0_energy1_Y + pi0_energy2_Y
    reco_px_hadsum = leadP_P*leadP_dirx + pi0_energy1_Y*pi0_dir1_x + pi0_energy2_Y*pi0_dir2_x
    reco_py_hadsum = leadP_P*leadP_diry + pi0_energy1_Y*pi0_dir1_y + pi0_energy2_Y*pi0_dir2_y
    reco_pz_hadsum = leadP_P*leadP_dirz + pi0_energy1_Y*pi0_dir1_z + pi0_energy2_Y*pi0_dir2_z
    reco_M_had = np.sqrt(reco_E_hadsum*reco_E_hadsum - reco_px_hadsum*reco_px_hadsum - reco_py_hadsum*reco_py_hadsum - reco_pz_hadsum*reco_pz_hadsum)
    df['reco_W'] = reco_M_had
    # multiplicities
    mip_mask = (trk_llr_pid_score_v>=protonidcut)&(trk_score_v>0.5)
    df['n_reco_protons'] = proton_mask.sum()
    df['n_reco_mip'] = mip_mask.sum()
    #
    # multiple pi0 combinatorics
    #
    shr_energy_y_v = up.array('shr_energy_y_v')
    shr_dist_v = up.array('shr_dist_v')
    shr_px_v = up.array('shr_px_v')
    shr_py_v = up.array('shr_py_v')
    shr_pz_v = up.array('shr_pz_v')
    trk_score_v = up.array('trk_score_v')
    shr_mask = (trk_score_v<0.5)
    shr_mask_args = [ np.argwhere(mask).flatten().tolist() for mask in shr_mask ]
    gg_combs = awkward.fromiter([ combs(args) for args in shr_mask_args ])
    mggs = awkward.fromiter([ all_comb_mgg(ev,pxv,pyv,pzv,combs) for ev,pxv,pyv,pzv,combs in zip(shr_energy_y_v,shr_px_v,shr_py_v,shr_pz_v,gg_combs) ])
    mdiffs = awkward.fromiter([ [np.abs(m-134.98) for m in ms] for ms in mggs])
    gg_combs_argsort = awkward.fromiter([ np.argsort(d) for d in mdiffs ])
    gg_unique_combs = awkward.fromiter([ unique_combs(c,a) for c,a in zip(gg_combs, gg_combs_argsort) ])
    npi0s_delta20 = (mdiffs[gg_unique_combs]<20).sum()
    npi0s_delta30 = (mdiffs[gg_unique_combs]<30).sum()
    npi0s_delta40 = (mdiffs[gg_unique_combs]<40).sum()
    npi0s_delta50 = (mdiffs[gg_unique_combs]<50).sum()
    df['npi0s_delta20'] = npi0s_delta20
    df['npi0s_delta30'] = npi0s_delta30
    df['npi0s_delta40'] = npi0s_delta40
    df['npi0s_delta50'] = npi0s_delta50
    #
    shr_mask_025 = ( (trk_score_v<0.5) & (shr_energy_y_v > 25.) )
    shr_mask_050 = ( (trk_score_v<0.5) & (shr_energy_y_v > 50.) )
    shr_mask_100 = ( (trk_score_v<0.5) & (shr_energy_y_v > 100.) )
    df['n_showers_025_tot2'] = shr_mask_025.sum()
    df['n_showers_050_tot2'] = shr_mask_050.sum()
    df['n_showers_100_tot2'] = shr_mask_100.sum()
    #
    #leading pi0 mc truth kinematics
    leadPi0Idx = get_idx_from_vec_sort(-1,mc_E,pi0_mask)
    leadPi0_E = get_elm_from_vec_idx(mc_E,leadPi0Idx,0.)
    leadPi0_px = get_elm_from_vec_idx(mc_px,leadPi0Idx,0.)
    leadPi0_py = get_elm_from_vec_idx(mc_py,leadPi0Idx,0.)
    leadPi0_pz = get_elm_from_vec_idx(mc_pz,leadPi0Idx,0.)
    df['leadPi0_E'] = leadPi0_E
    df['leadPi0_px'] = leadPi0_px
    df['leadPi0_py'] = leadPi0_py
    df['leadPi0_pz'] = leadPi0_pz
    df['leadPi0_uz'] = leadPi0_pz/np.sqrt(leadPi0_px*leadPi0_px + leadPi0_py*leadPi0_py + leadPi0_pz*leadPi0_pz)
    return
    
def get_variables():

    print('Getting Variables!')
    
    VARDICT = {}
    
    VARIABLES = [
        "nu_pdg", "slpdg", "backtracked_pdg", #"trk_score_v", 
        "category", "ccnc",
        "endmuonmichel",
        #"NeutrinoEnergy0","NeutrinoEnergy1","NeutrinoEnergy2",
        "run","sub","evt",
        "CosmicIP","CosmicDirAll3D","CosmicIPAll3D",
        #"nu_flashmatch_score","best_cosmic_flashmatch_score","best_obviouscosmic_flashmatch_score",
        "flash_pe",
        "trk_llr_pid_score_v", # trk-PID score
        "_opfilter_pe_beam", "_opfilter_pe_veto", # did the event pass the common optical filter (for MC only)
        "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
        "nproton", "nu_e", 
        #"hits_u", "hits_v", "hits_y", 
        "nneutron", "mc_pdg", "slnunhits", "slnhits", "true_e_visible",
        "npi0","npion","pion_e","muon_e","pi0truth_elec_etot",
        "pi0_e", "evnunhits", "nslice", "interaction",
        "proton_e",
        "slclustfrac", "reco_nu_vtx_x", "reco_nu_vtx_y", "reco_nu_vtx_z",
        "true_nu_vtx_sce_x","true_nu_vtx_sce_y","true_nu_vtx_sce_z",
        "true_nu_vtx_x","true_nu_vtx_y","true_nu_vtx_z",
        #"trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        #"trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
        #"trk_start_x_v","trk_start_z_v","trk_start_z_v",
        "topological_score",
        "isVtxInFiducial",
        "theta", # angle between incoming and outgoing leptons in radians
        #"nu_decay_mode","nu_hadron_pdg","nu_parent_pdg", # flux truth info
        #"shr_energy_tot_cali","selected","n_showers_contained",  # only if CC0piNp variables are saved!
    ]

    VARDICT['VARIABLES'] = VARIABLES
    
    CRTVARS = ["crtveto","crthitpe","_closestNuCosmicDist"]

    VARDICT['CRTVARS'] = CRTVARS
    
    WEIGHTS = ["weightSpline","weightTune","weightSplineTimesTune","nu_decay_mode",
               'knobRPAup','knobRPAdn',
               'knobCCMECup','knobCCMECdn',
               'knobAxFFCCQEup','knobAxFFCCQEdn',
               'knobVecFFCCQEup','knobVecFFCCQEdn',
               'knobDecayAngMECup','knobDecayAngMECdn',
               'knobThetaDelta2Npiup','knobThetaDelta2Npidn']

    VARDICT['WEIGHTS'] = WEIGHTS
    
    #WEIGHTSLEE = ["weightSpline","weightTune","weightSplineTimesTune","nu_decay_mode","leeweight"]
    WEIGHTSLEE = WEIGHTS+["leeweight"]

    VARDICT['WEIGHTSLEE'] = WEIGHTSLEE
    
    SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]

    VARDICT['SYSTVARS'] = SYSTVARS
    
    MCFVARS = ["mcf_nu_e","mcf_lep_e","mcf_actvol","mcf_nmm","mcf_nmp","mcf_nem","mcf_nep","mcf_np0","mcf_npp",
               "mcf_npm","mcf_mcshr_elec_etot","mcf_pass_ccpi0","mcf_pass_ncpi0",
               "mcf_pass_ccnopi","mcf_pass_ncnopi","mcf_pass_cccpi","mcf_pass_nccpi"]

    VARDICT['MCFVARS'] = MCFVARS
    
    NUEVARS = ["shr_dedx_Y", "shr_bkt_pdg", "shr_theta","shr_pfp_id_v",
               "shr_tkfit_dedx_U","shr_tkfit_dedx_V","shr_tkfit_dedx_Y",
               "shr_tkfit_gap10_dedx_U","shr_tkfit_gap10_dedx_V","shr_tkfit_gap10_dedx_Y",
               "shr_tkfit_2cm_dedx_U","shr_tkfit_2cm_dedx_V","shr_tkfit_2cm_dedx_Y",
               "shrmoliereavg","shrmoliererms","shr_energy_tot_cali","n_showers_contained","selected",
               "shr_tkfit_npointsvalid","shr_tkfit_npoints", # fitted vs. all hits for shower
               "shrclusfrac0","shrclusfrac1","shrclusfrac2", # track-fitted hits / all hits
               "trkshrhitdist2", "trkshrhitdist0","trkshrhitdist1", #distance between track and shower in 2D
               "shrsubclusters0","shrsubclusters1","shrsubclusters2", # number of sub-clusters in shower
               "secondshower_U_nhit","secondshower_U_vtxdist","secondshower_U_dot","secondshower_U_dir","shrclusdir0",
               "secondshower_V_nhit","secondshower_V_vtxdist","secondshower_V_dot","secondshower_V_dir","shrclusdir1",
               "secondshower_Y_nhit","secondshower_Y_vtxdist","secondshower_Y_dot","secondshower_Y_dir","shrclusdir2",
               "shrMCSMom","DeltaRMS2h","shrPCA1CMed_5cm","CylFrac2h_1cm",
               "shr_hits_tot", "shr_hits_u_tot", "shr_hits_v_tot", "shr_hits_y_tot",
               "shr_theta_v","shr_phi_v","shr_energy_y_v",
               "shr_start_x_v","shr_start_z_v","shr_start_z_v",
               "trk_bkt_pdg",
               "shr_tkfit_dedx_U", "shr_tkfit_dedx_V", "trk_bkt_pdg",
               "shr_energy", "shr_dedx_U", "shr_dedx_V", "shr_phi", "trk_phi", "trk_theta",
               "shr_distance", "trk_distance",
               "matched_E", "shr_bkt_E", "trk_bkt_E",
               "shr_tkfit_nhits_Y","shr_tkfit_nhits_U","shr_tkfit_nhits_V",
               "shr_tkfit_2cm_nhits_Y","shr_tkfit_2cm_nhits_U","shr_tkfit_2cm_nhits_V",
               "shr_tkfit_gap10_nhits_Y","shr_tkfit_gap10_nhits_U","shr_tkfit_gap10_nhits_V",
               "trk_energy", "tksh_distance", "tksh_angle","contained_fraction",
               "shr_score", "trk_score", "trk_hits_tot","trk_len",
               "trk_hits_tot", "trk_hits_u_tot", "trk_hits_v_tot", "trk_hits_y_tot",
               "shr_dedx_Y_cali", "trk_energy_tot","shr_id",
               "hits_ratio", "n_tracks_contained",
               "shr_px","shr_py","shr_pz","p", "pt", "hits_y",
               "shr_start_x","shr_start_x","shr_start_x",
               "elec_pz","elec_e","truthFiducial",
               'pi0truth_gamma1_edep','shr_bkt_E','pi0truth_gamma1_etot',
               'pi0truth_gamma1_zpos','shr_start_z',
               'pi0truth_gamma1_ypos','shr_start_y',
               'pi0truth_gamma1_xpos','shr_start_x'
    ]

    VARDICT['NUEVARS'] = NUEVARS
 
    
    NUMUVARS = []

    VARDICT['NUMUVARS'] = NUMUVARS
    
    RCVRYVARS = ["shr_energy_tot", "trk_energy_tot",
                 "trk_end_x_v","trk_end_y_v","trk_end_z_v",
                 "trk_phi_v","trk_theta_v","trk_len_v","trk_id",
                 "shr_start_x","shr_start_y","shr_start_z","trk_hits_max",
                 "shr_tkfit_dedx_u_v","shr_tkfit_dedx_v_v","shr_tkfit_dedx_y_v",
                 "shr_tkfit_dedx_nhits_u_v","shr_tkfit_dedx_nhits_v_v","shr_tkfit_dedx_nhits_y_v",
                 "trk2shrhitdist2","trk1trk2hitdist2","shr1shr2moliereavg","shr1trk1moliereavg","shr1trk2moliereavg",
                 "trk2_id","shr2_id","trk_hits_2nd","shr_hits_2nd"
    ]

    VARDICT['RCVRYVARS'] = RCVRYVARS
    
    PI0VARS = ["pi0_radlen1","pi0_radlen2","pi0_dot1","pi0_dot2","pi0_energy1_Y","pi0_energy2_Y",
               "pi0_dedx1_fit_Y","pi0_dedx2_fit_Y","pi0_shrscore1","pi0_shrscore2","pi0_gammadot",
               "pi0_dedx1_fit_V","pi0_dedx2_fit_V","pi0_dedx1_fit_U","pi0_dedx2_fit_U",
               "pi0_mass_Y","pi0_mass_V","pi0_mass_U",
               "pi0_nshower",
               "pi0_dir2_x","pi0_dir2_y","pi0_dir2_z","pi0_dir1_x","pi0_dir1_y","pi0_dir1_z",
               "pi0truth_gamma1_etot","pi0truth_gamma2_etot","pi0truth_gammadot","pi0truth_gamma_parent",
               "n_showers_contained"
    ]

    VARDICT['PI0VARS'] = PI0VARS
    
    R3VARS = []

    VARDICT['R3VARS'] = R3VARS
    
    return VARDICT
    
    
def load_data_run123(which_sideband='pi0', return_plotter=True, 
                     pi0scaling=0,
                     USEBDT=True,
                     loadpi0variables=False,
                     loadtruthfilters=True,
                     loadpi0filters=False,
                     loadfakedata=0,
                     loadshowervariables=True,
                     loadnumuntuples=False,
                     loadnumuvariables=False,
                     loadnumucrtonly=False,
                     loadeta=False,
                     loadsystematics=True,
                     loadrecoveryvars=False,
                     loadccncpi0vars=False,
                     updatedProtThresh=-1,
                     runs_to_load = [1,2,3]):

    fold = ls.fold
    tree = "NeutrinoSelectionFilter"

    # Load the variables dictionary 
    VARDICT = get_variables()

    #runs_to_load = [1,2,3]
    print("loading data and mc from runs",runs_to_load)
 
    ############################# Sample Lists #############################

    ################################ Run 1 #################################

    # sample list
    if which_sideband=='opendata': R1BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2_5e19'
    else: R1BNB = 'run1_nuepresel' # unblinded eLEE
    R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C_all_reco2'
    #R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_C2_D1_D2_E1_E2_all_reco2' #Run1 + Run2
    R1NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2' # OFFICIA
    #R1NU  = 'prodgenie_bnb_nu_overlay_extra' # DETVAR
    R1NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1NCPI0  = 'prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2_extra'#v48
    #R1NCPI0  = 'prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2'#v43
    R1CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2'
    R1CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run1_reco2_reco2'
    R1CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'

    # dummy samples to load data more quickly / using less memory when we only care about Run3 numus
    if (loadnumucrtonly):
        R1BNB = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1EXT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1NU  = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        
    ################################ Run 2 #################################

    R2NU = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run2_reco2_D1D2_reco2"
    R2NUE = "prodgenie_bnb_intrinsic_nue_overlay_run2_v08_00_00_35_run2a_reco2_reco2"
    R2EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_D_E_all_reco2'
    R2DRT = 'prodgenie_dirt_overlay_v08_00_00_35_all_run2_reco2_reco2'

    # dummy samples to load data more quickly / using less memory when we only care about Run3 numus
    if (loadnumucrtonly):
        R2NU  = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R2NUE = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R2DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R2EXT  = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    
    ################################ Run 3 #################################

    if which_sideband=='opendata': R3BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2_1e19'
    else: R3BNB = 'run3_nuepresel' # unblinded eLEE
    R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_F_G_all_reco2'
    if (loadnumucrtonly):
        R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_G_all_reco2'
    R3NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run3_reco2_G_reco2'#_newtune' # OFFICIAL
    #R3NU  = 'prodgenie_bnb_nu_overlay_extra' # DETVAR
    R3NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'
    R3DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'#_newtune'
    R3NCPI0  = 'prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2'
    R3CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2'
    R3CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run3_reco2_reco2'
    R3CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run3_reco2_reco2'
    R3NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_new_run3_reco2_reco2'
    R3NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_New_run3_reco2_reco2'
    R3ETA    = 'eta'

    ################################ Run 4 #################################

    # Run 4 simulation/data
    R4NU = 'run4b_bnb_nu_overlay_pandora_reco2_run4b_pandora_reco2_reco2'
    R4NUE = 'run4c_bnb_intrinsic_nue_overlay_pandora_reco2_v08_00_00_67_PARTIAL_reco2_ana' # TODO: Change this to the intrinsic nue when available
    R4DRT = 'prod_extunbiased_bnb_dirt_overlay_run4b_v08_00_00_63_run4b_reco2'
    R4EXT = 'bnb_run4b_ext_reco2_v08_00_00_63_run4b_reco2_all'
    R4BNB = 'bnb_on_run4b_reco2_v08_00_00_63_run4b_reco2_beam_good'

    ############################# Load Uproots #############################

    print("Loading uproot files")

    if 3 in runs_to_load:
        print("Loading run 3 uproots!!")
        ur3mc = uproot.open(ls.ntuple_path+ls.RUN3+R3NU+ls.APPEND+".root")[fold][tree]
        ur3nue = uproot.open(ls.ntuple_path+ls.RUN3+R3NUE+ls.APPEND+".root")[fold][tree]
        if (loadfakedata == 0):
            ur3data = uproot.open(ls.ntuple_path+ls.RUN3+R3BNB+ls.APPEND+".root")[fold][tree]
        elif (loadfakedata == 1):
            print ('loading dataset fakedata1 run 3...')
            print ('path is : %s'%(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set1_run3b_reco2_v08_00_00_41_reco2.root'))
            ur3data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set1_run3b_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 2):
            ur3data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set2_run3b_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 3):
            ur3data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set3_run3b_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 4):
            ur3data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set4_run3b_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 5):
            ur3data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set5_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 9):
            ur3data = uproot.open(ls.ntuple_path+'fakedata/high_stat_prodgenie_bnb_nu_overlay_DetVar_Run3_NuWro_reco2_reco2.root')['nuselection'][tree]
        ur3ext = uproot.open(ls.ntuple_path+ls.RUN3+R3EXT+ls.APPEND+".root")[fold][tree]
        ur3dirt = uproot.open(ls.ntuple_path+ls.RUN3+R3DRT+ls.APPEND+".root")[fold][tree]
        ur3lee = uproot.open(ls.ntuple_path+ls.RUN3+R3NUE+ls.APPEND+".root")[fold][tree]
        if (loadpi0filters):
            ur3ncpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3NCPI0+ls.APPEND+".root")[fold][tree]
            ur3ccpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3CCPI0+ls.APPEND+".root")[fold][tree]
            ur3eta   = uproot.open('/Users/davidc-local/data/searchingfornues/v08_00_00_48/run3/'+R3ETA+".root")[fold][tree]
        if (loadtruthfilters):
            ur3ccnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCNOPI+ls.APPEND+".root")[fold][tree]
            ur3cccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCCPI+ls.APPEND+".root")[fold][tree]
            ur3ncnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCNOPI+ls.APPEND+".root")[fold][tree]
            ur3nccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCCPI+ls.APPEND+".root")[fold][tree]
            ur3ncpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3NCPI0+ls.APPEND+".root")[fold][tree]
            ur3ccpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3CCPI0+ls.APPEND+".root")[fold][tree]
        print("Done loading run 3 uproots")

    if 2 in runs_to_load:
        print("Loading run 2 uproots!!")
        ur2mc = uproot.open(ls.ntuple_path+ls.RUN2+R2NU+ls.APPEND+".root")[fold][tree]
        ur2nue = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
        ur2drt = uproot.open(ls.ntuple_path+ls.RUN2+R2DRT+ls.APPEND+".root")[fold][tree]
        ur2lee = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
        ur2ext = uproot.open(ls.ntuple_path+ls.RUN2+R2EXT+ls.APPEND+".root")[fold][tree]
        if (loadfakedata == 9):
            ur2data = uproot.open(ls.ntuple_path+'fakedata/high_stat_prodgenie_bnb_nu_overlay_DetVar_Run2_NuWro_reco2_reco2.root')['nuselection'][tree]
        print("Done loading run 2 uproots")

    if 1 in runs_to_load:
        print("Loading run 1 uproots!!")
        ur1mc = uproot.open(ls.ntuple_path+ls.RUN1+R1NU+ls.APPEND+".root")[fold][tree]
        ur1nue = uproot.open(ls.ntuple_path+ls.RUN1+R1NUE+ls.APPEND+".root")[fold][tree]
        if (loadfakedata == 0):
            ur1data = uproot.open(ls.ntuple_path+ls.RUN1+R1BNB+ls.APPEND+".root")[fold][tree]
        elif (loadfakedata == 1):
            print ('loading dataset fakedata1 run 1...')
            print ('path is : %s'%(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set1_run1_reco2_v08_00_00_41_reco2.root'))
            ur1data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set1_run1_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 2):
            ur1data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set2_run1_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 3):
            ur1data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set3_run1_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 4):
            ur1data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set4_run1_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 5):
            ur1data = uproot.open(ls.ntuple_path+'fakedata/numupresel/prod_uboone_nu2020_fakedata_set5_reco2_v08_00_00_41_reco2.root')['nuselection'][tree]
        elif (loadfakedata == 9):
            ur1data = uproot.open(ls.ntuple_path+'fakedata/high_stat_prodgenie_bnb_nu_overlay_DetVar_Run1_NuWro_reco2_reco2.root')['nuselection'][tree]
        ur1ext = uproot.open(ls.ntuple_path+ls.RUN1+R1EXT+ls.APPEND+".root")[fold][tree]
        ur1dirt = uproot.open(ls.ntuple_path+ls.RUN1+R1DRT+ls.APPEND+".root")[fold][tree]
        ur1lee = uproot.open(ls.ntuple_path+ls.RUN1+R1NUE+ls.APPEND+".root")[fold][tree]
        if (loadpi0filters):
            ur1ncpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1NCPI0+ls.APPEND+".root")[fold][tree]
            ur1ccpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1CCPI0+ls.APPEND+".root")[fold][tree]    
        if (loadtruthfilters):
            ur1ccnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCNOPI+ls.APPEND+".root")[fold][tree]
            ur1cccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCCPI+ls.APPEND+".root")[fold][tree]
            ur1ncnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCNOPI+ls.APPEND+".root")[fold][tree]
            ur1nccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCCPI+ls.APPEND+".root")[fold][tree]
            ur1ncpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1NCPI0+ls.APPEND+".root")[fold][tree]
            ur1ccpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1CCPI0+ls.APPEND+".root")[fold][tree]
        print("Done loading run 1 uproots")
   
    if 4 in runs_to_load:
        print("Loading run 4 uproots!!")
        ur4data = uproot.open(ls.ntuple_path+ls.RUN4+R4BNB+".root")['nuselection'][tree]
        ur4ext = uproot.open(ls.ntuple_path+ls.RUN4+R4EXT+".root")['nuselection'][tree]
        ur4dirt = uproot.open(ls.ntuple_path+ls.RUN4+R4DRT+".root")['nuselection'][tree]
        ur4mc = uproot.open(ls.ntuple_path+ls.RUN4+R4NU+".root")['nuselection'][tree]
        ur4nue = uproot.open(ls.ntuple_path+ls.RUN4+R4NUE+".root")['nuselection'][tree]
        ur4lee = uproot.open(ls.ntuple_path+ls.RUN4+R4NUE+".root")['nuselection'][tree]
        print("Done loading run 4 uproots")

    R123_TWO_SHOWERS_SIDEBAND_BNB = 'neutrinoselection_filt_1e_2showers_sideband_skimmed_extended_v47'
    R123_NP_FAR_SIDEBAND_BNB = 'neutrinoselection_filt_1enp_far_sideband_skimmed_extended_v47'
    R123_0P_FAR_SIDEBAND_BNB = 'neutrinoselection_filt_1e0p_far_sideband_skimmed_v47'
    R1_PI0_SIDEBAND_BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_RUN1_pi0_reco2'
    R123_NP_NEAR_SIDEBAND_BNB = 'neutrinoselection_filt_1eNp_near_sideband_skimmed'
    R123_0P_NEAR_SIDEBAND_BNB = 'neutrinoselection_filt_1e0p_near_sideband_skimmed'
    R2_PI0_SIDEBAND_BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_RUN2_pi0_reco2'
    R3_PI0_SIDEBAND_BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_RUN3_pi0_reco2'

    R1_FULLDATA = 'run1_nuepresel' # '#'run1_neutrinoselection_filt_1e0p_near_sideband_skimmed' #data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2_5e19'
    R2_FULLDATA = 'run2_nuepresel' # 'run2_neutrinoselection_filt_1e0p_near_sideband_skimmed' #run2_data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2_1e19'
    R3_FULLDATA = 'run3_nuepresel' #'run3_neutrinoselection_filt_1e0p_near_sideband_skimmed' #data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2_1e19'

    if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
        #ur1data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        #ur2data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        #ur3data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur1data_np_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run1_'+R123_NP_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur2data_np_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run2_'+R123_NP_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur3data_np_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run3_'+R123_NP_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
        ur1data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur2data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur3data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    if (which_sideband == "0p_far"):
        #ur1data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        #ur2data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        #ur3data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur1data_0p_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run1_'+R123_0P_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur2data_0p_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run2_'+R123_0P_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur3data_0p_far_sidebands = uproot.open(ls.ntuple_path+'nearsidebands/run3_'+R123_0P_NEAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    if (which_sideband == "pi0"):
        ur1data_pi0_sidebands = uproot.open(ls.ntuple_path+'farsidebands/'+R1_PI0_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur2data_pi0_sidebands = uproot.open(ls.ntuple_path+'farsidebands/'+R2_PI0_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur3data_pi0_sidebands = uproot.open(ls.ntuple_path+'farsidebands/'+R3_PI0_SIDEBAND_BNB+".root")['nuselection'][tree]
    if(which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
        #ur1data_fulldata = uproot.open(ls.ntuple_path+R1_FULLDATA+".root")['nuselection'][tree]
        #ur2data_fulldata = uproot.open(ls.ntuple_path+R2_FULLDATA+".root")['nuselection'][tree]
        #ur3data_fulldata = uproot.open(ls.ntuple_path+R3_FULLDATA+".root")['nuselection'][tree]
        if 1 in runs_to_load:
            ur1data_fulldata = uproot.open(ls.ntuple_path+ls.RUN1+R1_FULLDATA+".root")['nuselection'][tree]
        if 2 in runs_to_load:
            ur2data_fulldata = uproot.open(ls.ntuple_path+ls.RUN2+R2_FULLDATA+".root")['nuselection'][tree]
        if 3 in runs_to_load:
            ur3data_fulldata = uproot.open(ls.ntuple_path+ls.RUN3+R3_FULLDATA+".root")['nuselection'][tree]
        
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        R123_NUMU_SIDEBAND_BNB = 'neutrinoselection_filt_numu_ALL'
        if (loadeta == False):
            ur1data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
            ur2data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
            if (loadnumucrtonly):
                ur3data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/'+"data_bnb_peleeFilter_uboone_v08_00_00_41_pot_run3_G1_neutrinoselection_filt.root")['nuselection'][tree]
            else:
                ur3data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
        else:
            ur1data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/nslice/run1_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
            ur2data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/nslice/run2_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
            ur3data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/nslice/run3_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]            

    ############################# Define Variables to Go Into Dataframes #############################

    VARIABLES  = VARDICT['VARIABLES']
    CRTVARS    = VARDICT['CRTVARS']
    WEIGHTS    = VARDICT['WEIGHTS']
    WEIGHTSLEE = VARDICT['WEIGHTSLEE']
    SYSTVARS   = VARDICT['SYSTVARS']
    MCFVARS    = VARDICT['MCFVARS']
    NUEVARS    = VARDICT['NUEVARS']
    NUMUVARS   = VARDICT['NUMUVARS']
    RCVRYVARS  = VARDICT['RCVRYVARS']
    PI0VARS    = VARDICT['PI0VARS']
    R3VARS     = VARDICT['R3VARS']

    if (loadnumucrtonly ==True):
        R3VARS += CRTVARS


    if (loadsystematics == True):
        WEIGHTS += SYSTVARS
        WEIGHTSLEE += SYSTVARS

    if (loadeta == True):
        WEIGHTS += ["neta"]
        WEIGHTSLEE += ["neta"]    

    RCVRYVARS = ["shr_energy_tot", "trk_energy_tot",
                 "trk_end_x_v","trk_end_y_v","trk_end_z_v",
                 "trk_phi_v","trk_theta_v","trk_len_v","trk_id",
                 "shr_start_x","shr_start_y","shr_start_z","trk_hits_max",
                 "shr_tkfit_dedx_u_v","shr_tkfit_dedx_v_v","shr_tkfit_dedx_y_v",
                 "shr_tkfit_dedx_nhits_u_v","shr_tkfit_dedx_nhits_v_v","shr_tkfit_dedx_nhits_y_v",
                 "trk2shrhitdist2","trk1trk2hitdist2","shr1shr2moliereavg","shr1trk1moliereavg","shr1trk2moliereavg",
                 "trk2_id","shr2_id","trk_hits_2nd","shr_hits_2nd"
    ]
    
    PI0VARS = ["pi0_radlen1","pi0_radlen2","pi0_dot1","pi0_dot2","pi0_energy1_Y","pi0_energy2_Y",
               "pi0_dedx1_fit_Y","pi0_dedx2_fit_Y","pi0_shrscore1","pi0_shrscore2","pi0_gammadot",
               "pi0_dedx1_fit_V","pi0_dedx2_fit_V","pi0_dedx1_fit_U","pi0_dedx2_fit_U",
               "pi0_mass_Y","pi0_mass_V","pi0_mass_U",
               "pi0_nshower",
               "pi0_dir2_x","pi0_dir2_y","pi0_dir2_z","pi0_dir1_x","pi0_dir1_y","pi0_dir1_z",
               "pi0truth_gamma1_etot","pi0truth_gamma2_etot","pi0truth_gammadot","pi0truth_gamma_parent",
               "pi0truth_gamma1_dist", "pi0truth_gamma1_edep", "pi0truth_gamma2_dist", "pi0truth_gamma2_edep",
               "true_nu_vtx_x", "true_nu_vtx_y", "true_nu_vtx_z"#,"n_showers_contained"
    ]

    
    if (loadpi0variables == True):
        VARIABLES += PI0VARS
    if (loadshowervariables == True):
        VARIABLES += NUEVARS
    if (loadrecoveryvars == True):
        VARIABLES += RCVRYVARS
    if (loadnumuvariables == True):
        VARIABLES += NUMUVARS

    #make the list unique
    VARIABLES = list(set(VARIABLES))

    #print (VARIABLES)

    ############################# Load the Dataframes #############################

    ################################# Run 4 ######################################

    # TODO: the weights are currently different in run4. Once you've figured out what's going on eith the validations, propagate the fix to here.
    if 4 in runs_to_load:
        print("Loading Run4 dataframes")
        # R3VARS are the CRT variables. Presumably load these for runs 4/5 as well.
        r4mc = ur4mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS + R3VARS, flatten=False) 
        r4nue = ur4nue.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r4data = ur4data.pandas.df(VARIABLES, flatten=False)
        r4ext = ur4ext.pandas.df(VARIABLES + R3VARS, flatten=False)
        r4dirt = ur4dirt.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r4lee = ur4lee.pandas.df(VARIABLES + WEIGHTSLEE + R3VARS, flatten=False)

        r4lee["is_signal"] = r4lee["category"] == 11
        r4data["is_signal"] = r4data["category"] == 11
        r4nue["is_signal"] = r4nue["category"] == 11
        r4mc["is_signal"] = r4mc["category"] == 11
        r4dirt["is_signal"] = r4dirt["category"] == 11
        r4ext["is_signal"] = r4ext["category"] == 11

        df_v = [r4lee,r4mc,r4nue,r4ext,r4data,r4dirt]
        uproot_v = [ur4lee,ur4mc,ur4nue,ur4ext,ur4data,ur4dirt]
        for i,df in enumerate(df_v):
            up = uproot_v[i]
            if (loadnumuvariables == True):
                process_uproot_numu(up,df)
            if (loadeta == True):
                process_uproot_eta(up,df)
            if (loadshowervariables == True):
                process_uproot(up,df)
            if (loadrecoveryvars == True):
                process_uproot_recoveryvars(up,df)
            if (loadccncpi0vars == True):
                process_uproot_ccncpi0vars(up,df)

        r4_datasets = [r4lee, r4data, r4nue, r4mc, r4dirt, r4ext]#, r4data_two_showers_sidebands, r4data_np_far_sidebands, r4data_0p_far_sidebands]
        for r4_dataset in r4_datasets:
            r4_dataset['run1'] = np.zeros(len(r4_dataset), dtype=bool)
            r4_dataset['run2'] = np.zeros(len(r4_dataset), dtype=bool)
            r4_dataset['run3'] = np.zeros(len(r4_dataset), dtype=bool)
            r4_dataset['run30'] = np.zeros(len(r4_dataset), dtype=bool)
            r4_dataset['run12'] = np.zeros(len(r4_dataset), dtype=bool)
            r4_dataset['run4'] = np.ones(len(r4_dataset), dtype=bool)

    ################################# Run 3 ######################################

    if 3 in runs_to_load:
        print("Loading Run3 dataframes")
        r3nue = ur3nue.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3mc = ur3mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS + R3VARS, flatten=False)
        if (loadpi0filters):
            r3ncpi0 = ur3ncpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3ccpi0 = ur3ccpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3eta   = ur3eta.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        if (loadtruthfilters):
            r3ncpi0 = ur3ncpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3ccpi0 = ur3ccpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3ccnopi = ur3ccnopi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3cccpi = ur3cccpi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3ncnopi = ur3ncnopi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
            r3nccpi = ur3nccpi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3data = ur3data.pandas.df(VARIABLES, flatten=False)
        r3ext = ur3ext.pandas.df(VARIABLES + R3VARS, flatten=False)
        r3dirt = ur3dirt.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3lee = ur3lee.pandas.df(VARIABLES + WEIGHTSLEE + R3VARS, flatten=False)

        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r3data_np_far_sidebands = ur3data_np_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r3data_two_showers_sidebands = ur3data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "0p_far"):
            r3data_0p_far_sidebands = ur3data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "pi0"):
            r3data_pi0_sidebands = ur3data_pi0_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r3data_fulldata = ur3data_fulldata.pandas.df(VARIABLES, flatten=False)
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r3data_numu_sidebands   = ur3data_numu_sidebands.pandas.df(VARIABLES + R3VARS, flatten=False)
            
        r3lee["is_signal"] = r3lee["category"] == 11
        r3data["is_signal"] = r3data["category"] == 11
        r3nue["is_signal"] = r3nue["category"] == 11
        r3mc["is_signal"] = r3mc["category"] == 11
        r3dirt["is_signal"] = r3dirt["category"] == 11
        r3ext["is_signal"] = r3ext["category"] == 11

        # TODO: Do we need these filers for runs 4/5. eg. higher stats in the pi0 sidebands
        if (loadpi0filters):
            r3ncpi0["is_signal"] = r3ncpi0["category"] == 11
            r3ccpi0["is_signal"] = r3ccpi0["category"] == 11
            r3eta["is_signal"] = r3eta["category"] == 11    
        if (loadtruthfilters):
            r3ncpi0["is_signal"] = r3ncpi0["category"] == 11
            r3ccpi0["is_signal"] = r3ccpi0["category"] == 11
            r3ccnopi["is_signal"] = r3ccnopi["category"] == 11
            r3cccpi["is_signal"] = r3cccpi["category"] == 11
            r3ncnopi["is_signal"] = r3ncnopi["category"] == 11
            r3nccpi["is_signal"] = r3nccpi["category"] == 11
        r3lee.loc[r3lee['category'] == 1, 'category'] = 111
        r3lee.loc[r3lee['category'] == 10, 'category'] = 111
        r3lee.loc[r3lee['category'] == 11, 'category'] = 111

        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r3data_two_showers_sidebands["is_signal"] = r3data_two_showers_sidebands["category"] == 11
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r3data_np_far_sidebands["is_signal"] = r3data_np_far_sidebands["category"] == 11
        if (which_sideband == "0p_far"):
            r3data_0p_far_sidebands["is_signal"] = r3data_0p_far_sidebands["category"] == 11
        if (which_sideband == "pi0"):
            r3data_pi0_sidebands["is_signal"] = r3data_pi0_sidebands["category"] == 11
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r3data_fulldata["is_signal"] = r3data_fulldata["category"] == 11
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r3data_numu_sidebands["is_signal"]   = r3data_numu_sidebands["category"] == 11
        
        r3_datasets = [r3lee, r3data, r3nue, r3mc, r3dirt, r3ext]#, r3data_two_showers_sidebands, r3data_np_far_sidebands, r3data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r3_datasets += [r3data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r3_datasets += [r3data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            r3_datasets += [r3data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            r3_datasets += [r3data_pi0_sidebands]
        if (loadpi0filters == True):
            r3_datasets += [r3ncpi0, r3ccpi0, r3eta]        
        if (loadtruthfilters == True):
            r3_datasets += [r3ncpi0, r3ccpi0, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r3_datasets += [r3data_fulldata]
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r3_datasets += [r3data_numu_sidebands]
            
        for r3_dataset in r3_datasets:
            r3_dataset['run1'] = np.zeros(len(r3_dataset), dtype=bool)
            r3_dataset['run2'] = np.zeros(len(r3_dataset), dtype=bool)
            r3_dataset['run3'] = np.ones(len(r3_dataset), dtype=bool)
            r3_dataset['run30'] = np.ones(len(r3_dataset), dtype=bool)
            r3_dataset['run12'] = np.zeros(len(r3_dataset), dtype=bool)
            r3_dataset['run4'] = np.zeros(len(r3_dataset), dtype=bool)
            
        uproot_v = [ur3lee,ur3mc,ur3nue,ur3ext,ur3data,ur3dirt]#, ur3data_two_showers_sidebands, ur3data_np_far_sidebands, ur3data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            uproot_v += [ur3data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            uproot_v += [ur3data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            uproot_v += [ur3data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            uproot_v += [ur3data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            uproot_v += [ur3data_fulldata]
        if (loadpi0filters == True):
            uproot_v += [ur3ncpi0,ur3ccpi0,ur3eta]        
        if (loadtruthfilters == True):
            uproot_v += [ur3ncpi0,ur3ccpi0,ur3ccnopi, ur3cccpi, ur3ncnopi, ur3nccpi]
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            uproot_v += [ur3data_numu_sidebands]

        df_v = [r3lee,r3mc,r3nue,r3ext,r3data,r3dirt]#, r3data_two_showers_sidebands, r3data_np_far_sidebands, r3data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            df_v += [r3data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            df_v += [r3data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            df_v += [r3data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            df_v += [r3data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            df_v += [r3data_fulldata]
        if (loadpi0filters == True):
            df_v += [r3ncpi0,r3ccpi0,r3eta]        
        if (loadtruthfilters == True):
            df_v += [r3ncpi0,r3ccpi0,r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        #if (loadshowervariables == False):
            df_v += [r3data_numu_sidebands]

        #if (loadshowervariables == True):
        for i,df in enumerate(df_v):
            up = uproot_v[i]
            if (loadnumuvariables == True):
                process_uproot_numu(up,df)
            if (loadeta == True):
                process_uproot_eta(up,df)
            if (loadshowervariables == True):
                process_uproot(up,df)
            if (loadrecoveryvars == True):
                process_uproot_recoveryvars(up,df)
            if (loadccncpi0vars == True):
                process_uproot_ccncpi0vars(up,df)

        if ((USEBDT == True) and (loadtruthfilters == True)):
            dfcsv = pd.read_csv(ls.ntuple_path+ls.RUN3+"ccpi0nontrainevents.csv")
            dfcsv['identifier']   = dfcsv['run'] * 100000 + dfcsv['evt']
            if (loadtruthfilters == True):
                r3ccpi0['identifier'] = r3ccpi0['run'] * 100000 + r3ccpi0['evt']
                Npre = float(r3ccpi0.shape[0])
                r3ccpi0 = pd.merge(r3ccpi0, dfcsv, how='inner', on=['identifier'],suffixes=('', '_VAR'))
                Npost = float(r3ccpi0.shape[0])
                print ('fraction of R3 CCpi0 sample after split : %.02f'%(Npost/Npre))
            #train_r3ccpi0, r3ccpi0 = train_test_split(r3ccpi0, test_size=0.5, random_state=1990)

    ################################# Run 1 ######################################

    if 1 in runs_to_load:
        print("Loading Run1 dataframes")
        r1nue = ur1nue.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1mc = ur1mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS, flatten=False)
        if (loadpi0filters):
            r1ncpi0 = ur1ncpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1ccpi0 = ur1ccpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)    
        if (loadtruthfilters):
            r1ncpi0 = ur1ncpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1ccpi0 = ur1ccpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1ccnopi = ur1ccnopi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1cccpi = ur1cccpi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1ncnopi = ur1ncnopi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
            r1nccpi = ur1nccpi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1data = ur1data.pandas.df(VARIABLES, flatten=False)
        r1ext = ur1ext.pandas.df(VARIABLES, flatten=False)
        r1dirt = ur1dirt.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1lee = ur1lee.pandas.df(VARIABLES + WEIGHTSLEE, flatten=False)

        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r1data_np_far_sidebands = ur1data_np_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r1data_two_showers_sidebands = ur1data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "0p_far"):
            r1data_0p_far_sidebands = ur1data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "pi0"):
            r1data_pi0_sidebands = ur1data_pi0_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r1data_fulldata = ur1data_fulldata.pandas.df(VARIABLES, flatten=False)
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r1data_numu_sidebands = ur1data_numu_sidebands.pandas.df(VARIABLES, flatten=False)


        r1lee["is_signal"] = r1lee["category"] == 11
        r1data["is_signal"] = r1data["category"] == 11
        r1nue["is_signal"] = r1nue["category"] == 11
        r1mc["is_signal"] = r1mc["category"] == 11
        r1dirt["is_signal"] = r1dirt["category"] == 11
        r1ext["is_signal"] = r1ext["category"] == 11
        if (loadpi0filters):
            r1ncpi0["is_signal"] = r1ncpi0["category"] == 11
            r1ccpi0["is_signal"] = r1ccpi0["category"] == 11    
        if (loadtruthfilters):
            r1ncpi0["is_signal"] = r1ncpi0["category"] == 11
            r1ccpi0["is_signal"] = r1ccpi0["category"] == 11
            r1ccnopi["is_signal"] = r1ccnopi["category"] == 11
            r1cccpi["is_signal"] = r1cccpi["category"] == 11
            r1ncnopi["is_signal"] = r1ncnopi["category"] == 11
            r1nccpi["is_signal"] = r1nccpi["category"] == 11
        r1lee.loc[r1lee['category'] == 1, 'category'] = 111
        r1lee.loc[r1lee['category'] == 10, 'category'] = 111
        r1lee.loc[r1lee['category'] == 11, 'category'] = 111

        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r1data_np_far_sidebands["is_signal"] = r1data_np_far_sidebands["category"] == 11
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r1data_two_showers_sidebands["is_signal"] = r1data_two_showers_sidebands["category"] == 11
        if (which_sideband == "0p_far"):
            r1data_0p_far_sidebands["is_signal"] = r1data_0p_far_sidebands["category"] == 11
        if (which_sideband == "pi0"):
            r1data_pi0_sidebands["is_signal"] = r1data_pi0_sidebands["category"] == 11
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r1data_fulldata["is_signal"] = r1data_fulldata["category"] == 11
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r1data_numu_sidebands["is_signal"]   = r1data_numu_sidebands["category"] == 11
        
        r1_datasets = [r1lee, r1data, r1nue, r1mc, r1dirt, r1ext] #, r1data_two_showers_sidebands, r1data_np_far_sidebands, r1data_0p_far_sidebands]
        
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r1_datasets += [r1data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r1_datasets += [r1data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            r1_datasets += [r1data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            r1_datasets += [r1data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r1_datasets += [r1data_fulldata]
        if (loadpi0filters == True):
            r1_datasets += [r1ncpi0, r1ccpi0]        
        if (loadtruthfilters == True):
            r1_datasets += [r1ncpi0, r1ccpi0, r1ccnopi, r1cccpi, r1ncnopi, r1nccpi]
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r1_datasets += [r1data_numu_sidebands]

        for r1_dataset in r1_datasets:
            r1_dataset['run1'] = np.ones(len(r1_dataset), dtype=bool)
            r1_dataset['run2'] = np.zeros(len(r1_dataset), dtype=bool)
            r1_dataset['run3'] = np.zeros(len(r1_dataset), dtype=bool)
            # TODO: Check if this is correct?
            #r3_dataset['run30'] = np.zeros(len(r3_dataset), dtype=bool)
            r1_dataset['run30'] = np.zeros(len(r1_dataset), dtype=bool)
            r1_dataset['run12'] = np.ones(len(r1_dataset), dtype=bool)
            r1_dataset['run4'] = np.zeros(len(r1_dataset), dtype=bool)
            if (loadnumucrtonly == True):
                #r1_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
                r1_dataset["crtveto"] = np.zeros(len(r1_dataset),dtype=int)
                r1_dataset["crthitpe"] = np.zeros(len(r1_dataset),dtype=float)
                r1_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
        
        uproot_v = [ur1lee,ur1mc,ur1nue,ur1ext,ur1data,ur1dirt]#, ur1data_two_showers_sidebands, ur1data_np_far_sidebands, ur1data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            uproot_v += [ur1data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            uproot_v += [ur1data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            uproot_v += [ur1data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            uproot_v += [ur1data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            uproot_v += [ur1data_fulldata]
        if (loadpi0filters == True):
            uproot_v += [ur1ncpi0,ur1ccpi0]        
        if (loadtruthfilters == True):
            uproot_v += [ur1ncpi0,ur1ccpi0,ur1ccnopi, ur1cccpi, ur1ncnopi, ur1nccpi]
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            uproot_v += [ur1data_numu_sidebands]
            
        df_v = [r1lee,r1mc,r1nue,r1ext,r1data,r1dirt]#, r1data_two_showers_sidebands, r1data_np_far_sidebands, r1data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            df_v += [r1data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            df_v += [r1data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            df_v += [r1data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            df_v += [r1data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            df_v += [r1data_fulldata]
        if (loadpi0filters == True):
            df_v += [r1ncpi0,r1ccpi0]        
        if (loadtruthfilters == True):
            df_v += [r1ncpi0,r1ccpi0,r1ccnopi, r1cccpi, r1ncnopi, r1nccpi]
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            df_v += [r1data_numu_sidebands]

        #if (loadshowervariables == True):
        for i,df in enumerate(df_v):
            up = uproot_v[i]
            if (loadnumuvariables == True):
                process_uproot_numu(up,df)
            if (loadeta == True):
                process_uproot_eta(up,df)
            if (loadshowervariables == True):
                process_uproot(up,df)
            if (loadrecoveryvars == True):
                # trk_energy_tot agrees is getting reset somewhere in here
                process_uproot_recoveryvars(up,df)
            if (loadccncpi0vars == True):
                process_uproot_ccncpi0vars(up,df)


    ################################# Run 2 ######################################

    if 2 in runs_to_load:
        print("Loading Run2 dataframes")
        r2nue = ur2nue.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r2drt = ur2drt.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r2mc  = ur2mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS, flatten=False)
        r2ext = ur2ext.pandas.df(VARIABLES, flatten=False)
        r2lee = ur2lee.pandas.df(VARIABLES + WEIGHTSLEE, flatten=False)
        if (loadfakedata == 9):
            r2data = ur2data.pandas.df(VARIABLES, flatten=False)

        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r2data_np_far_sidebands = ur2data_np_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r2data_two_showers_sidebands = ur2data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "0p_far"):
            r2data_0p_far_sidebands = ur2data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "pi0"):
            r2data_pi0_sidebands = ur2data_pi0_sidebands.pandas.df(VARIABLES, flatten=False)
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r2data_fulldata = ur2data_fulldata.pandas.df(VARIABLES, flatten=False)
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r2data_numu_sidebands = ur2data_numu_sidebands.pandas.df(VARIABLES, flatten=False)
            
        r2lee["is_signal"] = r2lee["category"] == 11
        r2nue["is_signal"] = r2nue["category"] == 11
        r2mc["is_signal"] = r2mc["category"] == 11
        r2ext["is_signal"] = r2ext["category"] == 11
        r2lee.loc[r2lee['category'] == 1, 'category'] = 111
        r2lee.loc[r2lee['category'] == 10, 'category'] = 111
        r2lee.loc[r2lee['category'] == 11, 'category'] = 111

        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r2data_two_showers_sidebands["is_signal"] = r2data_two_showers_sidebands["category"] == 11
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r2data_np_far_sidebands["is_signal"] = r2data_np_far_sidebands["category"] == 11
        if (which_sideband == "0p_far"):
            r2data_0p_far_sidebands["is_signal"] = r2data_0p_far_sidebands["category"] == 11
        if (which_sideband == "pi0"):
            r2data_pi0_sidebands["is_signal"] = r2data_pi0_sidebands["category"] == 11
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r2data_fulldata["is_signal"] = r2data_fulldata["category"] == 11
            #r2data_fulldata["run"] = r2data_fulldata["run"] + 1
        #if (loadshowervariables == False):
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r2data_numu_sidebands["is_signal"] = r2data_numu_sidebands["category"] == 11
        
        r2_datasets = [r2lee, r2nue, r2mc, r2ext]#, r2data_two_showers_sidebands, r2data_np_far_sidebands, r2data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            r2_datasets += [r2data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            r2_datasets += [r2data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            r2_datasets += [r2data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            r2_datasets += [r2data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            r2_datasets += [r2data_fulldata]
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            r2_datasets += [r2data_numu_sidebands]
        if (loadfakedata == 9):
            r2_datasets += [r2data]
            
        for r2_dataset in r2_datasets:
            r2_dataset['run1'] = np.zeros(len(r2_dataset), dtype=bool)
            r2_dataset['run2'] = np.ones(len(r2_dataset), dtype=bool)
            r2_dataset['run3'] = np.zeros(len(r2_dataset), dtype=bool)
            
            if 3 in runs_to_load:
                r3_dataset['run30'] = np.zeros(len(r3_dataset), dtype=bool)
            r2_dataset['run12'] = np.ones(len(r2_dataset), dtype=bool)
            r2_dataset['run4'] = np.zeros(len(r2_dataset), dtype=bool)
            if (loadnumucrtonly == True):
                #r2_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
                r2_dataset["crtveto"] = np.zeros(len(r2_dataset),dtype=int)
                r2_dataset["crthitpe"] = np.zeros(len(r2_dataset),dtype=float)
                r2_dataset["_closestNuCosmicDist"] = np.zeros(len(r2_dataset),dtype=float)

        if 1 in runs_to_load:
            r1dirt['run2'] = np.ones(len(r1dirt), dtype=bool)
        
        if 3 in runs_to_load:
            r3dirt['run2'] = np.ones(len(r3dirt), dtype=bool)

        if (loadpi0filters == True):
            for r_dataset in [r1ncpi0, r1ccpi0, r3ncpi0, r3ccpi0, r3eta]:
                r_dataset['run2'] = np.ones(len(r_dataset), dtype=bool)
        
        if (loadtruthfilters == True):
            for r_dataset in [r1ncpi0, r1ccpi0, r3ncpi0, r3ccpi0,r1ccnopi, r1cccpi, r1ncnopi, r1nccpi, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]:
                r_dataset['run2'] = np.ones(len(r_dataset), dtype=bool)
        
        uproot_v = [ur2lee,ur2mc,ur2nue, ur2drt, ur2ext]#, ur2data_two_showers_sidebands, ur2data_np_far_sidebands, ur2data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            uproot_v += [ur2data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            uproot_v += [ur2data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            uproot_v += [ur2data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            uproot_v += [ur2data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            uproot_v += [ur2data_fulldata]
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            uproot_v += [ur2data_numu_sidebands]
        if (loadfakedata == 9):
            uproot_v += [ur2data]

        df_v = [r2lee,r2mc,r2nue, r2drt, r2ext]#, r2data_two_showers_sidebands, r2data_np_far_sidebands, r2data_0p_far_sidebands]
        if (which_sideband == "2plus_showers" or which_sideband == "np_sb_comb"):
            df_v += [r2data_two_showers_sidebands]
        if (which_sideband == "np_far" or which_sideband == "np_sb_comb"):
            df_v += [r2data_np_far_sidebands]
        if (which_sideband == "0p_far"):
            df_v += [r2data_0p_far_sidebands]
        if (which_sideband == "pi0"):
            df_v += [r2data_pi0_sidebands]
        if (which_sideband == "fulldata" or which_sideband == "fulldatawrun4open"):
            df_v += [r2data_fulldata]
        if ( (loadshowervariables == False) and (loadnumuntuples == True)):
            df_v += [r2data_numu_sidebands]
        if (loadfakedata == 9):
            df_v += [r2data]

        # Does some manipulation of the variables in the dataframes we're actually using
        for i,df in enumerate(df_v):
            up = uproot_v[i]
            if (loadnumuvariables == True):
                process_uproot_numu(up,df)
            if (loadeta == True):
                process_uproot_eta(up,df)
            if (loadshowervariables == True):
                process_uproot(up,df)
            if (loadrecoveryvars == True):
                process_uproot_recoveryvars(up,df)
            if (loadccncpi0vars == True):
                process_uproot_ccncpi0vars(up,df)

    print("Finished loading dataframes")

    ####################################### Done loading and processing dataframes ###################################################
 
    print("Concatenate dataframes")

    if 1 in runs_to_load:
        r1nue["pot_scale"] = 1
        r1lee["pot_scale"] = 1
        r1mc["pot_scale"] = 1
        r1ext["pot_scale"] = 1 

    if 2 in runs_to_load:
        r2nue["pot_scale"] = 1
        r2drt["pot_scale"] = 1 # TODO: Why does this operation only apply to the dirt from run2?
        r2lee["pot_scale"] = 1
        r2mc["pot_scale"] = 1
        r2ext["pot_scale"] = 1 

    if 3 in runs_to_load:
        r3nue["pot_scale"] = 1
        r3lee["pot_scale"] = 1
        r3mc["pot_scale"] = 1
        r3ext["pot_scale"] = 1 

    if 4 in runs_to_load:
        r4nue["pot_scale"] = 1
        r4lee["pot_scale"] = 1
        r4mc["pot_scale"] = 1
        r4ext["pot_scale"] = 1 

    # Concantate the mc together depending on which filters are being applied
    print("Concantating the various dataframes based on value of which_sideband")
    nue_to_concat = []
    if 1 in runs_to_load:
        nue_to_concat.append(r1nue)
    if 2 in runs_to_load:
        nue_to_concat.append(r2nue)
    if 3 in runs_to_load:
        nue_to_concat.append(r3nue)
    if 4 in runs_to_load:
        nue_to_concat.append(r4nue)
 
    #nue = pd.concat([r1nue,r2nue,r3nue],ignore_index=True)
    nue = pd.concat(nue_to_concat,ignore_index=True)

    mc_to_concat = []
    if 1 in runs_to_load:
        mc_to_concat.append(r1mc)
    if 2 in runs_to_load:
        mc_to_concat.append(r2mc)
    if 3 in runs_to_load:
        mc_to_concat.append(r3mc)
    if 4 in runs_to_load:
        mc_to_concat.append(r4mc)

    #nue = pd.concat([r3nue,r1nue],ignore_index=True)
    #mc = pd.concat([r3mc,r2mc,r1mc],ignore_index=True)
    mc = pd.concat(mc_to_concat,ignore_index=True)
 
    #mc = pd.concat([r3mc,r1mc],ignore_index=True)
    if (loadpi0filters == True):
        if 1 not in runs_to_load or 3 not in runs_to_load:
              raise Exception("load_data_run123: Can't load pion samples without first loading run 1/3 data, change runs_to_load")
        ncpi0 = pd.concat([r3ncpi0,r1ncpi0],ignore_index=True)
        ccpi0 = pd.concat([r3ccpi0,r1ccpi0],ignore_index=True,sort=True)
        eta   = pd.concat([r3eta],ignore_index=True)
    # TODO: Make these truth filtered samples for runs 4 and 5
    if (loadtruthfilters == True):
        if 1 not in runs_to_load or 3 not in runs_to_load:
              raise Exception("load_data_run123: Can't load truth filtered samples without first loading run 1/3 data, change runs_to_load")
        ncpi0 = pd.concat([r3ncpi0,r1ncpi0],ignore_index=True)
        ccpi0 = pd.concat([r3ccpi0,r1ccpi0],ignore_index=True,sort=True)
        ccnopi = pd.concat([r3ccnopi,r1ccnopi],ignore_index=True)
        cccpi = pd.concat([r3cccpi,r1cccpi],ignore_index=True)
        ncnopi = pd.concat([r3ncnopi,r1ncnopi],ignore_index=True)
        nccpi = pd.concat([r3nccpi,r1nccpi],ignore_index=True)
    # data = pd.concat([r3data,r1data],ignore_index=True)

    # Concantate the various data dataframes together
  
    data_to_concat = []
 
    if which_sideband == '2plus_showers':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_two_showers_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_two_showers_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_two_showers_sidebands)
        #data = pd.concat([r1data_two_showers_sidebands, r2data_two_showers_sidebands, r3data_two_showers_sidebands],ignore_index=True)

    elif which_sideband == 'np_far':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_np_far_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_np_far_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_np_far_sidebands)
        #data = pd.concat([r1data_np_far_sidebands, r2data_np_far_sidebands, r3data_np_far_sidebands],ignore_index=True)

    elif which_sideband == 'np_sb_comb':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_two_showers_sidebands)
            data_to_concat.append(r1data_np_far_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_two_showers_sidebands)
            data_to_concat.append(r2data_np_far_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_two_showers_sidebands)
            data_to_concat.append(r3data_np_far_sidebands)
        #data = pd.concat([r1data_np_far_sidebands, r1data_two_showers_sidebands, r2data_np_far_sidebands, \
        #                  r2data_two_showers_sidebands, r3data_np_far_sidebands, r3data_two_showers_sidebands],\
        #                 ignore_index=True)

    elif which_sideband == '0p_far':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_0p_far_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_0p_far_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_0p_far_sidebands)
        #data = pd.concat([r1data_0p_far_sidebands, r2data_0p_far_sidebands, r3data_0p_far_sidebands],ignore_index=True)

    elif which_sideband == 'pi0':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_pi0_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_pi0_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_pi0_sidebands)
        #data = pd.concat([r1data_pi0_sidebands, r2data_pi0_sidebands, r3data_pi0_sidebands],ignore_index=True)

    elif which_sideband == 'numu':
        if 1 in runs_to_load:
            data_to_concat.append(r1data_numu_sidebands)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_numu_sidebands)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_numu_sidebands)
        #data = pd.concat([r1data_numu_sidebands, r2data_numu_sidebands, r3data_numu_sidebands],ignore_index=True)


    elif which_sideband == "opendata":
        #data = pd.concat([r1data, r3data],ignore_index=True) # 5e19 and 1e19
        if 1 in runs_to_load:
            data_to_concat.append(r1data)
        if (loadfakedata == 9):
            #data = pd.concat([r1data, r2data, r3data],ignore_index=True) # NuWro
            if 2 in runs_to_load:
                data_to_concat.append(r2data)     
        if 3 in runs_to_load:
            data_to_concat.append(r3data)

    elif which_sideband == "fulldata":
        if 1 in runs_to_load:
            data_to_concat.append(r1data_fulldata)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_fulldata)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_fulldata)
        #data = pd.concat([r1data_fulldata, r2data_fulldata, r3data_fulldata],ignore_index=True) # full dataset

    elif which_sideband == 'run4opendata':
        if 4 in runs_to_load:
            data_to_concat.append(r4data)

    elif which_sideband == "fulldatawrun4open":
        if 1 in runs_to_load:
            data_to_concat.append(r1data_fulldata)
        if 2 in runs_to_load:
            data_to_concat.append(r2data_fulldata)
        if 3 in runs_to_load:
            data_to_concat.append(r3data_fulldata)
        if 4 in runs_to_load:
            data_to_concat.append(r4data)
        #data = pd.concat([r1data_fulldata, r2data_fulldata, r3data_fulldata,r4data],ignore_index=True) # full dataset
        
    data = pd.concat(data_to_concat,ignore_index=True)

    ext_to_concat = []
    if 1 in runs_to_load:
        ext_to_concat.append(r1ext)
    if 2 in runs_to_load:
        ext_to_concat.append(r2ext)
    if 3 in runs_to_load:
        ext_to_concat.append(r3ext)
    if 4 in runs_to_load:
        ext_to_concat.append(r4ext)
    ext = pd.concat(ext_to_concat,ignore_index=True)

    dirt_to_concat = []
    if 1 in runs_to_load:
        dirt_to_concat.append(r1dirt)
    if 2 in runs_to_load:
        dirt_to_concat.append(r2drt)
    if 3 in runs_to_load:
        dirt_to_concat.append(r3dirt)
    if 4 in runs_to_load:
        dirt_to_concat.append(r4dirt)
    dirt = pd.concat(dirt_to_concat,ignore_index=True)

    lee_to_concat = []
    if 1 in runs_to_load:
        lee_to_concat.append(r1lee)
    if 2 in runs_to_load:
        lee_to_concat.append(r2lee)
    if 3 in runs_to_load:
        lee_to_concat.append(r3lee)
    if 4 in runs_to_load:
        lee_to_concat.append(r4lee)
    lee = pd.concat(lee_to_concat,ignore_index=True)
    #lee = pd.concat([r1lee,r2lee,r3lee],ignore_index=True)

    #lee = pd.concat([r3lee,r1lee],ignore_index=True)

    ################################## Done concantating dataframes and deciding what to analyse ############################################

    print("Add derived variables")
    # update CRT hit to calibrate out time-dependence [DcoDB 24031] 
    #if (loadnumucrtonly):                                             
    #    ext.loc[  ext['run'] > 16300 , 'crthitpe'] = ext['crthitpe'] * 1.09
    #    data.loc[data['run'] > 16300 , 'crthitpe'] = data['crthitpe'] * 1.09

    df_v_mc = [lee,mc,nue,dirt]
    if (loadpi0filters == True):
        df_v_mc += [ncpi0,ccpi0,eta]
    if (loadtruthfilters == True):
        df_v_mc += [ccnopi,cccpi,ncnopi,nccpi,ncpi0,ccpi0]

    for i,df in enumerate(df_v_mc):

        # add MCC8-style weights
        df['weightMCC8'] = 1.0

        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.0) & (df['nu_e'] < 0.1)), 'weightMCC8' ] = 1./0.05
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.1) & (df['nu_e'] < 0.2)), 'weightMCC8' ] = 1./0.1
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.2) & (df['nu_e'] < 0.3)), 'weightMCC8' ] = 1./0.25
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.3) & (df['nu_e'] < 0.4)), 'weightMCC8' ] = 1./0.4
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.4) & (df['nu_e'] < 0.5)), 'weightMCC8' ] = 1./0.5
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.5) & (df['nu_e'] < 0.6)), 'weightMCC8' ] = 1./0.65
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.6) & (df['nu_e'] < 0.7)), 'weightMCC8' ] = 1./0.65
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.7) & (df['nu_e'] < 0.8)), 'weightMCC8' ] = 1./0.7
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.8) & (df['nu_e'] < 0.9)), 'weightMCC8' ] = 1./0.8
        df.loc[ ((df['nu_pdg'] == 12) & (df['nu_e'] > 0.9) & (df['nu_e'] < 1.0)), 'weightMCC8' ] = 1./0.85

        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.6) & (df['nu_e'] < 0.7)), 'weightMCC8' ] = 1./0.65
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.7) & (df['nu_e'] < 0.8)), 'weightMCC8' ] = 1./0.73
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.8) & (df['nu_e'] < 0.9)), 'weightMCC8' ] = 1./0.75
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.9) & (df['nu_e'] < 1.0)), 'weightMCC8' ] = 1./0.8
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.0) & (df['nu_e'] < 0.1)), 'weightMCC8' ] = 1./0.05
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.1) & (df['nu_e'] < 0.2)), 'weightMCC8' ] = 1./0.1
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.2) & (df['nu_e'] < 0.3)), 'weightMCC8' ] = 1./0.2
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.3) & (df['nu_e'] < 0.4)), 'weightMCC8' ] = 1./0.35
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.4) & (df['nu_e'] < 0.5)), 'weightMCC8' ] = 1./0.45
        df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.5) & (df['nu_e'] < 0.6)), 'weightMCC8' ] = 1./0.55
        #df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.6) & (df['nu_e'] < 0.7)), 'weightMCC8' ] = 1./0.65
        #df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.7) & (df['nu_e'] < 0.8)), 'weightMCC8' ] = 1./0.73
        #df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.8) & (df['nu_e'] < 0.9)), 'weightMCC8' ] = 1./0.75
        #df.loc[ ((df['nu_pdg'] == 14) & (df['nu_e'] > 0.9) & (df['nu_e'] < 1.0)), 'weightMCC8' ] = 1./0.8
    
        # TODO: May need to add the fix for the change in weighting convention going from run 123 -> run 45 here    
        df.loc[ df['weightTune'] <= 0, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] == np.inf, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] > 100, 'weightTune' ] = 1.
        df.loc[ np.isnan(df['weightTune']) == True, 'weightTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
        df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
        # code below matches the usage of weights in SBNfit:
        #df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
        #df.loc[ df['weightSplineTimesTune'] > 1000, 'weightSplineTimesTune' ] = 1.
        #df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.

        # 0p scaling
        #df.loc[ (df['nu_pdg']==12)&(df['nproton'] > 0) , 'weightSplineTimesTune'] = df['weightSplineTimesTune'] * 1.00 #0.973
        #df.loc[ (df['nu_pdg']==12)&(df['nproton'] == 0), 'weightSplineTimesTune'] = df['weightSplineTimesTune'] * 1.30 # 1.30
        
        # backtracker category
        '''
        df.loc[ ( (df['trk1_backtracked_pdg'] != 2212) & (df['trk1_backtracked_pdg'] != 13) & (df['trk1_backtracked_pdg'] != 11) & (df['trk1_backtracked_pdg'] != 111) & (df['trk1_backtracked_pdg'] != -13) & (df['trk1_backtracked_pdg'] != -11) & (df['trk1_backtracked_pdg'] != 211) & (df['trk1_backtracked_pdg'] != -211) & (df['trk1_backtracked_pdg'] != 2112) & (df['trk1_backtracked_pdg'] != 22) & (df['trk1_backtracked_pdg'] != 321) & (df['trk1_backtracked_pdg'] != -321) ), 'trk1_backtracked_pdg' ] = 0
        df.loc[ ( df['trk1_backtracked_pdg'] == -13), 'trk1_backtracked_pdg' ] = 13
        df.loc[ ( df['trk1_backtracked_pdg'] == -11), 'trk1_backtracked_pdg' ] = 11
        df.loc[ ( df['trk1_backtracked_pdg'] == -211), 'trk1_backtracked_pdg' ] = 211
        df.loc[ ( df['trk1_backtracked_pdg'] == -321), 'trk1_backtracked_pdg' ] = 321
        '''

        # Michel tag
        #df.loc[ (df['endmuonmichel'] > 0), 'category' ] = 222
        
        # flux parentage
        df['flux'] = np.zeros_like(df['nslice'])
        df.loc[ (((df['nu_pdg'] == 12) | (df['nu_pdg'] == -12)) & (df['nu_decay_mode'] < 11)) , 'flux'] = 10
        df.loc[ (((df['nu_pdg'] == 12) | (df['nu_pdg'] == -12)) & (df['nu_decay_mode'] > 10)) , 'flux'] = 1
        df.loc[ (((df['nu_pdg'] == 14) | (df['nu_pdg'] == -14)) & (df['nu_decay_mode'] < 11)) , 'flux'] = 10
        df.loc[ (((df['nu_pdg'] == 14) | (df['nu_pdg'] == -14)) & (df['nu_decay_mode'] > 10)) , 'flux'] = 1
        df['pi0weight'] = df['weightSpline']
        # pi0 scaling
        if pi0scaling == 1:
            df.loc[ df['npi0'] > 0, 'weightSplineTimesTune' ] = df['weightSpline'] * df['weightTune'] * 0.759
            df.loc[ df['npi0'] > 0, 'weightSpline' ] = df['weightSpline'] * 0.759
        elif pi0scaling == 2:
            pi0emax = 0.6
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] < pi0emax) , 'weightSplineTimesTune'] = df['weightSplineTimesTune']*(1.-0.4*df['pi0_e'])
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] >= pi0emax), 'weightSplineTimesTune'] = df['weightSplineTimesTune']*(1.-0.4*pi0emax)
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] < pi0emax) , 'weightSpline'] = df['weightSpline'] * (1.-0.4 * df['pi0_e'])
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] >= pi0emax), 'weightSpline'] = df['weightSpline'] * (1.-0.4 * pi0emax)
        elif pi0scaling == 3:            
            df.loc[ df['npi0'] == 1, 'weightSplineTimesTune' ] = df['weightSpline'] * df['weightTune'] * 0.835
            df.loc[ df['npi0'] == 1, 'weightSpline' ] = df['weightSpline'] * 0.835
            df.loc[ df['npi0'] == 2, 'weightSplineTimesTune' ] = df['weightSpline'] * df['weightTune'] * 0.795
            df.loc[ df['npi0'] == 2, 'weightSpline' ] = df['weightSpline'] * 0.795
        #
        # create branch to weigh each run according to its POT (to be modified later in the plotter notebook)
        #df["weightSplineTimesTuneTimesRunMod"] = df["weightSplineTimesTune"]

        if (loadeta==True):
            print("In loadeta")
            df['pi0_mass_truth'] = np.sqrt( 2 * df['pi0truth_gamma1_etot'] * df['pi0truth_gamma2_etot'] * (1 - df['pi0truth_gammadot']) )
            df.loc[ (df['pi0truth_gamma_parent']== 22) & (df['npi0'] == 0) & (df['true_nu_vtx_x'] > 10) & (df['true_nu_vtx_x'] < 250) & (df['true_nu_vtx_y'] > -110) & (df['true_nu_vtx_y'] < 110) & (df['true_nu_v\
tx_z'] > 25) & (df['true_nu_vtx_z'] < 990) , 'category' ] = 802
            df.loc[ (df['pi0truth_gamma_parent']== 22) & (df['npi0'] == 0) & ((df['true_nu_vtx_x'] < 10) | (df['true_nu_vtx_x'] > 250) | (df['true_nu_vtx_y'] < -110) | (df['true_nu_vtx_y'] > 110) | (df['true_nu_\
vtx_z'] < 25) | (df['true_nu_vtx_z'] > 990) ), 'category' ] = 801
            df.loc[ (df['pi0truth_gamma_parent']== 22) & (df['npi0'] >  0) , 'category' ] = 801
            df.loc[ (df['category']== 31) & (df['npi0'] == 1), 'category' ] = 803
            df.loc[ (df['category']== 21) & (df['npi0'] == 1), 'category' ] = 803
            df.loc[ (df['category']== 31) & (df['npi0'] >  1), 'category' ] = 804
            df.loc[ (df['category']== 21) & (df['npi0'] >  1), 'category' ] = 804
            #df.loc[ (df['category']== 31) & (df['npi0'] >  2), 'category' ] = 807
            #df.loc[ (df['category']== 21) & (df['npi0'] >  2), 'category' ] = 807
            df.loc[ (df['category']== 4), 'category' ] = 806
            df.loc[ (df['category']== 5), 'category' ] = 806
            df.loc[ (df['category']== 1),   'category' ] = 805
            df.loc[ (df['category']== 10),  'category' ] = 805
            df.loc[ (df['category']== 11),  'category' ] = 805
            df.loc[ (df['category']== 111), 'category' ] = 805
            df.loc[ (df['category']== 2 ),  'category' ] = 805
            df.loc[ (df['category']== 3 ),  'category' ] = 805

        df['dx'] = df['reco_nu_vtx_x']-df['true_nu_vtx_sce_x']
        df['dy'] = df['reco_nu_vtx_y']-df['true_nu_vtx_sce_y']
        df['dz'] = df['reco_nu_vtx_z']-df['true_nu_vtx_sce_z']
        df['dr'] = np.sqrt( df['dx']*df['dx'] + df['dy']*df['dy'] + df['dz']*df['dz'] )
        if (loadshowervariables):
            df['dx_s'] = df['shr_start_x']-df['true_nu_vtx_sce_x']
            df['dy_s'] = df['shr_start_y']-df['true_nu_vtx_sce_y']
            df['dz_s'] = df['shr_start_z']-df['true_nu_vtx_sce_z']
            df['dr_s'] = np.sqrt( df['dx_s']*df['dx_s'] + df['dy_s']*df['dy_s'] + df['dz_s']*df['dz_s'] )
        
    df_v = [lee,mc,nue,ext,data,dirt]

    if (loadpi0filters == True):
        df_v += [ncpi0,ccpi0,eta]
    if (loadtruthfilters == True):
        df_v += [ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi]

    if (loadshowervariables):
        for i,df in enumerate(df_v):
            df['subcluster'] = df['shrsubclusters0'] + df['shrsubclusters1'] + df['shrsubclusters2']
            #df['subcluster2'] = df['shr2subclusters0'] + df['shr2subclusters1'] + df['shr2subclusters2']
            #
            df['trkfit'] = df['shr_tkfit_npointsvalid'] / df['shr_tkfit_npoints']
            # and the 2d angle difference
            df['anglediff_Y'] = np.abs(df['secondshower_Y_dir']-df['shrclusdir2'])
            df['anglediff_V'] = np.abs(df['secondshower_V_dir']-df['shrclusdir1'])
            df['anglediff_U'] = np.abs(df['secondshower_U_dir']-df['shrclusdir0'])
            #
            #df["hitratio_shr12"] = (df["shr2_nhits"]/df["shr1_nhits"])
            #df["hitratio_mod_shr12"] = (df["shr2_nhits"]/(df["shr1_nhits"]*np.sqrt(df["shr1_nhits"])))
            #df["cos_shr12"] = np.sin(df["shr1_theta"])*np.cos(df["shr1_phi"])*np.sin(df["shr2_theta"])*np.cos(df["shr2_phi"])\
            #                  + np.sin(df["shr1_theta"])*np.sin(df["shr1_phi"])*np.sin(df["shr2_theta"])*np.sin(df["shr2_phi"])\
            #                  + np.cos(df["shr1_theta"])*np.cos(df["shr2_theta"])
            #df["tksh1_dist"] = np.sqrt( (df["shr1_start_x"]-df["trk_start_x"])**2 + (df["shr1_start_y"]-df["trk_start_y"])**2 + (df["shr1_start_z"]-df["trk_start_z"])**2)
            #df["tksh2_dist"] = np.sqrt( (df["shr2_start_x"]-df["trk_start_x"])**2 + (df["shr2_start_y"]-df["trk_start_y"])**2 + (df["shr2_start_z"]-df["trk_start_z"])**2)
            #df["min_tksh_dist"] = np.minimum(df["tksh1_dist"],df["tksh2_dist"])
            #df["max_tksh_dist"] = np.maximum(df["tksh1_dist"],df["tksh2_dist"])

    if (loadshowervariables):                    
        for i,df in enumerate(df_v):
            df["ptOverP"] = df["pt"]/df["p"]
            df["phi1MinusPhi2"] = df["shr_phi"]-df["trk_phi"]
            df["theta1PlusTheta2"] = df["shr_theta"]+df["trk_theta"]
            df['cos_shr_theta'] = np.cos(df['shr_theta'])
            df['cos_trk_theta'] = np.cos(df['trk_theta'])
            df.loc[df['n_tracks_tot']==0,'trk_theta'] = -9999
            df.loc[df['n_tracks_tot']==0,'cos_trk_theta'] = -9999
            if (loadrecoveryvars == True):
                df['n_tracks_cont_attach'] = df['n_tracks_contained']
                df.loc[((df['tk2sh1_distance']>3)&(df['tk2sh1_distance']<9999)),'n_tracks_cont_attach'] = df['n_tracks_contained']-1
            df['showergammadist'] = np.sqrt( (df['pi0truth_gamma1_zpos'] - df['shr_start_z'])**2 +
                                             (df['pi0truth_gamma1_ypos'] - df['shr_start_y'])**2 +
                                             (df['pi0truth_gamma1_xpos'] - df['shr_start_x'] + 1.08)**2 )
            df['bktrgammaenergydiff'] = np.abs(df['shr_bkt_E'] * 1000 - df['pi0truth_gamma1_etot'])

    if (loadpi0variables == True):
        for i,df in enumerate(df_v):
            df['asymm'] = np.abs(df['pi0_energy1_Y']-df['pi0_energy2_Y'])/(df['pi0_energy1_Y']+df['pi0_energy2_Y'])
            df['pi0energy'] = 134.98 * np.sqrt( 2. / ( (1-(df['asymm'])**2) * (1-df['pi0_gammadot']) ) )
            df["pi0energygev"] = df["pi0energy"]*0.001
            df['pi0momentum'] = np.sqrt(df['pi0energy']**2 - 134.98**2)
            df['pi0beta'] = df['pi0momentum']/df['pi0energy']
            df['pi0thetacm'] = df['asymm']/df['pi0beta']
            df['pi0momx'] = df['pi0_energy2_Y']*df['pi0_dir2_x'] + df['pi0_energy1_Y']*df['pi0_dir1_x']
            df['pi0momy'] = df['pi0_energy2_Y']*df['pi0_dir2_y'] + df['pi0_energy1_Y']*df['pi0_dir1_y']
            df['pi0momz'] = df['pi0_energy2_Y']*df['pi0_dir2_z'] + df['pi0_energy1_Y']*df['pi0_dir1_z']
            df['pi0energyraw'] = df['pi0_energy2_Y'] + df['pi0_energy1_Y']
            df['pi0energyraw_corr'] = df['pi0energyraw'] / 0.83
            df['pi0momanglecos'] = df['pi0momz'] / df['pi0energyraw']
            df['epicospi'] = df['pi0energy'] * (1-df['pi0momanglecos'])
            df['boost'] = (np.abs(df['pi0_energy1_Y']-df['pi0_energy2_Y'])/0.8)/(np.sqrt((df['pi0energy'])**2-135**2))
            df['pi0_mass_Y_corr'] = df['pi0_mass_Y']/0.83
            df['pi0energymin'] = 135. * np.sqrt( 2. / (1-df['pi0_gammadot']) )
            df['pi0energyminratio'] = df['pi0energyraw_corr'] / df['pi0energymin']
    if (loadeta == True):
        for i,df in enumerate(df_v):
            df['eta_mass_Y_corr'] = df['pi0_mass_Y_corr'] * 1.054
            df['minmass'] = df['pi0energyraw_corr'] * np.sqrt( (1 - df['pi0_gammadot']) / 2. )
            df['etaenergy'] = 547.86 * np.sqrt( 2. / ( (1-(df['asymm'])**2) * (1-df['pi0_gammadot']) ) )
            df['etamomentum'] = np.sqrt(df['etaenergy']**2 - 547.86**2)
            df['etabeta'] = df['etamomentum']/df['etaenergy']
            df['etathetacm'] = df['asymm']/df['etabeta']
            df['sintheta'] = np.sin(np.arccos(df['pi0_gammadot']))
            df['kinmass'] = df['sintheta'] * (df['pi0energyraw']/0.83)            

    # and a way to filter out data
    for i,df in enumerate(df_v):
        df["bnbdata"] = np.zeros_like(df["nslice"])
        df["extdata"] = np.zeros_like(df["nslice"])
    data["bnbdata"] = np.ones_like(data["nslice"])
    ext["extdata"] = np.ones_like(ext["nslice"])
    data["nu_decay_mode"] = np.zeros_like(data["nslice"])
    ext["nu_decay_mode"]  = np.zeros_like(ext["nslice"])

    if (loadshowervariables):
        for i,df in enumerate(df_v):
            df['shr_tkfit_nhits_tot'] = (df['shr_tkfit_nhits_Y']+df['shr_tkfit_nhits_U']+df['shr_tkfit_nhits_V'])
            #df['shr_tkfit_dedx_avg'] = (df['shr_tkfit_nhits_Y']*df['shr_tkfit_dedx_Y'] + df['shr_tkfit_nhits_U']*df['shr_tkfit_dedx_U'] + df['shr_tkfit_nhits_V']*df['shr_tkfit_dedx_V'])/df['shr_tkfit_nhits_tot']
            df['shr_tkfit_2cm_nhits_tot'] = (df['shr_tkfit_2cm_nhits_Y']+df['shr_tkfit_2cm_nhits_U']+df['shr_tkfit_2cm_nhits_V'])
            #df['shr_tkfit_2cm_dedx_avg'] = (df['shr_tkfit_2cm_nhits_Y']*df['shr_tkfit_2cm_dedx_Y'] + df['shr_tkfit_2cm_nhits_U']*df['shr_tkfit_2cm_dedx_U'] + df['shr_tkfit_2cm_nhits_V']*df['shr_tkfit_2cm_dedx_V'])/df['shr_tkfit_2cm_nhits_tot']
            df['shr_tkfit_gap10_nhits_tot'] = (df['shr_tkfit_gap10_nhits_Y']+df['shr_tkfit_gap10_nhits_U']+df['shr_tkfit_gap10_nhits_V'])
            #df['shr_tkfit_gap10_dedx_avg'] = (df['shr_tkfit_gap10_nhits_Y']*df['shr_tkfit_gap10_dedx_Y'] + df['shr_tkfit_gap10_nhits_U']*df['shr_tkfit_gap10_dedx_U'] + df['shr_tkfit_gap10_nhits_V']*df['shr_tkfit_gap10_dedx_V'])/df['shr_tkfit_gap10_nhits_tot']
            df.loc[:,'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_Y']
            df.loc[(df['shr_tkfit_nhits_U']>df['shr_tkfit_nhits_Y']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_U']
            df.loc[(df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_Y']) & (df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_U']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_V']

            df.loc[:,'shr_tkfit_2cm_dedx_max'] = df['shr_tkfit_2cm_dedx_Y']
            df.loc[(df['shr_tkfit_2cm_nhits_U']>df['shr_tkfit_2cm_nhits_Y']),'shr_tkfit_2cm_dedx_max'] = df['shr_tkfit_2cm_dedx_U']
            df.loc[(df['shr_tkfit_2cm_nhits_V']>df['shr_tkfit_2cm_nhits_Y']) & (df['shr_tkfit_2cm_nhits_V']>df['shr_tkfit_2cm_nhits_U']),'shr_tkfit_2cm_dedx_max'] = df['shr_tkfit_2cm_dedx_V']

            df.loc[:,'shr_tkfit_gap10_dedx_max'] = df['shr_tkfit_gap10_dedx_Y']
            df.loc[(df['shr_tkfit_gap10_nhits_U']>df['shr_tkfit_gap10_nhits_Y']),'shr_tkfit_gap10_dedx_max'] = df['shr_tkfit_gap10_dedx_U']
            df.loc[(df['shr_tkfit_gap10_nhits_V']>df['shr_tkfit_gap10_nhits_Y']) & (df['shr_tkfit_gap10_nhits_V']>df['shr_tkfit_gap10_nhits_U']),'shr_tkfit_gap10_dedx_max'] = df['shr_tkfit_gap10_dedx_V']

        INTERCEPT = 0.0
        SLOPE = 0.83

        Me = 0.511e-3
        Mp = 0.938
        Mn = 0.940
        Eb = 0.0285

        # define some energy-related variables
        for i,df in enumerate(df_v):
            df["reco_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE + df["trk_energy_tot"]
            df["reco_e_overflow"] = df["reco_e"]
            df.loc[ (df['reco_e'] >= 2.25), 'reco_e_overflow' ] = 2.24
            df["reco_e_mev"] = df["reco_e"] * 1000.
            df["reco_e_mev_overflow"] = df["reco_e_overflow"] * 1000.
            df['electron_e'] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE
            df['proton_ke'] = df['proton_e']-Mp
            df.loc[(df['proton_ke']<0), 'proton_ke'] = 0
            df['protonenergy_corr'] = df['protonenergy']+0.000620/df['protonenergy']-0.001792
            df.loc[(df['protonenergy_corr']>9998.), 'protonenergy_corr'] = 0
            df['reco_proton_e'] = Mp + df['protonenergy']
            df['reco_proton_p'] = np.sqrt( (df['reco_proton_e'])**2 - Mp**2 )
            df['reco_e_qe_l'] = ( df['electron_e'] * (Mn-Eb) + 0.5 * ( Mp**2 - (Mn - Eb)**2 - Me**2 ) ) / ( (Mn - Eb) - df['electron_e'] * (1 - np.cos(df['shr_theta'])) )
            df['reco_e_qe_p'] = ( df['reco_proton_e']   * (Mn-Eb) + 0.5 * ( Me**2 - (Mn - Eb)**2 - Mp**2 ) ) / ( (Mn - Eb) + df['reco_proton_p'] * np.cos(df['trk_theta']) - df['reco_proton_e'] )
            df["reco_e_qe"] = 0.938*((df["shr_energy"]+INTERCEPT)/SLOPE)/(0.938 - ((df["shr_energy"]+INTERCEPT)/SLOPE)*(1-np.cos(df["shr_theta"])))
            df["reco_e_rqe"] = df["reco_e_qe"]/df["reco_e"]


    # define ratio of deposited to total shower energy for pi0
    if (loadpi0variables):
        for i,df in enumerate(df_v):
            df['pi0truth_gamma1_edep_frac'] = df["pi0truth_gamma1_edep"]/df["pi0truth_gamma1_etot"]
            df['pi0truth_gamma2_edep_frac'] = df["pi0truth_gamma2_edep"]/df["pi0truth_gamma2_etot"]



    # set EXT and DIRT contributions to 0 for fake-data studies
    if (loadfakedata > 0):
        dirt['nslice'] = np.zeros_like(dirt['nslice'])
        ext['nslice']  = np.zeros_like(ext['nslice'])
        
    # add back the cosmic category, for background only
    for i,df in enumerate(df_v):        
        #print(i)
        #print(df.query("category == 4").shape[0])
        df.loc[(df['category']!=1)&(df['category']!=10)&(df['category']!=11)&(df['category']!=111)&(df['slnunhits']/df['slnhits']<0.2), 'category'] = 4
        if (loadeta == True):
            df.loc[ (df['category']== 4), 'category' ] = 806
        #print(df.query("category == 4").shape[0])

    # change proton threshold (50 MeV for xsec)
    if updatedProtThresh>0:
        for i,df in enumerate(df_v):
            df.loc[(df['category']==11)&(df['proton_ke']<updatedProtThresh), 'category'] = 10
            df.loc[(df['nproton']>0)&(df['proton_ke']<updatedProtThresh), 'nproton'] = 0

    # category switch
    '''
    for i,df in enumerate([nue]):
        #1e0p
        df.loc[(df['category']==5)&(df['ccnc']==0)&(df['nproton']==0)&(df['npi0']==0)&(df['npion']==0), 'category'] = 10
        #1eNp
        df.loc[(df['category']==5)&(df['ccnc']==0)&(df['nproton']>0)&(df['npi0']==0)&(df['npion']==0), 'category'] = 11
        #1eMpi
        #df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npi0']>0) | (df['npion']>0)), 'category'] = 1
        df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npi0']>0)), 'category'] = 1
        df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npion']>0)), 'category'] = 1
        #NCpi0
        df.loc[(df['category']==5)&(df['ccnc']==1)&(df['npi0']==1) & (df['npion']==0), 'category'] = 31
        #NCOther
        #df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']>1) | (df['npion']>0)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']==0)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']>1)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npion']>=0)), 'category'] = 3
    for i,df in enumerate([lee]):
        df.loc[(df['category']==5), 'category'] = 111
    df_filter_v = [mc,ncpi0,ccpi0,ccnopi,cccpi,ncnopi,nccpi,dirt]
    for i,df in enumerate(df_filter_v):
        #NCpi0
        df.loc[(df['category']==5)&(df['ccnc']==1)&(df['npi0']==1) & (df['npion']==0), 'category'] = 31
        #NCOther
        #df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']>1) | (df['npion']>0)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']==0)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npi0']>1)), 'category'] = 3
        df.loc[(df['category']==5)&(df['ccnc']==1)&((df['npion']>=0)), 'category'] = 3
        #CCpi0
        df.loc[(df['category']==5)&(df['ccnc']==0)&(df['npi0']==1) & (df['npion']==0), 'category'] = 21
        #CCOther
        #df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npi0']>1) | (df['npion']>0)), 'category'] = 2
        #CCOther
        df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npi0']==0)), 'category'] = 2
        df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npi0']>1)), 'category'] = 2
        df.loc[(df['category']==5)&(df['ccnc']==0)&((df['npion']>=0)), 'category'] = 2
    '''

    print("Add BDT scores")
    # Np BDT

    TRAINVAR = ["shr_score","tksh_distance","tksh_angle",
                "shr_tkfit_dedx_max",
                "trkfit","trkpid",
                "subcluster","shrmoliereavg",
                "trkshrhitdist2","hits_ratio",
                "secondshower_Y_nhit","secondshower_Y_vtxdist","secondshower_Y_dot","anglediff_Y",
                "CosmicIPAll3D","CosmicDirAll3D"]
    
    LABELS =  ['pi0','nonpi0']
    
    if (USEBDT == True):
        for label, bkg_query in zip(LABELS, nue_booster.bkg_queries):
            with open(ls.pickle_path+'booster_%s_0304_extnumi.pickle' % label, 'rb') as booster_file:
                booster = pickle.load(booster_file)
                for df in df_v:
                    df[label+"_score"] = booster.predict(
                        xgb.DMatrix(df[TRAINVAR]),
                        ntree_limit=booster.best_iteration)
                

    # 0p BDT
    TRAINVARZP = ['shrmoliereavg','shr_score', "trkfit","subcluster",
                  "CosmicIPAll3D","CosmicDirAll3D",
                  'secondshower_Y_nhit','secondshower_Y_vtxdist','secondshower_Y_dot','anglediff_Y',
                  'secondshower_V_nhit','secondshower_V_vtxdist','secondshower_V_dot','anglediff_V',
                  'secondshower_U_nhit','secondshower_U_vtxdist','secondshower_U_dot','anglediff_U',
                  "shr_tkfit_2cm_dedx_U", "shr_tkfit_2cm_dedx_V", "shr_tkfit_2cm_dedx_Y",
                  "shr_tkfit_gap10_dedx_U", "shr_tkfit_gap10_dedx_V", "shr_tkfit_gap10_dedx_Y",
                  "shrMCSMom","DeltaRMS2h","shrPCA1CMed_5cm","CylFrac2h_1cm"]

    LABELSZP = ['bkg']

    if (USEBDT == True):
        for label, bkg_query in zip(LABELSZP, nue_booster.bkg_queries):
            with open(ls.pickle_path+'booster_%s_0304_extnumi_vx.pickle' % label, 'rb') as booster_file:
                booster = pickle.load(booster_file)
                for df in df_v:
                    df[label+"_score"] = booster.predict(
                        xgb.DMatrix(df[TRAINVARZP]),
                        ntree_limit=booster.best_iteration)
    ## avoid recycling unbiased ext events (i.e. selecting a slice with little nu content from these samples)
    ## note: this needs to be after setting the BDT scores, so that we do not mess with copied data frames
    if (loadpi0filters == True):
        ncpi0 = ncpi0.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        eta   = eta.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
    if (loadtruthfilters == True):
        ccnopi = ccnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        cccpi = cccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        ncnopi = ncnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        nccpi = nccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        ncpi0 = ncpi0.query('(nslice==0 | (slnunhits/slnhits)>0.1)')

    # avoid double-counting of events out of FV in the NC/CC pi0 samples
    # not needed anymore since we improved matching with filtered samples
    #ncpi0 = ncpi0.query('category != 5')
    #ccpi0 = ccpi0.query('category != 5')
    #ccnopi = ccnopi.query('category != 5')
    #nccpi = nccpi.query('category != 5')
    #ncnopi = ncnopi.query('category != 5')

    lee['flux'] = 111

    if (loadtruthfilters == True):
        Npre = float(ncpi0.shape[0])
        ncpi0 = ncpi0.drop_duplicates(subset=['run','evt'],keep='last') # keep last since the recovery samples are added at the end                                  
        Npos = float(ncpi0.shape[0])
        #print ('fraction of ncpi0 surviving duplicate removal : %.02f'%(Npos/Npre))

    Npre = float(data.shape[0])
    if (loadfakedata == 0):
       data = data.drop_duplicates(subset=['run','evt'],keep='last') # keep last since the recovery samples are added at the end
    Npos = float(data.shape[0])
    #print ('fraction of data surviving duplicate removal : %.02f'%(Npos/Npre))

    Npre = float(ext.shape[0])
    ext = ext.drop_duplicates(subset=['run','evt'],keep='last') # keep last since the recovery samples are added at the end
    Npos = float(ext.shape[0])
    #print ('fraction of ext surviving duplicate removal : %.02f'%(Npos/Npre))

    samples = {
    "mc": mc,
    "nue": nue,
    "data": data,
    "ext": ext,
    "dirt": dirt,
    "lee": lee
    }

    if (loadpi0filters == True):
        samples["ncpi0"]  = ncpi0
        samples["ccpi0"]  = ccpi0
        samples["eta"] = eta

    if (loadtruthfilters == True):
        samples["ccnopi"] = ccnopi
        samples["cccpi"]  = cccpi
        samples["ncnopi"] = ncnopi
        samples["nccpi"]  = nccpi
        samples["ncpi0"]  = ncpi0
        samples["ccpi0"]  = ccpi0

    # these variables incate which category of events the events belong to, used for drawing plots! 
    for key, df in samples.items():
        df.loc[:,"paper_category"] = df["category"]
        if key is 'data': continue
        df.loc[ (df['paper_category']== 1 ),  'paper_category' ] = 11
        df.loc[ (df['paper_category']== 10 ),  'paper_category' ] = 11
        if key is 'nue':
            df.loc[(df['paper_category']==5)&(df['ccnc']==0), 'paper_category'] = 11
            df.loc[(df['paper_category']==5)&(df['ccnc']==1), 'paper_category'] = 2
            df.loc[(df['paper_category']==4)&(df['ccnc']==0), 'paper_category'] = 11
            df.loc[(df['paper_category']==4)&(df['ccnc']==1), 'paper_category'] = 2
            df.loc[(df['paper_category']==3), 'paper_category'] = 2
            continue
        if key is 'lee':
            df.loc[(df['paper_category']==4), 'paper_category'] = 111
            df.loc[(df['paper_category']==5), 'paper_category'] = 111
            df.loc[(df['paper_category']==3), 'paper_category'] = 2
            continue
        if key is 'dirt':
            df['paper_category'] = 2
            continue
        df.loc[(df['npi0']>0), 'paper_category'] = 31
        df.loc[(df['npi0']==0), 'paper_category'] = 2

    for key, df in samples.items():
        df.loc[:,"paper_category_xsec"] = df["category"]
        if key is 'data': continue
        df.loc[(df['npi0']>0), 'paper_category_xsec'] = 31
        df.loc[(df['npi0']==0), 'paper_category_xsec'] = 2
        df.loc[ (df['category']== 1 ),  'paper_category_xsec' ] = 1
        df.loc[ (df['category']== 10 ),  'paper_category_xsec' ] = 10
        df.loc[ (df['category']== 11 ),  'paper_category_xsec' ] = 11
        if key is 'nue':
            df.loc[(df['category']==5)&(df['ccnc']==1), 'paper_category_xsec'] = 2
            df.loc[(df['category']==4)&(df['ccnc']==1), 'paper_category_xsec'] = 2
            df.loc[(df['category']==3), 'paper_category_xsec'] = 2
            df.loc[(df['category']==5)&(df['ccnc']==0)&(df['npi0']>0), 'paper_category_xsec'] = 1
            df.loc[(df['category']==5)&(df['ccnc']==0)&(df['npion']>0), 'paper_category_xsec'] = 1
            df.loc[(df['category']==4)&(df['ccnc']==0)&(df['npi0']>0), 'paper_category_xsec'] = 1
            df.loc[(df['category']==4)&(df['ccnc']==0)&(df['npion']>0), 'paper_category_xsec'] = 1
            df.loc[(df['category']==5)&(df['ccnc']==0)&(df['npi0']==0)&(df['npion']==0)&(df['nproton']==0), 'paper_category_xsec'] = 10
            df.loc[(df['category']==4)&(df['ccnc']==0)&(df['npi0']==0)&(df['npion']==0)&(df['nproton']==0), 'paper_category_xsec'] = 10
            df.loc[(df['category']==5)&(df['ccnc']==0)&(df['npi0']==0)&(df['npion']==0)&(df['nproton']>0), 'paper_category_xsec'] = 11
            df.loc[(df['category']==4)&(df['ccnc']==0)&(df['npi0']==0)&(df['npion']==0)&(df['nproton']>0), 'paper_category_xsec'] = 11
            continue
        if key is 'dirt':
            df['paper_category_xsec'] = 2
            continue

    for key, df in samples.items():
        df.loc[:,"paper_category_numu"] = 0
        if key is 'data': continue
        df.loc[ (df['ccnc'] == 0), 'paper_category_numu'] = 2
        #df.loc[ (df['ccnc'] == 0) & (df['nproton'] == 0), 'paper_category_numu'] = 22
        #df.loc[ (df['ccnc'] == 0) & (df['nproton'] == 1), 'paper_category_numu'] = 23
        #df.loc[ (df['ccnc'] == 0) & (df['nproton'] == 2), 'paper_category_numu'] = 24
        #df.loc[ (df['ccnc'] == 0) & (df['nproton'] >= 3), 'paper_category_numu'] = 25
        df.loc[ (df['ccnc'] == 1), 'paper_category_numu'] = 3
        if key is 'nue':
            df.loc[ (df['ccnc'] == 0), 'paper_category_numu'] = 11
            continue
        if key is 'lee':
            df.loc[ (df['ccnc'] == 0), 'paper_category_numu'] =	111
            continue
        if key is 'dirt':
            df['paper_category'] = 5
            df['paper_category_numu'] = 5
            continue

    for key, df in samples.items():
        if key is 'data':   df.loc[:, "sample"] = 0
        if key is 'mc':     df.loc[:, "sample"] = 1
        if key is 'nue':    df.loc[:, "sample"] = 2
        if key is 'ext':    df.loc[:, "sample"] = 3
        if key is 'lee':    df.loc[:, "sample"] = 4
        if key is 'dirt':   df.loc[:, "sample"] = 5
        if key is 'ccnopi': df.loc[:, "sample"] = 6
        if key is 'cccpi':  df.loc[:, "sample"] = 7
        if key is 'ncnopi': df.loc[:, "sample"] = 8
        if key is 'nccpi':  df.loc[:, "sample"] = 9
        if key is 'ncpi0':  df.loc[:, "sample"] = 10
        if key is 'ccpi0':  df.loc[:, "sample"] = 11

    if return_plotter is True:
        scaling = 1

        SPLIT = 1.0
        if (USEBDT == True):
            SPLIT = 1.48

        #''' 0304
        weights = {
            "mc": 1.61e-01 * scaling, 
            "ext": 5.01e-01 * scaling, 
            "nue": 3.32e-03 * scaling,
            "lee": 3.32e-03 * scaling,
            "dirt": 9.09e-01 * scaling,
            "ncpi0": 1.19e-01 * scaling,
            "ccpi0": 5.92e-02 * SPLIT * scaling,
        }
        pot = 5.88e20*scaling

        if (loadpi0filters == True):
            weights["ncpi0"]  = 1.19e-01 * scaling
            weights["ccpi0"]  = 5.92e-02 * SPLIT * scaling
            weights["eta"] = 1.19e-01 * scaling 
        
        if (loadtruthfilters == True):
            weights["ccnopi"] = 6.48e-02 * scaling
            weights["cccpi"]  = 5.18e-02 * scaling
            weights["ncnopi"] = 5.60e-02 * scaling
            weights["nccpi"]  = 2.58e-02 * scaling
            weights["ncpi0"]  = 1.19e-01 * scaling
            weights["ccpi0"]  = 5.92e-02 * SPLIT * scaling

        my_plotter = plotter.Plotter(samples, weights, pot=pot)
        return my_plotter
    else:

        #print ('number of data entries returned is : ',data.shape)
        #print ('number of data entries returned is : ',samples['data'].shape)
        print("Done!")
        return samples

################################## Done loading the data ###########################################################

BLIND = 1.00
#BLIND = 1.00/40.9
#BLIND = 1.00/220.2
    
pot_data_unblinded = {
# v47 NTuples
    "farsideband" : {
        #1: (1.67E+20, 37094101),
        #2: (2.62E+20, 62168648),
        #3: (2.57E+20, 61381194),
        #123: (6.86E+20, 160643943), },
        # Np blind-safe
        1: (BLIND*1.67E+20, BLIND*37094101),
        2: (BLIND*2.62E+20, BLIND*62168648),
        3: (BLIND*2.57E+20, BLIND*61381194), },
        # Zp blind-safe
        #1: ((0.953)*1.67E+20, (0.953)*37094101),
        #2: ((0.953)*2.62E+20, (0.953)*62168648),
        #3: ((0.953)*2.57E+20, (0.953)*61381194),
        #123: (6.86E+20*(0.953), 160643943*(0.953)), },#1eNp bdt-validation plots
      #123: (6.86E+20, 160643943), },
# 0304 samples
#    "opendata" : {
#        1: (4.08E+19, 9028010),
#        2: (1.00E+01, 1),
#        3: (7.63E+18, 1838700), },
# 0628 samples
    "fulldata" : {
        1: (1.67E+20, 37094101),
        2: (2.62E+20, 62168648),
        3: (2.57E+20, 61381194),},
        #1: (4.54E+19, 10080350),
	#2: (9.43E+18, 2271036),
	#3: (9.43E+18, 2271036),},
    "opendata" : {
        1: (4.54E+19, 10080350),
        2: (1.00E+01, 1),
        3: (9.43E+18, 2271036),
        4: (1e19 , 1000), }, # TODO: Get the correct POT/trigger for run 4
    "pi0" : {
        1: (1.509E+20, 33582996),
        2: (2.411E+20, 56116016),
        3: (1.971E+20, 47133521), },
    "numu" : {
        #1: (9.00E+20, 36139233),
        1: (1.62E+20, 36139233),
        2: (2.62E+20, 62045760),
        3: (2.55E+20, 61012955),
        30: (2.13E+20, 51090727), }, # 30 = 3 G1
    "fakeset1" : {
        1: (2.07E+20, 32139256),
        2: (1.00E+01, 1),
        3: (2.94E+20, 44266555), },
    "fakeset2" : {
        1: (4.06E+20, 32139256),
        2: (1.00E+01, 1),
        3: (3.91E+20, 44266555), },
    "fakeset3" : {
        1: (4.02E+20, 32139256),
        2: (1.00E+01, 1),
        3: (3.72E+20, 44266555), },
    "fakeset4" : {
        1: (3.79E+20, 32139256),
        2: (1.00E+01, 1),
        3: (3.96E+20, 44266555), },
    "fakeset5" : {
        1: (9.00E+20, 32139256),
        2: (1.00E+01, 1),
        3: (1.00E+01, 1), },
    "fakeset9" : {
        1: (3.10E+20, 68857313),#N triggers approx based on fulldata and POT
        2: (3.10E+20, 73558324),
        3: (2.90E+20, 69262826), },

    # TODO: data in here for runs 1-3 are currently junk values
    "run4opendata" : {
        1: (4.54E+19, 10080350),
        2: (1.00E+01, 1),
        3: (9.43E+18, 2271036),
        4: (1.396e+20,33067363.0) },
    "fulldatawrun4open" : {
        1: (1.67E+20, 37094101),
        2: (2.62E+20, 62168648),
        3: (2.57E+20, 61381194),
        4: (1.396e+20,33067363.0) }
}

pot_mc_samples = {}

# v48
pot_mc_samples[30] = {
    'mc': 1.34E+21, # 1.33E+21,
    'nue':7.75E+22, # 7.73E+22,
    'lee': 7.75E+22, #7.73E+22,
    'ncpi0': 2.31E+21, # 2.29E+21,
    'ccpi0': (6.43E+21),#/2., # (6.40E+21)/2.,
    'dirt': 3.28E+20, # 3.20E+20,
    'ncnopi': 7.14E+21, #7.23E+21,
    'nccpi': 1.82E+22, # 1.80E+22,
    'ccnopi': 5.51E+21, # 5.51E+21,
    'cccpi': 5.18E+21, # 5.19E+21,
    'ext': 198642758, # 30 -> Run3 G-only 
}

pot_mc_samples[3] = {
    'mc': 1.34E+21, # 1.33E+21,
    #'mc': 19.68E+20, # DETVAR
    'nue':7.75E+22, # 7.73E+22,
    'lee': 7.75E+22, #7.73E+22,
    'ncpi0': 2.31E+21, # 2.29E+21,
    'ccpi0': (6.43E+21)/2., # (6.40E+21)/2.,
    'dirt': 3.28E+20, # 3.20E+20,
    'ncnopi': 1.59E+22, #7.23E+21,
    'nccpi': 3.63E+22, # 1.80E+22,
    'ccnopi': 1.11E+22, # 5.51E+21,
    'cccpi': 1.48E+22, # 5.19E+21,
    'eta': 2.41E+22,
    'ext': 205802114, # OLD: 214555174
}

pot_mc_samples[2] = {
    'mc': 1.02E+21, # 1.01E+21,
    'nue': 6.32E+22, # 6.41E+22,
    'lee': 6.32E+22, #6.41E+22,
    'ext': 153236385, # OLD: 152404980
    'dirt': 9.50E+20,
}

pot_mc_samples[1] = {
    'mc': 1.31E+21, # 1.30E+21,
    #'mc': 18.93E+20, #DETVAR
    'nue': 5.25E+22, # 5.25E+22,
    'lee': 5.25E+22, #5.25E+22,
    'ncpi0': 1.16E+22, #2.66E+21, # 2.63E+21,
    'ccpi0': 3.48E+21, # 3.45E+21,
    'dirt': 3.23E+20, # 3.21E+20,
    'ncnopi': 1.31E+22, # 4.24E+21,
    'nccpi': 2.76E+22, # 8.93E+21,
    'ccnopi': 1.14E+22, # 5.81E+21,
    'cccpi': 1.55E+22, # 6.04E+21,
    'eta': 1.03E+0,
    'ext': 65473410, # OLD: 65498807
}

# CT: Run 4 MC
pot_mc_samples[4] = {
    'ext': 92797963., # OLD: 65498807
    'dirt': 2.5824e+20, 
    'mc': 6.14283e+20,
    #'mc': 18.93E+20, #DETVAR
    'nue': 4.6575e+22, 
    'lee': 4.6575e+22, 
}

'''
# v43
# 30 -> Run3 for CRT-only data (epoch G)
pot_mc_samples[30] = {
    'mc': 1.33E+21,
    'nue': 7.73E+22,
    'lee': 7.73E+22,
    'ncpi0': 2.29E+21,
    'ccpi0': (6.40E+21)/2.,
    'dirt': 3.20E+20,
    'ncnopi': 7.23E+21,
    'nccpi': 1.80E+22,
    'ccnopi': 5.51E+21,
    'cccpi': 5.19E+21,
    'ext': 198642758, # 30 -> Run3 G-only
}

pot_mc_samples[3] = {
    'mc': 1.33E+21,
    'nue': 7.73E+22,
    'lee': 7.73E+22,
    'ncpi0': 2.29E+21,
    'ccpi0': (6.40E+21)/2.,
    'dirt': 3.20E+20,
    'ncnopi': 7.23E+21,
    'nccpi': 1.80E+22,
    'ccnopi': 5.51E+21,
    'cccpi': 5.19E+21,
    'ext': 214555174,
}

pot_mc_samples[2] = {
    'mc': 1.01E+21,
    'nue': 6.41E+22,
    'lee': 6.41E+22,
    'ext': 152404980,
}

pot_mc_samples[1] = {
    'mc': 1.30E+21,
    'nue': 5.25E+22,
    'lee': 5.25E+22,
    'ncpi0': 2.63E+21,
    'ccpi0': 3.45E+21,
    'dirt': 3.21E+20,
    'ncnopi': 4.24E+21,
    'nccpi': 8.93E+21,
    'ccnopi': 5.81E+21,
    'cccpi': 6.04E+21,
    'ext': 65498807,
}

'''

def get_weights(run,dataset="farsideband",scaling=1.0):
    print("Loading data for dataset",dataset,"run",run)
    assert run in [1, 2, 3, 123, 12, 30, 4, 1234] # CT: Adding run 4
    weights_out = {}
    if run in [1, 2, 3, 4]:
        if run == 4 and dataset != 'run4opendata' and dataset != 'fulldatawrun4open': return 0,0 
        pot_on, n_trig_on = pot_data_unblinded[dataset][run]
        for sample, pot in pot_mc_samples[run].items():
                if sample == 'ext':
                        weights_out[sample] = n_trig_on/pot
                else:
                        weights_out[sample] = pot_on/pot
        if run == 2:
            for sample in ['ncpi0', 'ccpi0', 'ncnopi', 'nccpi', 'ccnopi', 'cccpi']:
                weights_out[sample] = pot_on/(pot_mc_samples[1][sample] + pot_mc_samples[3][sample])
        pot_out = pot_on

    elif run == 1234:
        if dataset != 'run4opendata' and dataset != 'fulldatawrun4open': return 0,0 
        total_pot_on = 0
        total_n_trig_on = 0
        for run in [1, 2, 3, 4]:
            pot_on, n_trig_on = pot_data_unblinded[dataset][run]
            total_pot_on += pot_on
            total_n_trig_on += n_trig_on
        for sample in pot_mc_samples[1].keys():
            this_sample_pot = 0
            for run in [1, 2, 3, 4]:
                if sample in pot_mc_samples[run].keys():
                    this_sample_pot += pot_mc_samples[run][sample]
            if sample == 'ext':
                weights_out[sample] = total_n_trig_on/this_sample_pot
            else:
                weights_out[sample] = total_pot_on/this_sample_pot
        pot_out = total_pot_on

    elif run == 123:
        total_pot_on = 0
        total_n_trig_on = 0
        for run in [1, 2, 3]:
            pot_on, n_trig_on = pot_data_unblinded[dataset][run]
            total_pot_on += pot_on
            total_n_trig_on += n_trig_on
        for sample in pot_mc_samples[1].keys():
            this_sample_pot = 0
            for run in [1, 2, 3]:
                if sample in pot_mc_samples[run].keys():
                    this_sample_pot += pot_mc_samples[run][sample]
            if sample == 'ext':
                weights_out[sample] = total_n_trig_on/this_sample_pot
            else:
                weights_out[sample] = total_pot_on/this_sample_pot
        pot_out = total_pot_on
    elif run == 12:
        total_pot_on = 0
        total_n_trig_on = 0
        for run in [1, 2]:
            pot_on, n_trig_on = pot_data_unblinded[dataset][run]
            total_pot_on += pot_on
            total_n_trig_on += n_trig_on

        for sample in pot_mc_samples[1].keys():
            this_sample_pot = 0
            for run in [1, 2]:
                if sample in pot_mc_samples[run].keys():
                    this_sample_pot += pot_mc_samples[run][sample]
            if sample == 'ext':
                weights_out[sample] = total_n_trig_on/this_sample_pot
            else:
                weights_out[sample] = total_pot_on/this_sample_pot
        pot_out = total_pot_on
    if run == 30:
        pot_on, n_trig_on = pot_data_unblinded[dataset][30]
        for sample, pot in pot_mc_samples[30].items():
            if sample == 'ext':
                weights_out[sample] = n_trig_on/pot
            else:
                weights_out[sample] = pot_on/pot
        pot_out = pot_on

    for key, val in weights_out.items():
        weights_out[key] *= scaling
        
    return weights_out, pot_out
