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

def distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

def cosAngleTwoVecs(vx1,vy1,vz1,vx2,vy2,vz2):
    return (vx1*vx2 + vy1*vy2 + vz1*vz2)/(np.sqrt(vx1**2+vy1**2+vz1**2) * np.sqrt(vx2**2+vy2**2+vz2**2))

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
    trk_llr_pid_v_sel = get_elm_from_vec_idx(trk_llr_pid_v,trk_id)
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
    trk_score_v = up.array("trk_score_v")
    shr_mask = (trk_score_v<0.5)
    trk_mask = (trk_score_v>0.5)
    df['n_tracks_tot'] = trk_mask.sum()
    df['n_showers_tot'] = shr_mask.sum()
    trk_len_v = up.array("trk_len_v")
    df["n_trks_gt10cm"] = (trk_len_v[trk_mask>=0.5]>10).sum()
    df["n_trks_gt25cm"] = (trk_len_v[trk_mask>=0.5]>25).sum()
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
    df['tk1sh1_distance_alltk'] = np.where(df['n_tracks_tot']==0,99999,
                                     distance(df['shr_start_x'],       df['shr_start_y'],       df['shr_start_z'],\
                                              df['trk1_start_x_alltk'],df['trk1_start_y_alltk'],df['trk1_start_z_alltk']))
    df["tk1sh1_angle_alltk"] = np.where(df['n_tracks_tot']==0,99999,
                                  cosAngleTwoVecs(df["trk1_dir_x_alltk"],df["trk1_dir_y_alltk"],df["trk1_dir_z_alltk"],\
                                                  df["shr_px"],          df["shr_py"],          df["shr_pz"]))

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
    df['sh1sh2_distance'] = np.where(df['n_showers_contained']>1,\
                                     distance(df['shr2_start_x'], df['shr2_start_y'], df['shr2_start_z'],\
                                     df['shr_start_x'],df['shr_start_y'],df['shr_start_z']),\
                                     9999.)
    #
    df['shr2pid'] = get_elm_from_vec_idx(trk_llr_pid_v,shr2_id,9999.)
    df['shr2_score'] = get_elm_from_vec_idx(trk_score_v,shr2_id,9999.)
    #
    #df.drop(columns=['shr_start_x', 'shr_start_y', 'shr_start_z'])
    #df.drop(columns=['trk1_start_x_alltk', 'trk1_start_y_alltk', 'trk1_start_z_alltk'])
    #df.drop(columns=['trk1_dir_x_alltk', 'trk1_dir_y_alltk', 'trk1_dir_z_alltk'])
    #df.drop(columns=['shr2subclusters0', 'shr2subclusters1', 'shr2subclusters2'])
    #
    #pick_closest_shower(up,df)
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
    df["trk1_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v,trk_id,-9999.)
    df["trk2_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v,trk2_id,-9999.)
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
    df.loc[trk2srtshr & (df["trk2_energy"]<0.), "trk2_energy"] = 0.
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
    df.loc[shr2prtn & (df["shr2_energy"]<0.), "shr2_energy"] = 0.
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
    
    #trk_dir_x_v = up.array('trk_dir_x_v')
    #trk_dir_y_v = up.array('trk_dir_y_v')
    #trk_dir_z_v = up.array('trk_dir_z_v')
    
    trk_mask = (trk_score_v>0.0)

    df["trk1_score"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_score_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_x_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_end_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_end_z_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_x_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_beg_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_start_z_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_len"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_len_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_pid"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_llr_pid_v[trk_mask],trk_len_v[trk_mask])])
    df["rk1_range_proton"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_energy_proton_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_calo"]    = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_calo_energy_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_range_muon"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_range_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_mcs_muon"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_mcs_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk1_theta"] = np.cos(awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_theta_v[trk_mask],trk_len_v[trk_mask])]))
    df["trk1_phi"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_phi_v[trk_mask],trk_len_v[trk_mask])])

    # 2nd longest track
    df["trk2_len"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_len_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_pid"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_llr_pid_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_range_proton"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_energy_proton_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_calo"]    = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_calo_energy_y_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_range_muon"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_range_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_mcs_muon"]   = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_mcs_muon_mom_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_theta"] = np.cos(awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_theta_v[trk_mask],trk_len_v[trk_mask])]))
    df["trk2_phi"]   = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(trk_phi_v[trk_mask],trk_len_v[trk_mask])])
    
    # get element-wise reconstructed neutrino energy (for each index the value will be the neutrino energy assuming the track at that index is the muon)
    df['trk_energy_tot'] = trk_energy_proton_v.sum()
    muon_energy_correction_v = np.sqrt(trk_range_muon_mom_v**2 + 0.105**2) - trk_energy_proton_v
    # get element-wise MCS consistency
    muon_mcs_consistency_v    = ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v )
    muon_calo_consistency_v   = ( (trk_calo_energy_y_v - trk_range_muon_mom_v) / trk_range_muon_mom_v )
    proton_calo_consistency_v = ( (trk_calo_energy_y_v * 0.001 - trk_energy_proton_v) / trk_energy_proton_v )

    df["trk1_muon_mcs_consistency"]    = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(muon_mcs_consistency_v[trk_mask] ,trk_len_v[trk_mask])])
    df["trk1_muon_calo_consistency"]   = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(muon_calo_consistency_v[trk_mask],trk_len_v[trk_mask])])
    df["trk2_proton_calo_consistency"] = awkward.fromiter([vec[vid.argsort()[-2]] if len(vid)>1 else -9999. for vec,vid in zip(proton_calo_consistency_v[trk_mask],trk_len_v[trk_mask])])
    
    # apply numu selection as defined by Ryan
    trk_score_v = up.array("trk_score_v")
    muon_mask = (trk_score_v>0.8) & (trk_llr_pid_v > 0.2) \
                & (trk_start_x_v > 5.) & (trk_start_x_v < 251.) & (trk_end_x_v > 5.) & (trk_end_x_v < 251.) \
                & (trk_start_y_v > -110.) & (trk_start_y_v < 110.) & (trk_end_y_v > -110.) & (trk_end_y_v < 110.) \
                & (trk_start_z_v > 20.) & (trk_start_z_v < 986.) & (trk_end_z_v > 20.) & (trk_end_z_v < 986.) \
                & (trk_len_v > 10) & (trk_distance_v < 4.) & (pfp_generation_v == 2) \
                & ( ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v ) > -0.5 ) \
                & ( ( (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v ) < 0.5 )

    contained_track_mask = (trk_start_x_v > 5.) & (trk_start_x_v < 251.) & (trk_end_x_v > 5.) & (trk_end_x_v < 251.) \
                           & (trk_start_y_v > -110.) & (trk_start_y_v < 110.) & (trk_end_y_v > -110.) & (trk_end_y_v < 110.) \
                           & (trk_start_z_v > 20.) & (trk_start_z_v < 986.) & (trk_end_z_v > 20.) & (trk_end_z_v < 986.) \
                           & (trk_score_v>0.5)

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
    shr_mask = (trk_score_v<0.5)
    trk_mask = (trk_score_v>0.5)
    df['n_muons_tot'] = muon_mask.sum()
    df['n_tracks_tot'] = trk_mask.sum()
    df['n_tracks_contained'] = contained_track_mask.sum()
    df['n_showers_tot'] = shr_mask.sum()    
    
    #df["trk1_dir_x"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_x_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_dir_y"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_y_v[trk_mask],trk_len_v[trk_mask])])
    #df["trk1_dir_z"] = awkward.fromiter([vec[vid.argsort()[-1]] if len(vid)>0 else -9999. for vec,vid in zip(trk_dir_z_v[trk_mask],trk_len_v[trk_mask])])
    return

def process_uproot_eta(up,df):
    #
    trk_score_v = up.array("trk_score_v")
    shr_mask = (trk_score_v<0.5)
    #df['n_tracks_tot'] = trk_mask.sum()
    df['n_showers_tot'] = shr_mask.sum()


def load_data_run123(which_sideband='pi0', return_plotter=True, 
                     pi0scaling=0,
                     USEBDT=True,
                     loadpi0variables=False,
                     loadtruthfilters=True,
                     loadfakedata=0,
                     loadshowervariables=True,
                     loadnumuntuples=False,
                     loadnumuvariables=False,
                     loadnumucrtonly=False,
                     loadeta=False,
                     loadsystematics=True,
                     loadrecoveryvars=False):

    fold = ls.fold
    tree = "NeutrinoSelectionFilter"

    
    VARIABLES = [
        "nu_pdg", "slpdg", "backtracked_pdg", #"trk_score_v", 
        "category", "ccnc",
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
        "nproton", "mc_pdg", "slnunhits", "slnhits", "true_e_visible",
        "npi0","npion","pion_e","muon_e","pi0truth_elec_etot",
        "pi0_e", "evnunhits", "nslice", "interaction",
        "slclustfrac", "reco_nu_vtx_x", "reco_nu_vtx_y", "reco_nu_vtx_z",
        #"trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        #"trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
        #"trk_start_x_v","trk_start_z_v","trk_start_z_v",
        "topological_score",
        "isVtxInFiducial",
        "theta", # angle between incoming and outgoing leptons in radians
        #"nu_decay_mode","nu_hadron_pdg","nu_parent_pdg", # flux truth info
        #"shr_energy_tot_cali","selected","n_showers_contained",  # only if CC0piNp variables are saved!
    ]
    
    CRTVARS = ["crtveto","crthitpe","_closestNuCosmicDist"]
    
    WEIGHTS = ["weightSpline","weightTune","weightSplineTimesTune"]
    WEIGHTSLEE = ["weightSpline","weightTune","weightSplineTimesTune", "leeweight"]
    SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]
    MCFVARS = ["mcf_nu_e","mcf_lep_e","mcf_actvol","mcf_nmm","mcf_nmp","mcf_nem","mcf_nep","mcf_np0","mcf_npp",
               "mcf_npm","mcf_mcshr_elec_etot","mcf_pass_ccpi0","mcf_pass_ncpi0",
               "mcf_pass_ccnopi","mcf_pass_ncnopi","mcf_pass_cccpi","mcf_pass_nccpi"]
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
               "shr_px","shr_py","shr_pz","p", "pt", 
    ]
    
    NUMUVARS = []#'contained_fraction']
    
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
               "pi0truth_gamma1_etot","pi0truth_gamma2_etot","pi0truth_gammadot","pi0truth_gamma_parent"
    ]
    
    R3VARS = []
    
    
    
    # sample list
    R1BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2_5e19'
    R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C_all_reco2'
    #R1EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_C2_D1_D2_E1_E2_all_reco2' #Run1 + Run2
    R1NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2'
    R1NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
    R1NCPI0  = 'prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2'
    R1CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2'
    R1CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run1_reco2_reco2'
    R1CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'
    R1NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2'

    # dummy samples to load faster
    if (loadnumucrtonly):
        R1BNB = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1EXT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1NU  = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        R1DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2'
        
    R2NU = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run2_reco2_D1D2_reco2"
    R2NUE = "prodgenie_bnb_intrinsic_nue_overlay_run2_v08_00_00_35_run2a_reco2_reco2"
    R2EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_D_E_all_reco2'
    
    R3BNB = 'data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2_1e19'
    R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_F_G_all_reco2'
    if (loadnumucrtonly):
        R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_G_all_reco2'
    #R3EXT = 'data_extbnb_mcc9.1_v08_00_00_25_reco2_G_all_reco2'
    R3NU  = 'prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run3_reco2_G_reco2'
    R3NUE = 'prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'
    R3DRT = 'prodgenie_bnb_dirt_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2'
    R3NCPI0  = 'prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2'
    R3CCPI0  = 'prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2'
    R3CCNOPI = 'prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run3_reco2_reco2'
    R3CCCPI  = 'prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run3_reco2_reco2'
    R3NCNOPI = 'prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_new_run3_reco2_reco2'
    R3NCCPI  = 'prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_New_run3_reco2_reco2'

    print("Loading uproot files")
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
    ur3ext = uproot.open(ls.ntuple_path+ls.RUN3+R3EXT+ls.APPEND+".root")[fold][tree]
    ur3dirt = uproot.open(ls.ntuple_path+ls.RUN3+R3DRT+ls.APPEND+".root")[fold][tree]
    ur3lee = uproot.open(ls.ntuple_path+ls.RUN3+R3NUE+ls.APPEND+".root")[fold][tree]
    if (loadtruthfilters):
        ur3ccnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCNOPI+ls.APPEND+".root")[fold][tree]
        ur3cccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3CCCPI+ls.APPEND+".root")[fold][tree]
        ur3ncnopi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCNOPI+ls.APPEND+".root")[fold][tree]
        ur3nccpi = uproot.open(ls.ntuple_path+ls.RUN3+R3NCCPI+ls.APPEND+".root")[fold][tree]
        ur3ncpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3NCPI0+ls.APPEND+".root")[fold][tree]
        ur3ccpi0 = uproot.open(ls.ntuple_path+ls.RUN3+R3CCPI0+ls.APPEND+".root")[fold][tree]

    ur2mc = uproot.open(ls.ntuple_path+ls.RUN2+R2NU+ls.APPEND+".root")[fold][tree]
    ur2nue = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
    ur2lee = uproot.open(ls.ntuple_path+ls.RUN2+R2NUE+ls.APPEND+".root")[fold][tree]
    ur2ext = uproot.open(ls.ntuple_path+ls.RUN2+R2EXT+ls.APPEND+".root")[fold][tree]

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
    ur1ext = uproot.open(ls.ntuple_path+ls.RUN1+R1EXT+ls.APPEND+".root")[fold][tree]
    ur1dirt = uproot.open(ls.ntuple_path+ls.RUN1+R1DRT+ls.APPEND+".root")[fold][tree]
    ur1lee = uproot.open(ls.ntuple_path+ls.RUN1+R1NUE+ls.APPEND+".root")[fold][tree]
    if (loadtruthfilters):
        ur1ccnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCNOPI+ls.APPEND+".root")[fold][tree]
        ur1cccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1CCCPI+ls.APPEND+".root")[fold][tree]
        ur1ncnopi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCNOPI+ls.APPEND+".root")[fold][tree]
        ur1nccpi = uproot.open(ls.ntuple_path+ls.RUN1+R1NCCPI+ls.APPEND+".root")[fold][tree]
        ur1ncpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1NCPI0+ls.APPEND+".root")[fold][tree]
        ur1ccpi0 = uproot.open(ls.ntuple_path+ls.RUN1+R1CCPI0+ls.APPEND+".root")[fold][tree]
    
    R123_TWO_SHOWERS_SIDEBAND_BNB = 'neutrinoselection_filt_1e_2showers_sideband_skimmed_extended_v47'
    R123_NP_FAR_SIDEBAND_BNB = 'neutrinoselection_filt_1enp_far_sideband_skimmed_extended_v47'
    R123_0P_FAR_SIDEBAND_BNB = 'neutrinoselection_filt_1e0p_far_sideband_skimmed_v47'
    R123_NP_RECOVERY_BNB = 'bnb_recovery'
    R123_NP_RECOVERY_EXT = 'bnb_recovery_ext'
    
    ur1data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur2data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur3data_two_showers_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_TWO_SHOWERS_SIDEBAND_BNB+".root")['nuselection'][tree]

    ur1data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur2data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur3data_np_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_NP_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    
    ur1data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur2data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]
    ur3data_0p_far_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_0P_FAR_SIDEBAND_BNB+".root")['nuselection'][tree]

    if (loadrecoveryvars == True):
        ur1ext_np_recovery_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_NP_RECOVERY_EXT+".root")['nuselection'][tree]
        ur2ext_np_recovery_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_NP_RECOVERY_EXT+".root")['nuselection'][tree]
        ur3ext_np_recovery_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_NP_RECOVERY_EXT+".root")['nuselection'][tree]

    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        R123_NUMU_SIDEBAND_BNB = 'neutrinoselection_filt_numu_ALL'
        ur1data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run1_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
        ur2data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run2_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
        if (loadnumucrtonly):
            ur3data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/'+"data_bnb_peleeFilter_uboone_v08_00_00_41_pot_run3_G1_neutrinoselection_filt.root")['nuselection'][tree]
        else:
            ur3data_numu_sidebands = uproot.open(ls.ntuple_path+'farsidebands/run3_'+R123_NUMU_SIDEBAND_BNB+".root")['nuselection'][tree]
        

    if (loadnumucrtonly ==True):
        R3VARS += CRTVARS

    if (loadsystematics == True):
        WEIGHTS += SYSTVARS
        WEIGHTSLEE += SYSTVARS
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

    print (VARIABLES)

    print("Loading Run3 dataframes")
    r3nue = ur3nue.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
    r3mc = ur3mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS + R3VARS, flatten=False)
    if (loadtruthfilters):
        r3ncpi0 = ur3ncpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3ccpi0 = ur3ccpi0.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3ccnopi = ur3ccnopi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3cccpi = ur3cccpi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3ncnopi = ur3ncnopi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
        r3nccpi = ur3nccpi.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
    r3data = ur3data.pandas.df(VARIABLES, flatten=False)
    print ('r3data has shape : ',r3data.shape)
    r3ext = ur3ext.pandas.df(VARIABLES + R3VARS, flatten=False)
    r3dirt = ur3dirt.pandas.df(VARIABLES + WEIGHTS + R3VARS, flatten=False)
    r3lee = ur3lee.pandas.df(VARIABLES + WEIGHTSLEE + R3VARS, flatten=False)
    
    r3data_two_showers_sidebands = ur3data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False) # note removed R3VARS due to missing CRT vars
    r3data_np_far_sidebands = ur3data_np_far_sidebands.pandas.df(VARIABLES, flatten=False) # note removed R3VARS due to missing CRT vars
    r3data_0p_far_sidebands = ur3data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False) # note removed R3VARS due to missing CRT vars
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r3data_numu_sidebands   = ur3data_numu_sidebands.pandas.df(VARIABLES + R3VARS, flatten=False)
    if (loadrecoveryvars == True):
        r3ext_np_recovery_sidebands = ur3ext_np_recovery_sidebands.pandas.df(VARIABLES, flatten=False)
        
    r3lee["is_signal"] = r3lee["category"] == 11
    r3data["is_signal"] = r3data["category"] == 11
    r3nue["is_signal"] = r3nue["category"] == 11
    r3mc["is_signal"] = r3mc["category"] == 11
    r3dirt["is_signal"] = r3dirt["category"] == 11
    r3ext["is_signal"] = r3ext["category"] == 11
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
    
    r3data_two_showers_sidebands["is_signal"] = r3data_two_showers_sidebands["category"] == 11
    r3data_np_far_sidebands["is_signal"] = r3data_np_far_sidebands["category"] == 11
    r3data_0p_far_sidebands["is_signal"] = r3data_0p_far_sidebands["category"] == 11
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r3data_numu_sidebands["is_signal"]   = r3data_numu_sidebands["category"] == 11
    if (loadrecoveryvars == True):
        r3ext_np_recovery_sidebands["is_signal"] = r3ext_np_recovery_sidebands["category"] == 11
    
    r3_datasets = [r3lee, r3data, r3nue, r3mc, r3dirt, r3ext, r3lee, r3lee, r3lee, r3data_two_showers_sidebands, r3data_np_far_sidebands, r3data_0p_far_sidebands]
    if (loadtruthfilters == True):
        r3_datasets += [r3ncpi0, r3ccpi0, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r3_datasets += [r3data_numu_sidebands]
    if (loadrecoveryvars == True):
        r3_datasets += [r3ext_np_recovery_sidebands]
        
    for r3_dataset in r3_datasets:
        r3_dataset['run1'] = np.zeros(len(r3_dataset), dtype=bool)
        r3_dataset['run2'] = np.zeros(len(r3_dataset), dtype=bool)
        r3_dataset['run3'] = np.ones(len(r3_dataset), dtype=bool)
        r3_dataset['run12'] = np.zeros(len(r3_dataset), dtype=bool)
        
    uproot_v = [ur3lee,ur3mc,ur3nue,ur3ext,ur3data,ur3dirt, ur3data_two_showers_sidebands, ur3data_np_far_sidebands, ur3data_0p_far_sidebands]
    if (loadtruthfilters == True):
        uproot_v += [ur3ncpi0,ur3ccpi0,ur3ccnopi, ur3cccpi, ur3ncnopi, ur3nccpi]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        uproot_v += [ur3data_numu_sidebands]
    if (loadrecoveryvars == True):
        uproot_v += [ur3ext_np_recovery_sidebands]

    df_v = [r3lee,r3mc,r3nue,r3ext,r3data,r3dirt, r3data_two_showers_sidebands, r3data_np_far_sidebands, r3data_0p_far_sidebands]
    if (loadtruthfilters == True):
        df_v += [r3ncpi0,r3ccpi0,r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
    #if (loadshowervariables == False):
        df_v += [r3data_numu_sidebands]
    if (loadrecoveryvars == True):
        df_v += [r3ext_np_recovery_sidebands]

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

    if (USEBDT == True):
        dfcsv = pd.read_csv(ls.ntuple_path+ls.RUN3+"ccpi0nontrainevents.csv")
        dfcsv['identifier']   = dfcsv['run'] * 100000 + dfcsv['evt']
        r3ccpi0['identifier'] = r3ccpi0['run'] * 100000 + r3ccpi0['evt']
        Npre = float(r3ccpi0.shape[0])
        r3ccpi0 = pd.merge(r3ccpi0, dfcsv, how='inner', on=['identifier'],suffixes=('', '_VAR'))
        Npost = float(r3ccpi0.shape[0])
        print ('fraction of R3 CCpi0 sample after split : %.02f'%(Npost/Npre))
        #train_r3ccpi0, r3ccpi0 = train_test_split(r3ccpi0, test_size=0.5, random_state=1990)

    print("Loading Run1 dataframes")
    r1nue = ur1nue.pandas.df(VARIABLES + WEIGHTS, flatten=False)
    r1mc = ur1mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS, flatten=False)
    if (loadtruthfilters):
        r1ncpi0 = ur1ncpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1ccpi0 = ur1ccpi0.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1ccnopi = ur1ccnopi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1cccpi = ur1cccpi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1ncnopi = ur1ncnopi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
        r1nccpi = ur1nccpi.pandas.df(VARIABLES + WEIGHTS, flatten=False)
    r1data = ur1data.pandas.df(VARIABLES, flatten=False)
    print ('r1data has shape : ',r1data.shape)
    r1ext = ur1ext.pandas.df(VARIABLES, flatten=False)
    r1dirt = ur1dirt.pandas.df(VARIABLES + WEIGHTS, flatten=False)
    r1lee = ur1lee.pandas.df(VARIABLES + WEIGHTSLEE, flatten=False)


    r1data_two_showers_sidebands = ur1data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False)
    r1data_np_far_sidebands = ur1data_np_far_sidebands.pandas.df(VARIABLES, flatten=False)
    r1data_0p_far_sidebands = ur1data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False)
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r1data_numu_sidebands = ur1data_numu_sidebands.pandas.df(VARIABLES, flatten=False)
    if (loadrecoveryvars == True):
        r1ext_np_recovery_sidebands = ur1ext_np_recovery_sidebands.pandas.df(VARIABLES, flatten=False)

    r1lee["is_signal"] = r1lee["category"] == 11
    r1data["is_signal"] = r1data["category"] == 11
    r1nue["is_signal"] = r1nue["category"] == 11
    r1mc["is_signal"] = r1mc["category"] == 11
    r1dirt["is_signal"] = r1dirt["category"] == 11
    r1ext["is_signal"] = r1ext["category"] == 11
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

    r1data_two_showers_sidebands["is_signal"] = r1data_two_showers_sidebands["category"] == 11
    r1data_np_far_sidebands["is_signal"] = r1data_np_far_sidebands["category"] == 11
    r1data_0p_far_sidebands["is_signal"] = r1data_0p_far_sidebands["category"] == 11
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r1data_numu_sidebands["is_signal"]   = r1data_numu_sidebands["category"] == 11
    if (loadrecoveryvars == True):
        r1ext_np_recovery_sidebands["is_signal"] = r1ext_np_recovery_sidebands["category"] == 11
    
    r1_datasets = [r1lee, r1data, r1nue, r1mc, r1dirt, r1ext, r1lee, r1data_two_showers_sidebands, r1data_np_far_sidebands, r1data_0p_far_sidebands]
    if (loadtruthfilters == True):
        r1_datasets += [r1ncpi0, r1ccpi0, r1ccnopi, r1cccpi, r1ncnopi, r1nccpi]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r1_datasets += [r1data_numu_sidebands]
    if (loadrecoveryvars == True):
        r1_datasets += [r1ext_np_recovery_sidebands]

    for r1_dataset in r1_datasets:
        r1_dataset['run1'] = np.ones(len(r1_dataset), dtype=bool)
        r1_dataset['run2'] = np.zeros(len(r1_dataset), dtype=bool)
        r1_dataset['run3'] = np.zeros(len(r1_dataset), dtype=bool)
        r1_dataset['run12'] = np.ones(len(r1_dataset), dtype=bool)
        if (loadnumucrtonly == True):
            #r1_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
            r1_dataset["crtveto"] = np.zeros(len(r1_dataset),dtype=int)
            r1_dataset["crthitpe"] = np.zeros(len(r1_dataset),dtype=float)
            r1_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
    
    uproot_v = [ur1lee,ur1mc,ur1nue,ur1ext,ur1data,ur1dirt, ur1data_two_showers_sidebands, ur1data_np_far_sidebands, ur1data_0p_far_sidebands]
    if (loadtruthfilters == True):
        uproot_v += [ur1ncpi0,ur1ccpi0,ur1ccnopi, ur1cccpi, ur1ncnopi, ur1nccpi]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        uproot_v += [ur1data_numu_sidebands]
    if (loadrecoveryvars == True):
        uproot_v += [ur1ext_np_recovery_sidebands]
        
    df_v = [r1lee,r1mc,r1nue,r1ext,r1data,r1dirt, r1data_two_showers_sidebands, r1data_np_far_sidebands, r1data_0p_far_sidebands]
    if (loadtruthfilters == True):
        df_v += [r1ncpi0,r1ccpi0,r1ccnopi, r1cccpi, r1ncnopi, r1nccpi]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        df_v += [r1data_numu_sidebands]
    if (loadrecoveryvars == True):
        df_v += [r1ext_np_recovery_sidebands]

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

    print("Loading Run2 dataframes")
    r2nue = ur2nue.pandas.df(VARIABLES + WEIGHTS, flatten=False)
    r2mc = ur2mc.pandas.df(VARIABLES + WEIGHTS + MCFVARS, flatten=False)
    r2ext = ur2ext.pandas.df(VARIABLES, flatten=False)
    r2lee = ur2lee.pandas.df(VARIABLES + WEIGHTSLEE, flatten=False)
    
    r2data_two_showers_sidebands = ur2data_two_showers_sidebands.pandas.df(VARIABLES, flatten=False)
    r2data_np_far_sidebands = ur2data_np_far_sidebands.pandas.df(VARIABLES, flatten=False)
    r2data_0p_far_sidebands = ur2data_0p_far_sidebands.pandas.df(VARIABLES, flatten=False)
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r2data_numu_sidebands = ur2data_numu_sidebands.pandas.df(VARIABLES, flatten=False)
    if (loadrecoveryvars == True):
        r2ext_np_recovery_sidebands = ur2ext_np_recovery_sidebands.pandas.df(VARIABLES, flatten=False)
        
    r2lee["is_signal"] = r2lee["category"] == 11
    r2nue["is_signal"] = r2nue["category"] == 11
    r2mc["is_signal"] = r2mc["category"] == 11
    r2ext["is_signal"] = r2ext["category"] == 11
    r2lee.loc[r2lee['category'] == 1, 'category'] = 111
    r2lee.loc[r2lee['category'] == 10, 'category'] = 111
    r2lee.loc[r2lee['category'] == 11, 'category'] = 111
    
    r2data_two_showers_sidebands["is_signal"] = r2data_two_showers_sidebands["category"] == 11
    r2data_np_far_sidebands["is_signal"] = r2data_np_far_sidebands["category"] == 11
    r2data_0p_far_sidebands["is_signal"] = r2data_0p_far_sidebands["category"] == 11
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r2data_numu_sidebands["is_signal"] = r2data_numu_sidebands["category"] == 11
    if (loadrecoveryvars == True):
        r2ext_np_recovery_sidebands["is_signal"] = r2ext_np_recovery_sidebands["category"] == 11
    
    r2_datasets = [r2lee, r2nue, r2mc, r2ext, r2data_two_showers_sidebands, r2data_np_far_sidebands, r2data_0p_far_sidebands]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        r2_datasets += [r2data_numu_sidebands]
    if (loadrecoveryvars == True):
        r2_datasets += [r2ext_np_recovery_sidebands]
        
    for r2_dataset in r2_datasets:
        r2_dataset['run1'] = np.zeros(len(r2_dataset), dtype=bool)
        r2_dataset['run2'] = np.ones(len(r2_dataset), dtype=bool)
        r2_dataset['run3'] = np.zeros(len(r2_dataset), dtype=bool)
        r2_dataset['run12'] = np.ones(len(r2_dataset), dtype=bool)
        if (loadnumucrtonly == True):
            #r2_dataset["_closestNuCosmicDist"] = np.zeros(len(r1_dataset),dtype=float)
            r2_dataset["crtveto"] = np.zeros(len(r2_dataset),dtype=int)
            r2_dataset["crthitpe"] = np.zeros(len(r2_dataset),dtype=float)
            r2_dataset["_closestNuCosmicDist"] = np.zeros(len(r2_dataset),dtype=float)

    r1dirt['run2'] = np.ones(len(r1dirt), dtype=bool)
    r3dirt['run2'] = np.ones(len(r3dirt), dtype=bool)
    if (loadtruthfilters == True):
        for r_dataset in [r1ncpi0, r1ccpi0, r3ncpi0, r3ccpi0,r1ccnopi, r1cccpi, r1ncnopi, r1nccpi, r3ccnopi, r3cccpi, r3ncnopi, r3nccpi]:
            r_dataset['run2'] = np.ones(len(r_dataset), dtype=bool)
    
    uproot_v = [ur2lee,ur2mc,ur2nue, ur2ext, ur2data_two_showers_sidebands, ur2data_np_far_sidebands, ur2data_0p_far_sidebands]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        uproot_v += [ur2data_numu_sidebands]
    if (loadrecoveryvars == True):
        uproot_v += [ur2ext_np_recovery_sidebands]

    df_v = [r2lee,r2mc,r2nue, r2ext, r2data_two_showers_sidebands, r2data_np_far_sidebands, r2data_0p_far_sidebands]
    #if (loadshowervariables == False):
    if ( (loadshowervariables == False) and (loadnumuntuples == True)):
        df_v += [r2data_numu_sidebands]
    if (loadrecoveryvars == True):
        df_v += [r2ext_np_recovery_sidebands]

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
    
    print("Concatenate dataframes")

    r1nue["pot_scale"] = 1
    r2nue["pot_scale"] = 1
    r3nue["pot_scale"] = 1
    
    r1lee["pot_scale"] = 1
    r2lee["pot_scale"] = 1
    r3lee["pot_scale"] = 1
    
    r1mc["pot_scale"] = 1
    r2mc["pot_scale"] = 1
    r3mc["pot_scale"] = 1
    
    r1ext["pot_scale"] = 1 
    r2ext["pot_scale"] = 1 
    r3ext["pot_scale"] = 1 

    nue = pd.concat([r1nue,r2nue,r3nue],ignore_index=True)
    #nue = pd.concat([r3nue,r1nue],ignore_index=True)
    mc = pd.concat([r3mc,r2mc,r1mc],ignore_index=True)
    #mc = pd.concat([r3mc,r1mc],ignore_index=True)
    if (loadtruthfilters == True):
        ncpi0 = pd.concat([r3ncpi0,r1ncpi0],ignore_index=True)
        ccpi0 = pd.concat([r3ccpi0,r1ccpi0],ignore_index=True,sort=True)
        ccnopi = pd.concat([r3ccnopi,r1ccnopi],ignore_index=True)
        cccpi = pd.concat([r3cccpi,r1cccpi],ignore_index=True)
        ncnopi = pd.concat([r3ncnopi,r1ncnopi],ignore_index=True)
        nccpi = pd.concat([r3nccpi,r1nccpi],ignore_index=True)
    # data = pd.concat([r3data,r1data],ignore_index=True)
    if which_sideband == '2plus_showers':
        data = pd.concat([r1data_two_showers_sidebands, r2data_two_showers_sidebands, r3data_two_showers_sidebands],ignore_index=True)
    elif which_sideband == 'np_far':
        data = pd.concat([r1data_np_far_sidebands, r2data_np_far_sidebands, r3data_np_far_sidebands],ignore_index=True)
    elif which_sideband == 'np_sb_comb':
        data = pd.concat([r1data_np_far_sidebands, r1data_two_showers_sidebands, r2data_np_far_sidebands, \
                          r2data_two_showers_sidebands, r3data_np_far_sidebands, r3data_two_showers_sidebands],\
                         ignore_index=True)
    elif which_sideband == '0p_far':
        data = pd.concat([r1data_0p_far_sidebands, r2data_0p_far_sidebands, r3data_0p_far_sidebands],ignore_index=True)
    elif which_sideband == 'numu':
        data = pd.concat([r1data_numu_sidebands, r2data_numu_sidebands, r3data_numu_sidebands],ignore_index=True)
    elif which_sideband == "opendata":
        data = pd.concat([r1data, r3data],ignore_index=True) # 5e19 and 1e19
        print ('opendata dataset has shape : ',data.shape)
    if (loadrecoveryvars == True):
        ext = pd.concat([r3ext, r3ext_np_recovery_sidebands, r2ext, r2ext_np_recovery_sidebands, \
                         r1ext, r1ext_np_recovery_sidebands],ignore_index=True, sort=True)
    else:
        ext = pd.concat([r3ext, r2ext, r1ext],ignore_index=True)
    dirt = pd.concat([r3dirt,r1dirt],ignore_index=True)
    lee = pd.concat([r1lee,r2lee,r3lee],ignore_index=True)
    #lee = pd.concat([r3lee,r1lee],ignore_index=True)
    
    print("Add derived variables")

    df_v_mc = [lee,mc,nue,dirt]
    if (loadtruthfilters == True):
        df_v_mc += [ccnopi,cccpi,ncnopi,nccpi,ncpi0,ccpi0]

    for i,df in enumerate(df_v_mc):

        df.loc[ df['weightTune'] <= 0, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] == np.inf, 'weightTune' ] = 1.
        df.loc[ df['weightTune'] > 100, 'weightTune' ] = 1.
        df.loc[ np.isnan(df['weightTune']) == True, 'weightTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
        df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
        df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
        # flux parentage
        df['flux'] = np.zeros_like(df['nslice'])
        #df.loc[ (((df['nu_pdg'] == 12) | (df['nu_pdg'] == -12)) & (df['nu_decay_mode'] < 11)) , 'flux'] = 10
        #df.loc[ (((df['nu_pdg'] == 12) | (df['nu_pdg'] == -12)) & (df['nu_decay_mode'] > 10)) , 'flux'] = 1
        # pi0 scaling
        if pi0scaling == 1:
            df.loc[ df['npi0'] > 0, 'weightSplineTimesTune' ] = df['weightSpline'] * df['weightTune'] * 0.759
        elif pi0scaling == 2:
            pi0emax = 0.6
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] < pi0emax) , 'weightSplineTimesTune'] = df['weightSplineTimesTune']*(1.-0.4*df['pi0_e'])
            df.loc[ (df['pi0_e'] > 0.1) & (df['pi0_e'] >= pi0emax), 'weightSplineTimesTune'] = df['weightSplineTimesTune']*(1.-0.4*pi0emax)

        if (loadeta==True):
            df['pi0_mass_truth'] = np.sqrt( 2 * df['pi0truth_gamma1_etot'] * df['pi0truth_gamma2_etot'] * (1 - df['pi0truth_gammadot']) )
            df.loc[ (df['pi0truth_gamma_parent']== 22) & (df['npi0'] == 0) , 'category' ] = 802
            df.loc[ (df['pi0truth_gamma_parent']== 22) & (df['npi0'] >  0) , 'category' ] = 801
            df.loc[ (df['category']== 31) & (df['npi0'] == 1), 'category' ] = 803
            df.loc[ (df['category']== 21) & (df['npi0'] == 1), 'category' ] = 803
            df.loc[ (df['category']== 31) & (df['npi0'] >  1), 'category' ] = 804
            df.loc[ (df['category']== 21) & (df['npi0'] >  1), 'category' ] = 804
            df.loc[ (df['category']== 4), 'category' ] = 806
            df.loc[ (df['category']== 5), 'category' ] = 806
            df.loc[ (df['category']== 1),   'category' ] = 805
            df.loc[ (df['category']== 10),  'category' ] = 805
            df.loc[ (df['category']== 11),  'category' ] = 805
            df.loc[ (df['category']== 111), 'category' ] = 805
            df.loc[ (df['category']== 2 ),  'category' ] = 805
            df.loc[ (df['category']== 3 ),  'category' ] = 805
        
    df_v = [lee,mc,nue,ext,data,dirt]
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
            

    if (loadpi0variables == True):
        for i,df in enumerate(df_v):
            df['asymm'] = np.abs(df['pi0_energy1_Y']-df['pi0_energy2_Y'])/(df['pi0_energy1_Y']+df['pi0_energy2_Y'])
            df['pi0energy'] = 134.98 * np.sqrt( 2. / ( (1-(df['asymm'])**2) * (1-df['pi0_gammadot']) ) )
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
            
    
        INTERCEPT = 0.0
        SLOPE = 0.83

        Me = 0.511e-3
        Mp = 0.938
        Mn = 0.940
        Eb = 0.0285

        # define some energy-related variables
        for i,df in enumerate(df_v):
            df["reco_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE + df["trk_energy_tot"]
            df['electron_e'] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE
            df['proton_e'] = Mp + df['protonenergy']
            df['proton_p'] = np.sqrt( (df['proton_e'])**2 - Mp**2 )
            df['reco_e_qe_l'] = ( df['electron_e'] * (Mn-Eb) + 0.5 * ( Mp**2 - (Mn - Eb)**2 - Me**2 ) ) / ( (Mn - Eb) - df['electron_e'] * (1 - np.cos(df['shr_theta'])) )
            df['reco_e_qe_p'] = ( df['proton_e']   * (Mn-Eb) + 0.5 * ( Me**2 - (Mn - Eb)**2 - Mp**2 ) ) / ( (Mn - Eb) + df['proton_p'] * np.cos(df['trk_theta']) - df['proton_e'] )
            df["reco_e_qe"] = 0.938*((df["shr_energy"]+INTERCEPT)/SLOPE)/(0.938 - ((df["shr_energy"]+INTERCEPT)/SLOPE)*(1-np.cos(df["shr_theta"])))
            df["reco_e_rqe"] = df["reco_e_qe"]/df["reco_e"]

    # and a way to filter out data
    for i,df in enumerate(df_v):
        df["bnbdata"] = np.zeros_like(df["nslice"])
        df["extdata"] = np.zeros_like(df["nslice"])
    data["bnbdata"] = np.ones_like(data["nslice"])
    ext["extdata"] = np.ones_like(ext["nslice"])

    # set EXT and DIRT contributions to 0 for fake-data studies
    if (loadfakedata > 0):
        dirt['nslice'] = np.zeros_like(dirt['nslice'])
        ext['nslice']  = np.zeros_like(ext['nslice'])
    
    # add back the cosmic category, for background only
    for i,df in enumerate(df_v):
        df.loc[(df['category']!=1)&(df['category']!=10)&(df['category']!=11)&(df['category']!=111)&(df['slnunhits']/df['slnhits']<0.2), 'category'] = 4
        if (loadeta == True):
            df.loc[ (df['category']== 4), 'category' ] = 806
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
    if (loadtruthfilters == True):
        ccnopi = ccnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        cccpi = cccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        ncnopi = ncnopi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')
        nccpi = nccpi.query('(nslice==0 | (slnunhits/slnhits)>0.1)')

    # avoid double-counting of events out of FV in the NC/CC pi0 samples
    # not needed anymore since we improved matching with filtered samples
    #ncpi0 = ncpi0.query('category != 5')
    #ccpi0 = ccpi0.query('category != 5')
    #ccnopi = ccnopi.query('category != 5')
    #nccpi = nccpi.query('category != 5')
    #ncnopi = ncnopi.query('category != 5')

    lee['flux'] = 111
                
    Npre = float(data.shape[0])
    if (loadfakedata == 0):
        data = data.drop_duplicates(subset=['run','evt'],keep='last') # keep last since the recovery samples are added at the end
    Npos = float(data.shape[0])
    print ('fraction of data surviving duplicate removal : %.02f'%(Npos/Npre))

    Npre = float(ext.shape[0])
    ext = ext.drop_duplicates(subset=['run','evt'],keep='last') # keep last since the recovery samples are added at the end
    Npos = float(ext.shape[0])
    print ('fraction of ext surviving duplicate removal : %.02f'%(Npos/Npre))

    samples = {
    "mc": mc,
    "nue": nue,
    "data": data,
    "ext": ext,
    "dirt": dirt,
    "lee": lee
    }
    if (loadtruthfilters == True):
        samples["ccnopi"] = ccnopi
        samples["cccpi"]  = cccpi
        samples["ncnopi"] = ncnopi
        samples["nccpi"]  = nccpi
        samples["ncpi0"]  = ncpi0
        samples["ccpi0"]  = ccpi0
    
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

        print ('number of data entries returned is : ',data.shape)
        print ('number of data entries returned is : ',samples['data'].shape)
        return samples
    
pot_data_unblinded = {
# v47 NTuples
    "farsideband" : { 
        1: (1.67E+20, 37094101),
        2: (2.62E+20, 62168648),
        3: (2.57E+20, 61381194),
        123: (6.86E+20, 160643943), },
# 0304 samples
#    "opendata" : {
#        1: (4.08E+19, 9028010),
#        2: (1.00E+01, 1),
#        3: (7.63E+18, 1838700), },
# 0628 samples
    "opendata" : {
        1: (4.54E+19, 10080350),
        2: (1.00E+01, 1),
        3: (9.43E+18, 2271036), },    
    "numu" : {
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
}

pot_mc_samples = {}

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

def get_weights(run,dataset="farsideband",scaling=1.0):
    assert run in [1, 2, 3, 123, 12, 30]
    weights_out = {}
    if run in [1, 2, 3]:
        pot_on, n_trig_on = pot_data_unblinded[dataset][run]
        for sample, pot in pot_mc_samples[run].items():
            if sample == 'ext':
                weights_out[sample] = n_trig_on/pot
            else:
                weights_out[sample] = pot_on/pot
        if run == 2:
            for sample in ['ncpi0', 'ccpi0', 'dirt', 'ncnopi', 'nccpi', 'ccnopi', 'cccpi']:
                weights_out[sample] = pot_on/(pot_mc_samples[1][sample] + pot_mc_samples[3][sample])
        pot_out = pot_on
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
        for sample, pot in pot_mc_samples[3].items():
            if sample == 'ext':
                weights_out[sample] = n_trig_on/pot
            else:
                weights_out[sample] = pot_on/pot
        pot_out = pot_on

    for key, val in weights_out.items():
        weights_out[key] *= scaling
        
    return weights_out, pot_out
                    
                
                    
            
