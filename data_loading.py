"""A unified module for loading data for different runs.

This is a refactorization of the code that was formerly in "load_data_run123.py".
"""

import hashlib
import logging
import os
import pickle
import numpy as np
import pandas as pd
import uproot
import yaml
import localSettings as ls
from typing import List
import numpy as np
import awkward as ak
import nue_booster
import xgboost as xgb
from typing import List, Tuple, Any, Union
from numpy.typing import NDArray
from numu_tki import selection_1muNp 
from numu_tki import signal_1muNp 
from numu_tki import tki_calculators 

from microfit.selections import extract_variables_from_query

datasets = ["bnb","opendata_bnb","bdt_sideband","shr_energy_sideband","two_shr_sideband","muon_sideband","near_sideband","far_sideband"]
detector_variations = ["cv","lydown","lyatt","lyrayleigh","sce","recomb2","wiremodx","wiremodyz","wiremodthetaxz","wiremodthetayz"]
verbose=True

# Set to true if trying to exactly reproduce old plots, otherwise, false
use_buggy_energy_estimator=False

def generate_hash(*args, **kwargs):
    hash_obj = hashlib.md5()
    # Even though the order should not matter in sets, it turns out that 
    # the conversion into the string does depend on the order in which items
    # were added to the set. To avoid this, we convert the sets into lists
    # and sort them before hashing.
    for key, value in kwargs.items():
        if isinstance(value, set):
            kwargs[key] = sorted(list(value))
    data = str(args) + str(kwargs)
    hash_obj.update(data.encode("utf-8"))
    return hash_obj.hexdigest()


def cache_dataframe(func):
    def wrapper(*args, enable_cache=False, cache_dir=ls.dataframe_cache_path, **kwargs):

        if not enable_cache:
            return func(*args, **kwargs)

        hash_value = generate_hash(*args, **kwargs)

        hdf_filepath = os.path.join(cache_dir, f"{hash_value}.h5")

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        if os.path.exists(hdf_filepath):
            df = pd.read_hdf(hdf_filepath, "data")
        else:
            df = func(*args, **kwargs)
            assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"
            df.to_hdf(hdf_filepath, key="data", mode="w")

        return df

    return wrapper

def get_variables():
    VARDICT = {}

    VARIABLES = [
        "nu_pdg",
        #############
        # The variables below are not floating point numbers, but vectors of variable length
        # that can only be stored in awkward arrays. These are very memory intensive, so we
        # do not want to load them into the final dataframe.
        # "mc_pdg",
        # "mc_px",
        # "mc_py",
        # "mc_pz",
        # "mc_E",
        #############
        "slpdg",
        # "backtracked_pdg",
        # "trk_score_v",
        "category",
        "ccnc",
        "endmuonmichel",
        # "NeutrinoEnergy0","NeutrinoEnergy1","NeutrinoEnergy2",
        "run",
        "sub",
        "evt",
        "CosmicIP",
        "CosmicDirAll3D",
        "CosmicIPAll3D",
        # "nu_flashmatch_score","best_cosmic_flashmatch_score","best_obviouscosmic_flashmatch_score",
        "flash_pe",
        # The TRK scroe is a rugged array and loading it directly into the Dataframe is very memory intensive
        # "trk_llr_pid_score_v",  # trk-PID score
        "_opfilter_pe_beam",
        "_opfilter_pe_veto",  # did the event pass the common optical filter (for MC only)
        "reco_nu_vtx_sce_x",
        "reco_nu_vtx_sce_y",
        "reco_nu_vtx_sce_z",
        "nproton",
        "nelec",
        "nu_e",
        # "hits_u", "hits_v", "hits_y",
        "nneutron",
        "slnunhits",
        "slnhits",
        "true_e_visible",
        "npi0",
        "npion",
        "pion_e",
        "muon_e",
        "pi0truth_elec_etot",
        "pi0_e",
        "evnunhits",
        "nslice",
        "interaction",
        "proton_e",
        "slclustfrac",
        "reco_nu_vtx_x",
        "reco_nu_vtx_y",
        "reco_nu_vtx_z",
        "true_nu_vtx_sce_x",
        "true_nu_vtx_sce_y",
        "true_nu_vtx_sce_z",
        "true_nu_vtx_x",
        "true_nu_vtx_y",
        "true_nu_vtx_z",
        # "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        # "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
        # "trk_start_x_v","trk_start_z_v","trk_start_z_v",
        "topological_score",
        "isVtxInFiducial",
        "theta",  # angle between incoming and outgoing leptons in radians
        # "nu_decay_mode","nu_hadron_pdg","nu_parent_pdg", # flux truth info
        # "shr_energy_tot_cali","selected","n_showers_contained",  # only if CC0piNp variables are saved!
        # We do not want to load "vector" variables into the final dataframe, as they take up 
        # a lot of memory
        # "pfp_generation_v",
        "shr_energy_cali",
        # "trk_dir_x_v",
        # "trk_dir_y_v",
        # "trk_dir_z_v",
    ]

    VARDICT["VARIABLES"] = VARIABLES

    CRTVARS = ["crtveto", "crthitpe", "_closestNuCosmicDist"]

    VARDICT["CRTVARS"] = CRTVARS

    WEIGHTS = [
        "weightSpline",
        "weightTune",
        "weightSplineTimesTune",
        "nu_decay_mode",
        "knobRPAup",
        "knobRPAdn",
        "knobCCMECup",
        "knobCCMECdn",
        "knobAxFFCCQEup",
        "knobAxFFCCQEdn",
        "knobVecFFCCQEup",
        "knobVecFFCCQEdn",
        "knobDecayAngMECup",
        "knobDecayAngMECdn",
        "knobThetaDelta2Npiup",
        "knobThetaDelta2Npidn",
    ]

    VARDICT["WEIGHTS"] = WEIGHTS

    # WEIGHTSLEE = ["weightSpline","weightTune","weightSplineTimesTune","nu_decay_mode","leeweight"]
    WEIGHTSLEE = WEIGHTS + ["leeweight"]

    VARDICT["WEIGHTSLEE"] = WEIGHTSLEE

    SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]

    VARDICT["SYSTVARS"] = SYSTVARS

    MCFVARS = [
        "mcf_nu_e",
        "mcf_lep_e",
        "mcf_actvol",
        "mcf_nmm",
        "mcf_nmp",
        "mcf_nem",
        "mcf_nep",
        "mcf_np0",
        "mcf_npp",
        "mcf_npm",
        "mcf_mcshr_elec_etot",
        "mcf_pass_ccpi0",
        "mcf_pass_ncpi0",
        "mcf_pass_ccnopi",
        "mcf_pass_ncnopi",
        "mcf_pass_cccpi",
        "mcf_pass_nccpi",
    ]

    VARDICT["MCFVARS"] = MCFVARS

    NUEVARS = [
        "shr_dedx_Y",
        "shr_bkt_pdg",
        "shr_theta",
        # "shr_pfp_id_v",  # this is an array
        "shr_tkfit_dedx_U",
        "shr_tkfit_dedx_V",
        "shr_tkfit_dedx_Y",
        "shr_tkfit_gap10_dedx_U",
        "shr_tkfit_gap10_dedx_V",
        "shr_tkfit_gap10_dedx_Y",
        "shr_tkfit_2cm_dedx_U",
        "shr_tkfit_2cm_dedx_V",
        "shr_tkfit_2cm_dedx_Y",
        "shrmoliereavg",
        "shrmoliererms",
        "shr_energy_tot_cali",
        "n_showers_contained",
        "selected",
        "shr_tkfit_npointsvalid",
        "shr_tkfit_npoints",  # fitted vs. all hits for shower
        "shrclusfrac0",
        "shrclusfrac1",
        "shrclusfrac2",  # track-fitted hits / all hits
        "trkshrhitdist2",
        "trkshrhitdist0",
        "trkshrhitdist1",  # distance between track and shower in 2D
        "shrsubclusters0",
        "shrsubclusters1",
        "shrsubclusters2",  # number of sub-clusters in shower
        "secondshower_U_nhit",
        "secondshower_U_vtxdist",
        "secondshower_U_dot",
        "secondshower_U_dir",
        "shrclusdir0",
        "secondshower_V_nhit",
        "secondshower_V_vtxdist",
        "secondshower_V_dot",
        "secondshower_V_dir",
        "shrclusdir1",
        "secondshower_Y_nhit",
        "secondshower_Y_vtxdist",
        "secondshower_Y_dot",
        "secondshower_Y_dir",
        "shrclusdir2",
        "shrMCSMom",
        "DeltaRMS2h",
        "shrPCA1CMed_5cm",
        "CylFrac2h_1cm",
        "shr_hits_tot",
        "shr_hits_u_tot",
        "shr_hits_v_tot",
        "shr_hits_y_tot",
        # "shr_theta_v",
        # "shr_phi_v",
        # "shr_energy_y_v",
        # "shr_start_x_v",
        # "shr_start_z_v",
        # "shr_start_z_v",
        "trk_bkt_pdg",
        "shr_tkfit_dedx_U",
        "shr_tkfit_dedx_V",
        "trk_bkt_pdg",
        "shr_energy",
        "shr_dedx_U",
        "shr_dedx_V",
        "shr_phi",
        "trk_phi",
        "trk_theta",
        "shr_distance",
        "trk_distance",
        "matched_E",
        "shr_bkt_E",
        "trk_bkt_E",
        "shr_tkfit_nhits_Y",
        "shr_tkfit_nhits_U",
        "shr_tkfit_nhits_V",
        "shr_tkfit_2cm_nhits_Y",
        "shr_tkfit_2cm_nhits_U",
        "shr_tkfit_2cm_nhits_V",
        "shr_tkfit_gap10_nhits_Y",
        "shr_tkfit_gap10_nhits_U",
        "shr_tkfit_gap10_nhits_V",
        "trk_energy",
        "tksh_distance",
        "tksh_angle",
        "contained_fraction",
        "shr_score",
        "trk_score",
        "trk_hits_tot",
        "trk_len",
        "trk_hits_tot",
        "trk_hits_u_tot",
        "trk_hits_v_tot",
        "trk_hits_y_tot",
        "shr_dedx_Y_cali",
        "trk_energy_tot",
        "shr_id",
        "hits_ratio",
        "n_tracks_contained",
        "shr_px",
        "shr_py",
        "shr_pz",
        "p",
        "pt",
        "hits_y",
        "shr_start_x",
        "shr_start_x",
        "shr_start_x",
        "elec_pz",
        "elec_e",
        "truthFiducial",
        "pi0truth_gamma1_edep",
        "shr_bkt_E",
        "pi0truth_gamma1_etot",
        "pi0truth_gamma1_zpos",
        "shr_start_z",
        "pi0truth_gamma1_ypos",
        "shr_start_y",
        "pi0truth_gamma1_xpos",
        "shr_start_x",
    ]

    VARDICT["NUEVARS"] = NUEVARS

    NUMUVARS = []

    VARDICT["NUMUVARS"] = []

    RCVRYVARS = [
        "shr_energy_tot",
        "trk_energy_tot",
        # NOTE: Loading rugged vector columns like these directly into the dataframe is very memory intensive
        # and tanks performance. We should avoid doing this. If these variables are needed for intermediate
        # calculations, please load them from the root file manually in the processing function.
        # "trk_end_x_v",
        # "trk_end_y_v",
        # "trk_end_z_v",
        # "trk_phi_v",
        # "trk_theta_v",
        # "trk_len_v",
        "trk_id",
        "shr_start_x",
        "shr_start_y",
        "shr_start_z",
        "trk_hits_max",
        # "shr_tkfit_dedx_u_v",
        # "shr_tkfit_dedx_v_v",
        # "shr_tkfit_dedx_y_v",
        # "shr_tkfit_dedx_nhits_u_v",
        # "shr_tkfit_dedx_nhits_v_v",
        # "shr_tkfit_dedx_nhits_y_v",
        "trk2shrhitdist2",
        "trk1trk2hitdist2",
        "shr1shr2moliereavg",
        "shr1trk1moliereavg",
        "shr1trk2moliereavg",
        "trk2_id",
        "shr2_id",
        "trk_hits_2nd",
        "shr_hits_2nd",
    ]

    VARDICT["RCVRYVARS"] = RCVRYVARS

    PI0VARS = [
        "pi0_radlen1",
        "pi0_radlen2",
        "pi0_dot1",
        "pi0_dot2",
        "pi0_energy1_Y",
        "pi0_energy2_Y",
        "pi0_dedx1_fit_Y",
        "pi0_dedx2_fit_Y",
        "pi0_shrscore1",
        "pi0_shrscore2",
        "pi0_gammadot",
        "pi0_dedx1_fit_V",
        "pi0_dedx2_fit_V",
        "pi0_dedx1_fit_U",
        "pi0_dedx2_fit_U",
        "pi0_mass_Y",
        "pi0_mass_V",
        "pi0_mass_U",
        "pi0_nshower",
        "pi0_dir2_x",
        "pi0_dir2_y",
        "pi0_dir2_z",
        "pi0_dir1_x",
        "pi0_dir1_y",
        "pi0_dir1_z",
        "pi0truth_gamma1_etot",
        "pi0truth_gamma2_etot",
        "pi0truth_gammadot",
        "pi0truth_gamma_parent",
        "pi0truth_gamma1_dist",
        "pi0truth_gamma1_edep",
        "pi0truth_gamma2_dist",
        "pi0truth_gamma2_edep",
        "true_nu_vtx_x",
        "true_nu_vtx_y",
        "true_nu_vtx_z",  # ,"n_showers_contained"
    ]

    VARDICT["PI0VARS"] = PI0VARS

    R3VARS = []

    VARDICT["R3VARS"] = R3VARS

    return VARDICT


def add_paper_category(df, key):
    df.loc[:, "paper_category"] = df["category"]
    if key in ["data", "nu"]:
        return
    df.loc[df["paper_category"].isin([1, 10]), "paper_category"] = 11
    if key is "nue":
        df.loc[df["category"].isin([4, 5]) & (df["ccnc"] == 0), "paper_category"] = 11
        df.loc[df["category"].isin([4, 5]) & (df["ccnc"] == 1), "paper_category"] = 2
        df.loc[(df["paper_category"] == 3), "paper_category"] = 2
        return
    if key is "lee":
        df.loc[df["category"].isin([4, 5]), "paper_category"] = 111
        df.loc[(df["paper_category"] == 3), "paper_category"] = 2
        return
    if key is "dirt":
        df["paper_category"] = 2
        return
    df.loc[(df["npi0"] > 0), "paper_category"] = 31
    df.loc[(df["npi0"] == 0), "paper_category"] = 2


def add_paper_category_1e1p(df, key):
    df.loc[:, "category_1e1p"] = df[
        "category"
    ]  # makes a new column called 'category_1e1p' in df which copies the 'category'
    if key in ["data"]:
        return
    df.loc[(df["nproton"] == 1), "category_1e1p"] = 12
    df.loc[(df["nproton"] > 1), "category_1e1p"] = 13


def add_paper_xsec_category(df, key):
    df.loc[:, "paper_category_xsec"] = df["category"]
    if key in ["data", "nu"]:
        return
    df.loc[(df["npi0"] > 0), "paper_category_xsec"] = 31
    df.loc[(df["npi0"] == 0), "paper_category_xsec"] = 2
    if key is "nue":
        df.loc[df["category"].isin([4, 5]) & (df["ccnc"] == 1), "paper_category_xsec"] = 2
        df.loc[(df["category"] == 3), "paper_category_xsec"] = 2
        df.loc[
            df["category"].isin([4, 5]) & (df["ccnc"] == 0) & ((df["npi0"] > 0) | (df["npion"] > 0)),
            "paper_category_xsec",
        ] = 1
        df.loc[
            df["category"].isin([4, 5])
            & (df["ccnc"] == 0)
            & (df["npi0"] == 0)
            & (df["npion"] == 0)
            & (df["nproton"] == 0),
            "paper_category_xsec",
        ] = 10
        df.loc[
            df["category"].isin([4, 5])
            & (df["ccnc"] == 0)
            & (df["npi0"] == 0)
            & (df["npion"] == 0)
            & (df["nproton"] > 0),
            "paper_category_xsec",
        ] = 11
        return
    if key is "dirt":
        df["paper_category_xsec"] = 2
        return


def add_paper_numu_category(df, key):
    df.loc[:, "paper_category_numu"] = 0
    if key in ["data", "nu"]:
        return
    df.loc[(df["ccnc"] == 0), "paper_category_numu"] = 2
    df.loc[(df["ccnc"] == 1), "paper_category_numu"] = 3
    if key is "nue":
        df.loc[(df["ccnc"] == 0), "paper_category_numu"] = 11
        return
    if key is "lee":
        df.loc[(df["ccnc"] == 0), "paper_category_numu"] = 111
        return
    if key is "dirt":
        df["paper_category"] = 5
        df["paper_category_numu"] = 5
        return


def add_paper_categories(df, key):
    add_paper_category(df, key)
    add_paper_xsec_category(df, key)
    add_paper_numu_category(df, key)
    add_paper_category_1e1p(df, key)


def load_data_run(
    run_number,
    which_sideband="pi0",
    return_plotter=True,
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
):
    raise NotImplementedError("Full data run loading not implemented yet")

def get_elm_from_vec_idx(
    myvec: ak.JaggedArray, idx: Union[List[int], NDArray[Any]], fillval=np.nan
) -> NDArray[np.float64]:
    """Returns the element of a vector at position idx, where idx is a vector of indices. If idx is out of bounds, returns a filler value"""

    return np.array([pidv[tid] if ((tid < len(pidv)) & (tid >= 0)) else fillval for pidv, tid in zip(myvec, idx)])


def get_idx_from_vec_sort(argidx: int, vecsort: NDArray[np.float64], mask: NDArray[np.bool_]) -> List[int]:
    """Returns the index of the element of a vector at position argidx, where argidx is a vector of indices. If argidx is out of bounds, returns -1."""
    vid = vecsort[mask]
    sizecheck = argidx if argidx >= 0 else abs(argidx) - 1
    # find the position in the array after masking
    mskd_pos = [v.argsort()[argidx] if len(v) > sizecheck else -1 for v in vid]
    # go back to the corresponding position in the origin array before masking
    result = [[i for i, n in enumerate(m) if n == 1][p] if (p) >= 0 else -1 for m, p in zip(mask, mskd_pos)]
    return result


def distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    Calculates the Euclidean distance between two points in 3D space.

    Args:
    x1 (float): x-coordinate of the first point.
    y1 (float): y-coordinate of the first point.
    z1 (float): z-coordinate of the first point.
    x2 (float): x-coordinate of the second point.
    y2 (float): y-coordinate of the second point.
    z2 (float): z-coordinate of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def cos_angle_two_vecs(vx1: float, vy1: float, vz1: float, vx2: float, vy2: float, vz2: float) -> float:
    """
    Calculates the cosine of the angle between two vectors in 3D space.

    Args:
    vx1 (float): x-component of the first vector.
    vy1 (float): y-component of the first vector.
    vz1 (float): z-component of the first vector.
    vx2 (float): x-component of the second vector.
    vy2 (float): y-component of the second vector.
    vz2 (float): z-component of the second vector.

    Returns:
    float: The cosine of the angle between the two vectors.
    """
    numerator = vx1 * vx2 + vy1 * vy2 + vz1 * vz2
    denominator = np.sqrt(vx1**2 + vy1**2 + vz1**2) * np.sqrt(vx2**2 + vy2**2 + vz2**2)
    return numerator / denominator


def mgg(e1: float, e2: float, px1: float, px2: float, py1: float, py2: float, pz1: float, pz2: float) -> float:
    """
    Calculates the invariant mass of two particles using their energies and momenta.

    Args:
        e1 (float): Energy of the first particle.
        e2 (float): Energy of the second particle.
        px1 (float): x-component of the momentum of the first particle.
        px2 (float): x-component of the momentum of the second particle.
        py1 (float): y-component of the momentum of the first particle.
        py2 (float): y-component of the momentum of the second particle.
        pz1 (float): z-component of the momentum of the first particle.
        pz2 (float): z-component of the momentum of the second particle.

    Returns:
        float: The invariant mass of the two particles.
    """
    return np.sqrt(2 * e1 * e2 * (1 - px1 * px2 - py1 * py2 - pz1 * pz2))


def combs(args):
    """Returns all pairs from the input list.

    Args:
        args: List of objects.

    Returns:
        List of tuples containing all pairs from the input list.
    """
    res = []
    for i in range(0, len(args)):
        for j in range(i + 1, len(args)):
            res.append((args[i], args[j]))
    return res


def all_comb_mgg(
    ev: np.ndarray, pxv: np.ndarray, pyv: np.ndarray, pzv: np.ndarray, combs: List[Tuple[int, int]]
) -> List[float]:
    """Compute the invariant mass of all pairs of electrons in the event

    Args:
        ev (np.array): Array of event numbers
        pxv (np.array): Array of electron px
        pyv (np.array): Array of electron py
        pzv (np.array): Array of electron pz
        combs (list): List of index pairs

    Returns:
        list: List of invariant masses
    """
    res = []
    for i, j in combs:
        res.append(np.nan_to_num(mgg(ev[i], ev[j], pxv[i], pxv[j], pyv[i], pyv[j], pzv[i], pzv[j])))
    return res


def sum_elements_from_mask(vector: np.ndarray, mask: np.ndarray) -> float:
    """
    Sums the elements of a numpy array that correspond to a given mask.

    Args:
        vector (np.ndarray): The input numpy array.
        mask (np.ndarray): The mask to apply to the input array.

    Returns:
        float: The sum of the elements of the input array that correspond to the mask.
    """
    vid = vector[mask]
    result = vid.sum()
    return result


def unique_combs(combs: List[Tuple[int, int]], combs_argsort: List[int]) -> List[int]:
    """
    Given a list of index pairs and a list of sorted indices, returns a list of unique index pairs.

    Args:
        combs (List[Tuple[int, int]]): List of index pairs
        combs_argsort (List[int]): List of sorted indices

    Returns:
        List[int]: List of unique index pairs
    """
    res = []
    usedargs = []
    for arg in combs_argsort:
        i, j = combs[arg]
        if i in usedargs or j in usedargs:
            continue
        usedargs.append(i)
        usedargs.append(j)
        res.append(arg)
    return res


def process_uproot_shower_variables(up, df):
    """Add shower variables to the dataframe using the ROOT tree."""

    # this subtraction of one is needed because an ID number of 1 corresponds to the first element 
    # in an associated vector variable, but in python this is denoted by 0, hence the -1 requirement.
    trk_id = up.array("trk_id") - 1
    shr_id = up.array("shr_id") - 1

    trk_llr_pid_v = up.array("trk_llr_pid_score_v")
    trk_calo_energy_y_v = up.array("trk_calo_energy_y_v")
    trk_energy_proton_v = up.array("trk_energy_proton_v")

    trk_llr_pid_v_sel = get_elm_from_vec_idx(trk_llr_pid_v, trk_id)
    trk_calo_energy_y_sel = get_elm_from_vec_idx(trk_calo_energy_y_v, trk_id)
    trk_energy_proton_sel = get_elm_from_vec_idx(trk_energy_proton_v, trk_id)
    df["trkpid"] = trk_llr_pid_v_sel
    df["trackcaloenergy"] = trk_calo_energy_y_sel
    df["protonenergy"] = trk_energy_proton_sel
    trk_sce_start_x_v = up.array("trk_sce_start_x_v")
    trk_sce_start_y_v = up.array("trk_sce_start_y_v")
    trk_sce_start_z_v = up.array("trk_sce_start_z_v")
    trk_sce_end_x_v = up.array("trk_sce_end_x_v")
    trk_sce_end_y_v = up.array("trk_sce_end_y_v")
    trk_sce_end_z_v = up.array("trk_sce_end_z_v")
    df["shr_trk_sce_start_x"] = get_elm_from_vec_idx(trk_sce_start_x_v, shr_id)
    df["shr_trk_sce_start_y"] = get_elm_from_vec_idx(trk_sce_start_y_v, shr_id)
    df["shr_trk_sce_start_z"] = get_elm_from_vec_idx(trk_sce_start_z_v, shr_id)
    df["shr_trk_sce_end_x"] = get_elm_from_vec_idx(trk_sce_end_x_v, shr_id)
    df["shr_trk_sce_end_y"] = get_elm_from_vec_idx(trk_sce_end_y_v, shr_id)
    df["shr_trk_sce_end_z"] = get_elm_from_vec_idx(trk_sce_end_z_v, shr_id)
    df["shr_trk_len"] = distance(
        df["shr_trk_sce_start_x"],
        df["shr_trk_sce_start_y"],
        df["shr_trk_sce_start_z"],
        df["shr_trk_sce_end_x"],
        df["shr_trk_sce_end_y"],
        df["shr_trk_sce_end_z"],
    )
    df["mevcm"] = 1000 * df["shr_energy_tot_cali"] / df["shr_trk_len"]
    #
    df["slclnhits"] = up.array("pfnhits").sum()
    df["slclnunhits"] = up.array("pfnunhits").sum()
    #
    pfp_pdg_v = up.array("backtracked_pdg")
    trk_pdg = get_elm_from_vec_idx(pfp_pdg_v, trk_id)
    df["trk_pdg"] = trk_pdg
    pfp_pur_v = up.array("backtracked_purity")
    trk_pur = get_elm_from_vec_idx(pfp_pur_v, trk_id)
    df["trk_pur"] = trk_pur
    pfp_cmp_v = up.array("backtracked_completeness")
    trk_cmp = get_elm_from_vec_idx(pfp_cmp_v, trk_id)
    df["trk_cmp"] = trk_cmp
    #
    # fix elec_pz for positrons
    nu_pdg = up.array("nu_pdg")
    ccnc = up.array("ccnc")
    mc_pdg = up.array("mc_pdg")
    mc_E = up.array("mc_E")
    mc_px = up.array("mc_px")
    mc_py = up.array("mc_py")
    mc_pz = up.array("mc_pz")
    elec_pz = up.array("elec_pz")
    positr_mask = mc_pdg == -11
    mostEpositrIdx = get_idx_from_vec_sort(-1, mc_E, positr_mask)
    mc_E_posi = get_elm_from_vec_idx(mc_E, mostEpositrIdx)
    mc_px_posi = get_elm_from_vec_idx(mc_px, mostEpositrIdx)
    mc_py_posi = get_elm_from_vec_idx(mc_py, mostEpositrIdx)
    mc_pz_posi = get_elm_from_vec_idx(mc_pz, mostEpositrIdx)
    mc_p_posi = np.sqrt(mc_px_posi * mc_px_posi + mc_py_posi * mc_py_posi + mc_pz_posi * mc_pz_posi)
    df["positr_px"] = np.where((mc_E_posi > 0), mc_px_posi / mc_p_posi, np.nan)
    df["positr_py"] = np.where((mc_E_posi > 0), mc_py_posi / mc_p_posi, np.nan)
    df["positr_pz"] = np.where((mc_E_posi > 0), mc_pz_posi / mc_p_posi, np.nan)
    df.loc[(nu_pdg == -12) & (ccnc == 0) & (elec_pz < -2), "elec_px"] = df["positr_px"]
    df.loc[(nu_pdg == -12) & (ccnc == 0) & (elec_pz < -2), "elec_py"] = df["positr_py"]
    df.loc[(nu_pdg == -12) & (ccnc == 0) & (elec_pz < -2), "elec_pz"] = df["positr_pz"]
    #
    # get true proton angle
    prot_mask = mc_pdg == 2212
    mostEprotIdx = get_idx_from_vec_sort(-1, mc_E, prot_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E, mostEprotIdx)
    mc_px_prot = get_elm_from_vec_idx(mc_px, mostEprotIdx)
    mc_py_prot = get_elm_from_vec_idx(mc_py, mostEprotIdx)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz, mostEprotIdx)
    mc_p_prot = np.sqrt(mc_px_prot * mc_px_prot + mc_py_prot * mc_py_prot + mc_pz_prot * mc_pz_prot)
    df["proton_pz"] = np.where((mc_E_prot > 0), mc_pz_prot / mc_p_prot, np.nan)
    #
    # true proton length (assuming straight line)
    mc_vx = up.array("mc_vx")
    mc_vy = up.array("mc_vy")
    mc_vz = up.array("mc_vz")
    mc_endx = up.array("mc_endx")
    mc_endy = up.array("mc_endy")
    mc_endz = up.array("mc_endz")
    p_vx = get_elm_from_vec_idx(mc_vx, mostEprotIdx)
    p_vy = get_elm_from_vec_idx(mc_vy, mostEprotIdx)
    p_vz = get_elm_from_vec_idx(mc_vz, mostEprotIdx)
    p_endx = get_elm_from_vec_idx(mc_endx, mostEprotIdx)
    p_endy = get_elm_from_vec_idx(mc_endy, mostEprotIdx)
    p_endz = get_elm_from_vec_idx(mc_endz, mostEprotIdx)
    df["proton_len"] = np.sqrt(
        (p_endx - p_vx) * (p_endx - p_vx) + (p_endy - p_vy) * (p_endy - p_vy) + (p_endz - p_vz) * (p_endz - p_vz)
    )
    #
    trk_score_v = up.array("trk_score_v")
    shr_mask = trk_score_v < 0.5
    trk_mask = trk_score_v > 0.5
    df["n_tracks_tot"] = trk_mask.sum()
    df["n_showers_tot"] = shr_mask.sum()
    trk_len_v = up.array("trk_len_v")
    df["n_trks_gt10cm"] = (trk_len_v[trk_mask >= 0.5] > 10).sum()
    df["n_trks_gt25cm"] = (trk_len_v[trk_mask >= 0.5] > 25).sum()
    trk_distance_v = up.array("trk_distance_v")
    df["n_tracks_attach"] = (trk_distance_v[trk_mask >= 0.5] < 3).sum()
    df["n_protons_attach"] = ((trk_distance_v[trk_mask >= 0.5] < 3) & (trk_llr_pid_v[trk_mask >= 0.5] < 0.02)).sum()
    #
    pfnhits_v = up.array("pfnhits")
    trk_id_all = get_idx_from_vec_sort(-1, pfnhits_v, trk_mask)  # this includes also uncontained tracks
    #
    shr_start_x_v = up.array("shr_start_x_v")
    shr_start_y_v = up.array("shr_start_y_v")
    shr_start_z_v = up.array("shr_start_z_v")
    df["shr_start_x"] = get_elm_from_vec_idx(shr_start_x_v, shr_id)
    df["shr_start_y"] = get_elm_from_vec_idx(shr_start_y_v, shr_id)
    df["shr_start_z"] = get_elm_from_vec_idx(shr_start_z_v, shr_id)
    trk_start_x_v = up.array("trk_start_x_v")
    trk_start_y_v = up.array("trk_start_y_v")
    trk_start_z_v = up.array("trk_start_z_v")
    df["trk1_start_x_alltk"] = get_elm_from_vec_idx(trk_start_x_v, trk_id_all)
    df["trk1_start_y_alltk"] = get_elm_from_vec_idx(trk_start_y_v, trk_id_all)
    df["trk1_start_z_alltk"] = get_elm_from_vec_idx(trk_start_z_v, trk_id_all)
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    df["trk1_dir_x_alltk"] = get_elm_from_vec_idx(trk_dir_x_v, trk_id_all)
    df["trk1_dir_y_alltk"] = get_elm_from_vec_idx(trk_dir_y_v, trk_id_all)
    df["trk1_dir_z_alltk"] = get_elm_from_vec_idx(trk_dir_z_v, trk_id_all)
    #
    # tksh_distance and tksh_angle for track with most hits, regardless of containment
    #
    if np.shape(df["shr_start_x"]) != np.shape(df["trk1_start_x_alltk"]):
        return
    df["tk1sh1_distance_alltk"] = np.where(
        df["n_tracks_tot"] == 0,
        np.nan,
        distance(
            df["shr_start_x"],
            df["shr_start_y"],
            df["shr_start_z"],
            df["trk1_start_x_alltk"],
            df["trk1_start_y_alltk"],
            df["trk1_start_z_alltk"],
        ),
    )
    df["tk1sh1_angle_alltk"] = np.where(
        df["n_tracks_tot"] == 0,
        np.nan,
        cos_angle_two_vecs(
            df["trk1_dir_x_alltk"],
            df["trk1_dir_y_alltk"],
            df["trk1_dir_z_alltk"],
            df["shr_px"],
            df["shr_py"],
            df["shr_pz"],
        ),
    )
    #
    df["shr_ptot"] = np.sqrt(df["shr_px"] ** 2 + df["shr_py"] ** 2 + df["shr_pz"] ** 2)
    df["shr_px_unit"] = df["shr_px"] / df["shr_ptot"]
    df["shr_py_unit"] = df["shr_py"] / df["shr_ptot"]
    df["shr_pz_unit"] = df["shr_pz"] / df["shr_ptot"]

    # return # DAVIDC

    #
    # fix the 'subcluster' bug (in case of more than one shower, it comes from the one with least hits, not the one with most)
    # so we overwrite the dataframe column taking the correct value from the corrsponding vector branches
    #
    pfpplanesubclusters_U_v = up.array("pfpplanesubclusters_U")
    pfpplanesubclusters_V_v = up.array("pfpplanesubclusters_V")
    pfpplanesubclusters_Y_v = up.array("pfpplanesubclusters_Y")
    df["shrsubclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v, shr_id, 0)
    df["shrsubclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v, shr_id, 0)
    df["shrsubclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v, shr_id, 0)
    #
    # do the best we can to get the right shr2_id
    #
    shr2_id_corr = up.array("shr2_id") - 1  # I think we need this -1 to get the right result
    shr2_id_appr = get_idx_from_vec_sort(-2, pfnhits_v, shr_mask)
    shr2_id = np.where((shr2_id_corr >= 0) & (shr2_id_corr < df["n_showers_tot"]), shr2_id_corr, shr2_id_appr)
    #
    df["shr2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v, shr2_id, 0)
    df["shr2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v, shr2_id, 0)
    df["shr2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v, shr2_id, 0)
    df["subcluster2tmp"] = df["shr2subclusters0"] + df["shr2subclusters1"] + df["shr2subclusters2"]
    #
    df["shr2_start_x"] = get_elm_from_vec_idx(shr_start_x_v, shr2_id)
    df["shr2_start_y"] = get_elm_from_vec_idx(shr_start_y_v, shr2_id)
    df["shr2_start_z"] = get_elm_from_vec_idx(shr_start_z_v, shr2_id)
    df["trk1_start_x"] = get_elm_from_vec_idx(trk_start_x_v, trk_id)
    df["trk1_start_y"] = get_elm_from_vec_idx(trk_start_y_v, trk_id)
    df["trk1_start_z"] = get_elm_from_vec_idx(trk_start_z_v, trk_id)
    df["tk1sh2_distance"] = np.where(
        (df["n_showers_contained"] > 1) & (df["n_tracks_contained"] > 0),
        distance(
            df["shr2_start_x"],
            df["shr2_start_y"],
            df["shr2_start_z"],
            df["trk1_start_x"],
            df["trk1_start_y"],
            df["trk1_start_z"],
        ),
        np.nan,
    )

    df["shr2pid"] = get_elm_from_vec_idx(trk_llr_pid_v, shr2_id)
    df["shr2_score"] = get_elm_from_vec_idx(trk_score_v, shr2_id)
    shr_moliere_avg_v = up.array("shr_moliere_avg_v")
    df["shr2_moliereavg"] = get_elm_from_vec_idx(shr_moliere_avg_v, shr2_id)

    return


def post_process_shower_vars(up, df):
    
    # These shower variables depend on variables that may be subject to recovery

    df["dx_s"] = df["shr_start_x"] - df["true_nu_vtx_sce_x"]
    df["dy_s"] = df["shr_start_y"] - df["true_nu_vtx_sce_y"]
    df["dz_s"] = df["shr_start_z"] - df["true_nu_vtx_sce_z"]
    df["dr_s"] = np.sqrt(df["dx_s"] * df["dx_s"] + df["dy_s"] * df["dy_s"] + df["dz_s"] * df["dz_s"])

    df["ptOverP"] = df["pt"] / df["p"]
    df["phi1MinusPhi2"] = df["shr_phi"] - df["trk_phi"]
    df["theta1PlusTheta2"] = df["shr_theta"] + df["trk_theta"]
    df["cos_shr_theta"] = np.cos(df["shr_theta"])
    df["cos_trk_theta"] = np.cos(df["trk_theta"])
    df.loc[df["n_tracks_tot"] == 0, "trk_theta"] = -9999
    df.loc[df["n_tracks_tot"] == 0, "cos_trk_theta"] = -9999

    df["showergammadist"] = np.sqrt(
        (df["pi0truth_gamma1_zpos"] - df["shr_start_z"]) ** 2
        + (df["pi0truth_gamma1_ypos"] - df["shr_start_y"]) ** 2
        + (df["pi0truth_gamma1_xpos"] - df["shr_start_x"] + 1.08) ** 2
    )
    df["bktrgammaenergydiff"] = np.abs(df["shr_bkt_E"] * 1000 - df["pi0truth_gamma1_etot"])
    df["subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    #
    df["trkfit"] = df["shr_tkfit_npointsvalid"] / df["shr_tkfit_npoints"]
    # and the 2d angle difference
    df["anglediff_Y"] = np.abs(df["secondshower_Y_dir"] - df["shrclusdir2"])
    df["anglediff_V"] = np.abs(df["secondshower_V_dir"] - df["shrclusdir1"])
    df["anglediff_U"] = np.abs(df["secondshower_U_dir"] - df["shrclusdir0"])

    df["shr_tkfit_nhits_tot"] = df["shr_tkfit_nhits_Y"] + df["shr_tkfit_nhits_U"] + df["shr_tkfit_nhits_V"]
    # df['shr_tkfit_dedx_avg'] = (df['shr_tkfit_nhits_Y']*df['shr_tkfit_dedx_Y'] + df['shr_tkfit_nhits_U']*df['shr_tkfit_dedx_U'] + df['shr_tkfit_nhits_V']*df['shr_tkfit_dedx_V'])/df['shr_tkfit_nhits_tot']
    df["shr_tkfit_2cm_nhits_tot"] = (
        df["shr_tkfit_2cm_nhits_Y"] + df["shr_tkfit_2cm_nhits_U"] + df["shr_tkfit_2cm_nhits_V"]
    )
    # df['shr_tkfit_2cm_dedx_avg'] = (df['shr_tkfit_2cm_nhits_Y']*df['shr_tkfit_2cm_dedx_Y'] + df['shr_tkfit_2cm_nhits_U']*df['shr_tkfit_2cm_dedx_U'] + df['shr_tkfit_2cm_nhits_V']*df['shr_tkfit_2cm_dedx_V'])/df['shr_tkfit_2cm_nhits_tot']
    df["shr_tkfit_gap10_nhits_tot"] = (
        df["shr_tkfit_gap10_nhits_Y"] + df["shr_tkfit_gap10_nhits_U"] + df["shr_tkfit_gap10_nhits_V"]
    )
    # df['shr_tkfit_gap10_dedx_avg'] = (df['shr_tkfit_gap10_nhits_Y']*df['shr_tkfit_gap10_dedx_Y'] + df['shr_tkfit_gap10_nhits_U']*df['shr_tkfit_gap10_dedx_U'] + df['shr_tkfit_gap10_nhits_V']*df['shr_tkfit_gap10_dedx_V'])/df['shr_tkfit_gap10_nhits_tot']
    df.loc[:, "shr_tkfit_dedx_max"] = df["shr_tkfit_dedx_Y"]
    df.loc[(df["shr_tkfit_nhits_U"] > df["shr_tkfit_nhits_Y"]), "shr_tkfit_dedx_max"] = df["shr_tkfit_dedx_U"]
    df.loc[
        (df["shr_tkfit_nhits_V"] > df["shr_tkfit_nhits_Y"]) & (df["shr_tkfit_nhits_V"] > df["shr_tkfit_nhits_U"]),
        "shr_tkfit_dedx_max",
    ] = df["shr_tkfit_dedx_V"]

    df.loc[:, "shr_tkfit_2cm_dedx_max"] = df["shr_tkfit_2cm_dedx_Y"]
    df.loc[(df["shr_tkfit_2cm_nhits_U"] > df["shr_tkfit_2cm_nhits_Y"]), "shr_tkfit_2cm_dedx_max"] = df[
        "shr_tkfit_2cm_dedx_U"
    ]
    df.loc[
        (df["shr_tkfit_2cm_nhits_V"] > df["shr_tkfit_2cm_nhits_Y"])
        & (df["shr_tkfit_2cm_nhits_V"] > df["shr_tkfit_2cm_nhits_U"]),
        "shr_tkfit_2cm_dedx_max",
    ] = df["shr_tkfit_2cm_dedx_V"]

    df.loc[:, "shr_tkfit_gap10_dedx_max"] = df["shr_tkfit_gap10_dedx_Y"]
    df.loc[(df["shr_tkfit_gap10_nhits_U"] > df["shr_tkfit_gap10_nhits_Y"]), "shr_tkfit_gap10_dedx_max"] = df[
        "shr_tkfit_gap10_dedx_U"
    ]
    df.loc[
        (df["shr_tkfit_gap10_nhits_V"] > df["shr_tkfit_gap10_nhits_Y"])
        & (df["shr_tkfit_gap10_nhits_V"] > df["shr_tkfit_gap10_nhits_U"]),
        "shr_tkfit_gap10_dedx_max",
    ] = df["shr_tkfit_gap10_dedx_V"]

    # Some energy-related variables
    INTERCEPT = 0.0
    SLOPE = 0.83

    Me = 0.511e-3
    Mp = 0.938
    Mn = 0.940
    Eb = 0.0285

    df["reco_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE + df["trk_energy_tot"]
   

    df["reco_e_overflow"] = df["reco_e"]
    df.loc[(df["reco_e"] >= 2.25), "reco_e_overflow"] = 2.24
    df["reco_e_mev"] = df["reco_e"] * 1000.0
    df["reco_e_mev_overflow"] = df["reco_e_overflow"] * 1000.0
    df["electron_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE
    df["proton_ke"] = df["proton_e"] - Mp
    df.loc[(df["proton_ke"] < 0), "proton_ke"] = 0
    df["protonenergy_corr"] = df["protonenergy"] + 0.000620 / df["protonenergy"] - 0.001792
    df.loc[(df["protonenergy_corr"] > 9998.0), "protonenergy_corr"] = 0
    df["reco_proton_e"] = Mp + df["protonenergy"]
    df["reco_proton_p"] = np.sqrt((df["reco_proton_e"]) ** 2 - Mp**2)
    df["reco_e_qe_l"] = (df["electron_e"] * (Mn - Eb) + 0.5 * (Mp**2 - (Mn - Eb) ** 2 - Me**2)) / (
        (Mn - Eb) - df["electron_e"] * (1 - np.cos(df["shr_theta"]))
    )
    df["reco_e_qe_p"] = (df["reco_proton_e"] * (Mn - Eb) + 0.5 * (Me**2 - (Mn - Eb) ** 2 - Mp**2)) / (
        (Mn - Eb) + df["reco_proton_p"] * np.cos(df["trk_theta"]) - df["reco_proton_e"]
    )
    df["reco_e_qe"] = (
        0.938
        * ((df["shr_energy"] + INTERCEPT) / SLOPE)
        / (0.938 - ((df["shr_energy"] + INTERCEPT) / SLOPE) * (1 - np.cos(df["shr_theta"])))
    )
    df["reco_e_rqe"] = df["reco_e_qe"] / df["reco_e"]
    return


def process_uproot_ccncpi0vars(up, df):
    mc_pdg = up.array("mc_pdg")
    mc_E = up.array("mc_E")
    mc_px = up.array("mc_px")
    mc_py = up.array("mc_py")
    mc_pz = up.array("mc_pz")

    proton_mask = mc_pdg == 2212
    pi0_mask = mc_pdg == 111
    pipm_mask = (mc_pdg == 211) | (mc_pdg == -211)
    pi_mask = pi0_mask | pipm_mask
    mostEprotIdx = get_idx_from_vec_sort(-1, mc_E, proton_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E, mostEprotIdx)
    mc_px_prot = get_elm_from_vec_idx(mc_px, mostEprotIdx)
    mc_py_prot = get_elm_from_vec_idx(mc_py, mostEprotIdx)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz, mostEprotIdx)
    mc_E_pi = sum_elements_from_mask(mc_E, pi_mask)
    mc_px_pi = sum_elements_from_mask(mc_px, pi_mask)
    mc_py_pi = sum_elements_from_mask(mc_py, pi_mask)
    mc_pz_pi = sum_elements_from_mask(mc_pz, pi_mask)
    mc_E_hadsum = mc_E_prot + mc_E_pi
    mc_px_hadsum = mc_px_prot + mc_px_pi
    mc_py_hadsum = mc_py_prot + mc_py_pi
    mc_pz_hadsum = mc_pz_prot + mc_pz_pi
    mc_M_had = np.sqrt(
        mc_E_hadsum * mc_E_hadsum
        - mc_px_hadsum * mc_px_hadsum
        - mc_py_hadsum * mc_py_hadsum
        - mc_pz_hadsum * mc_pz_hadsum
    )
    df["mc_W"] = mc_M_had
    #
    # compute hadronic mass, consider the leading proton and leading pi0 only
    proton_mask = mc_pdg == 2212
    pi0_mask = mc_pdg == 111
    mostEprotIdx = get_idx_from_vec_sort(-1, mc_E, proton_mask)
    mc_E_prot = get_elm_from_vec_idx(mc_E, mostEprotIdx)
    mc_px_prot = get_elm_from_vec_idx(mc_px, mostEprotIdx)
    mc_py_prot = get_elm_from_vec_idx(mc_py, mostEprotIdx)
    mc_pz_prot = get_elm_from_vec_idx(mc_pz, mostEprotIdx)
    mostEpi0Idx = get_idx_from_vec_sort(-1, mc_E, pi0_mask)
    mc_E_pi0 = get_elm_from_vec_idx(mc_E, mostEpi0Idx)
    mc_px_pi0 = get_elm_from_vec_idx(mc_px, mostEpi0Idx)
    mc_py_pi0 = get_elm_from_vec_idx(mc_py, mostEpi0Idx)
    mc_pz_pi0 = get_elm_from_vec_idx(mc_pz, mostEpi0Idx)
    mc_E_ppi0 = mc_E_pi0 + mc_E_prot
    mc_px_ppi0 = mc_px_pi0 + mc_px_prot
    mc_py_ppi0 = mc_py_pi0 + mc_py_prot
    mc_pz_ppi0 = mc_pz_pi0 + mc_pz_prot
    mc_M_ppi0 = np.sqrt(
        mc_E_ppi0 * mc_E_ppi0 - mc_px_ppi0 * mc_px_ppi0 - mc_py_ppi0 * mc_py_ppi0 - mc_pz_ppi0 * mc_pz_ppi0
    )
    df["mc_W_ppi0"] = mc_M_ppi0
    # compute momentum transfer
    nu_e = up.array("nu_e")
    lepton_mask = (
        (mc_pdg == 11)
        | (mc_pdg == -11)
        | (mc_pdg == 13)
        | (mc_pdg == -13)
        | (mc_pdg == 15)
        | (mc_pdg == -15)
        | (mc_pdg == 12)
        | (mc_pdg == -12)
        | (mc_pdg == 14)
        | (mc_pdg == -14)
        | (mc_pdg == 16)
        | (mc_pdg == -16)
    )
    mc_E_lep = sum_elements_from_mask(mc_E, lepton_mask)
    mc_px_lep = sum_elements_from_mask(mc_px, lepton_mask)
    mc_py_lep = sum_elements_from_mask(mc_py, lepton_mask)
    mc_pz_lep = sum_elements_from_mask(mc_pz, lepton_mask)
    mc_q_E = nu_e - mc_E_lep
    mc_q_px = -1 * mc_px_lep
    mc_q_py = -1 * mc_py_lep
    mc_q_pz = nu_e - mc_pz_lep
    mc_Q2 = -1 * (mc_q_E * mc_q_E - mc_q_px * mc_q_px - mc_q_py * mc_q_py - mc_q_pz * mc_q_pz)
    df["mc_Q2"] = mc_Q2
    #
    #
    protonidcut = 0.5  # fimxe (original was 0)
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    trk_energy_proton_v = up.array("trk_energy_proton_v")
    trk_llr_pid_score_v = up.array("trk_llr_pid_score_v")
    trk_score_v = up.array("trk_score_v")
    pi0_energy1_Y = 0.001 * up.array("pi0_energy1_Y") / 0.83
    pi0_dir1_x = up.array("pi0_dir1_x")
    pi0_dir1_y = up.array("pi0_dir1_y")
    pi0_dir1_z = up.array("pi0_dir1_z")
    pi0_energy2_Y = 0.001 * up.array("pi0_energy2_Y") / 0.83
    pi0_dir2_x = up.array("pi0_dir2_x")
    pi0_dir2_y = up.array("pi0_dir2_y")
    pi0_dir2_z = up.array("pi0_dir2_z")
    proton_mask = (trk_llr_pid_score_v < protonidcut) & (trk_score_v > 0.5)
    leadProtonIdx = get_idx_from_vec_sort(-1, trk_energy_proton_v, proton_mask)
    leadP_KE = get_elm_from_vec_idx(trk_energy_proton_v, leadProtonIdx)
    leadP_dirx = get_elm_from_vec_idx(trk_dir_x_v, leadProtonIdx)
    leadP_diry = get_elm_from_vec_idx(trk_dir_y_v, leadProtonIdx)
    leadP_dirz = get_elm_from_vec_idx(trk_dir_z_v, leadProtonIdx)
    leadP_E = leadP_KE + 0.938
    leadP_P = np.sqrt(leadP_E * leadP_E - 0.938 * 0.938)
    reco_E_hadsum = leadP_E + pi0_energy1_Y + pi0_energy2_Y
    reco_px_hadsum = leadP_P * leadP_dirx + pi0_energy1_Y * pi0_dir1_x + pi0_energy2_Y * pi0_dir2_x
    reco_py_hadsum = leadP_P * leadP_diry + pi0_energy1_Y * pi0_dir1_y + pi0_energy2_Y * pi0_dir2_y
    reco_pz_hadsum = leadP_P * leadP_dirz + pi0_energy1_Y * pi0_dir1_z + pi0_energy2_Y * pi0_dir2_z
    reco_M_had = np.sqrt(
        reco_E_hadsum * reco_E_hadsum
        - reco_px_hadsum * reco_px_hadsum
        - reco_py_hadsum * reco_py_hadsum
        - reco_pz_hadsum * reco_pz_hadsum
    )
    df["reco_W"] = reco_M_had
    # multiplicities
    mip_mask = (trk_llr_pid_score_v >= protonidcut) & (trk_score_v > 0.5)
    df["n_reco_protons"] = proton_mask.sum()
    df["n_reco_mip"] = mip_mask.sum()
    #
    # multiple pi0 combinatorics
    #
    shr_energy_y_v = up.array("shr_energy_y_v")
    shr_dist_v = up.array("shr_dist_v")
    shr_px_v = up.array("shr_px_v")
    shr_py_v = up.array("shr_py_v")
    shr_pz_v = up.array("shr_pz_v")
    trk_score_v = up.array("trk_score_v")
    shr_mask = trk_score_v < 0.5
    shr_mask_args = [np.argwhere(mask).flatten().tolist() for mask in shr_mask]
    gg_combs = ak.fromiter([combs(args) for args in shr_mask_args])
    mggs = ak.fromiter(
        [
            all_comb_mgg(ev, pxv, pyv, pzv, combs)
            for ev, pxv, pyv, pzv, combs in zip(shr_energy_y_v, shr_px_v, shr_py_v, shr_pz_v, gg_combs)
        ]
    )
    mdiffs = ak.fromiter([[np.abs(m - 134.98) for m in ms] for ms in mggs])
    gg_combs_argsort = ak.fromiter([np.argsort(d) for d in mdiffs])
    gg_unique_combs = ak.fromiter([unique_combs(c, a) for c, a in zip(gg_combs, gg_combs_argsort)])
    npi0s_delta20 = (mdiffs[gg_unique_combs] < 20).sum()
    npi0s_delta30 = (mdiffs[gg_unique_combs] < 30).sum()
    npi0s_delta40 = (mdiffs[gg_unique_combs] < 40).sum()
    npi0s_delta50 = (mdiffs[gg_unique_combs] < 50).sum()
    df["npi0s_delta20"] = npi0s_delta20
    df["npi0s_delta30"] = npi0s_delta30
    df["npi0s_delta40"] = npi0s_delta40
    df["npi0s_delta50"] = npi0s_delta50
    #
    shr_mask_025 = (trk_score_v < 0.5) & (shr_energy_y_v > 25.0)
    shr_mask_050 = (trk_score_v < 0.5) & (shr_energy_y_v > 50.0)
    shr_mask_100 = (trk_score_v < 0.5) & (shr_energy_y_v > 100.0)
    df["n_showers_025_tot2"] = shr_mask_025.sum()
    df["n_showers_050_tot2"] = shr_mask_050.sum()
    df["n_showers_100_tot2"] = shr_mask_100.sum()
    #
    # leading pi0 mc truth kinematics
    leadPi0Idx = get_idx_from_vec_sort(-1, mc_E, pi0_mask)
    leadPi0_E = get_elm_from_vec_idx(mc_E, leadPi0Idx)
    leadPi0_px = get_elm_from_vec_idx(mc_px, leadPi0Idx)
    leadPi0_py = get_elm_from_vec_idx(mc_py, leadPi0Idx)
    leadPi0_pz = get_elm_from_vec_idx(mc_pz, leadPi0Idx)
    df["leadPi0_E"] = leadPi0_E
    df["leadPi0_px"] = leadPi0_px
    df["leadPi0_py"] = leadPi0_py
    df["leadPi0_pz"] = leadPi0_pz
    df["leadPi0_uz"] = leadPi0_pz / np.sqrt(
        leadPi0_px * leadPi0_px + leadPi0_py * leadPi0_py + leadPi0_pz * leadPi0_pz
    )

    # The following code was originally located in the main loading function, but was only
    # called when loadpi0variables was set to True, which is also when this function was called.
    df["asymm"] = np.abs(df["pi0_energy1_Y"] - df["pi0_energy2_Y"]) / (df["pi0_energy1_Y"] + df["pi0_energy2_Y"])
    df["pi0energy"] = 134.98 * np.sqrt(2.0 / ((1 - (df["asymm"]) ** 2) * (1 - df["pi0_gammadot"])))
    df["pi0energygev"] = df["pi0energy"] * 0.001
    df["pi0momentum"] = np.sqrt(df["pi0energy"] ** 2 - 134.98**2)
    df["pi0beta"] = df["pi0momentum"] / df["pi0energy"]
    df["pi0thetacm"] = df["asymm"] / df["pi0beta"]
    df["pi0momx"] = df["pi0_energy2_Y"] * df["pi0_dir2_x"] + df["pi0_energy1_Y"] * df["pi0_dir1_x"]
    df["pi0momy"] = df["pi0_energy2_Y"] * df["pi0_dir2_y"] + df["pi0_energy1_Y"] * df["pi0_dir1_y"]
    df["pi0momz"] = df["pi0_energy2_Y"] * df["pi0_dir2_z"] + df["pi0_energy1_Y"] * df["pi0_dir1_z"]
    df["pi0energyraw"] = df["pi0_energy2_Y"] + df["pi0_energy1_Y"]
    df["pi0energyraw_corr"] = df["pi0energyraw"] / 0.83
    df["pi0momanglecos"] = df["pi0momz"] / df["pi0energyraw"]
    df["epicospi"] = df["pi0energy"] * (1 - df["pi0momanglecos"])
    df["boost"] = (np.abs(df["pi0_energy1_Y"] - df["pi0_energy2_Y"]) / 0.8) / (
        np.sqrt((df["pi0energy"]) ** 2 - 135**2)
    )
    df["pi0_mass_Y_corr"] = df["pi0_mass_Y"] / 0.83
    df["pi0energymin"] = 135.0 * np.sqrt(2.0 / (1 - df["pi0_gammadot"]))
    df["pi0energyminratio"] = df["pi0energyraw_corr"] / df["pi0energymin"]

    # define ratio of deposited to total shower energy for pi0
    df["pi0truth_gamma1_edep_frac"] = df["pi0truth_gamma1_edep"] / df["pi0truth_gamma1_etot"]
    df["pi0truth_gamma2_edep_frac"] = df["pi0truth_gamma2_edep"] / df["pi0truth_gamma2_etot"]
    return


def process_uproot_recoveryvars(up, df):

    #
    # data events where recovery matters should have shr2_id and trk2_id properly set
    #
    trk_id = up.array("trk_id") - 1  # I think we need this -1 to get the right result
    shr_id = up.array("shr_id") - 1  # I think we need this -1 to get the right result
    trk2_id = up.array("trk2_id") - 1  # I think we need this -1 to get the right result
    shr2_id = up.array("shr2_id") - 1  # I think we need this -1 to get the right result
    #
    shr_energy_y_v = up.array("shr_energy_y_v")
    df["trk2_energy"] = get_elm_from_vec_idx(shr_energy_y_v, trk2_id, 0.0)
    df["shr2_energy"] = get_elm_from_vec_idx(shr_energy_y_v, shr2_id, 0.0)

    #
    shr_start_x_v = up.array("shr_start_x_v")
    shr_start_y_v = up.array("shr_start_y_v")
    shr_start_z_v = up.array("shr_start_z_v")
    df["shr1_start_x"] = get_elm_from_vec_idx(shr_start_x_v, shr_id)
    df["shr2_start_x"] = get_elm_from_vec_idx(shr_start_x_v, shr2_id)
    df["shr1_start_y"] = get_elm_from_vec_idx(shr_start_y_v, shr_id)
    df["shr2_start_y"] = get_elm_from_vec_idx(shr_start_y_v, shr2_id)
    df["shr1_start_z"] = get_elm_from_vec_idx(shr_start_z_v, shr_id)
    df["shr2_start_z"] = get_elm_from_vec_idx(shr_start_z_v, shr2_id)
    #
    df["shr12_start_dx"] = df["shr2_start_x"] - df["shr1_start_x"]
    df["shr12_start_dy"] = df["shr2_start_y"] - df["shr1_start_y"]
    df["shr12_start_dz"] = df["shr2_start_z"] - df["shr1_start_z"]
    #
    df["shr12_cos_p1_dstart"] = np.where(
        (df["n_showers_contained"] < 2) | (df["shr2_energy"] < 0) | (df["shr12_start_dx"] == 0),
        np.nan,
        cos_angle_two_vecs(
            df["shr12_start_dx"], df["shr12_start_dy"], df["shr12_start_dz"], df["shr_px"], df["shr_py"], df["shr_pz"]
        ),
    )
    #
    trk_len_v = up.array("trk_len_v")
    df["trk1_len"] = get_elm_from_vec_idx(trk_len_v, trk_id)
    df["trk2_len"] = get_elm_from_vec_idx(trk_len_v, trk2_id)
    #
    trk_distance_v = up.array("trk_distance_v")
    df["trk1_distance"] = get_elm_from_vec_idx(trk_distance_v, trk_id)
    df["trk2_distance"] = get_elm_from_vec_idx(trk_distance_v, trk2_id)
    #
    trk_llr_pid_v = up.array("trk_llr_pid_score_v")
    df["trk1_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v, trk_id)
    df["trk2_llr_pid"] = get_elm_from_vec_idx(trk_llr_pid_v, trk2_id)
    #
    pfnhits_v = up.array("pfnhits")
    df["trk1_nhits"] = get_elm_from_vec_idx(pfnhits_v, trk_id)
    df["trk2_nhits"] = get_elm_from_vec_idx(pfnhits_v, trk2_id)
    df["shr1_nhits"] = get_elm_from_vec_idx(pfnhits_v, shr_id)
    df["shr2_nhits"] = get_elm_from_vec_idx(pfnhits_v, shr2_id)
    #
    trk_start_x_v = up.array("trk_start_x_v")
    trk_start_y_v = up.array("trk_start_y_v")
    trk_start_z_v = up.array("trk_start_z_v")
    df["trk1_start_x"] = get_elm_from_vec_idx(trk_start_x_v, trk_id)
    df["trk2_start_x"] = get_elm_from_vec_idx(trk_start_x_v, trk2_id)
    df["trk1_start_y"] = get_elm_from_vec_idx(trk_start_y_v, trk_id)
    df["trk2_start_y"] = get_elm_from_vec_idx(trk_start_y_v, trk2_id)
    df["trk1_start_z"] = get_elm_from_vec_idx(trk_start_z_v, trk_id)
    df["trk2_start_z"] = get_elm_from_vec_idx(trk_start_z_v, trk2_id)
    df["tk1sh1_distance"] = np.where(
        (df["n_showers_contained"] > 0) & (df["n_tracks_contained"] > 0),
        distance(
            df["shr_start_x"],
            df["shr_start_y"],
            df["shr_start_z"],
            df["trk1_start_x"],
            df["trk1_start_y"],
            df["trk1_start_z"],
        ),
        np.nan,
    )
    df["tk2sh1_distance"] = np.where(
        (df["n_showers_contained"] > 0) & (df["n_tracks_contained"] > 1),
        distance(
            df["shr_start_x"],
            df["shr_start_y"],
            df["shr_start_z"],
            df["trk2_start_x"],
            df["trk2_start_y"],
            df["trk2_start_z"],
        ),
        np.nan,
    )
    df["tk1tk2_distance"] = np.where(
        df["n_tracks_contained"] > 1,
        distance(
            df["trk1_start_x"],
            df["trk1_start_y"],
            df["trk1_start_z"],
            df["trk2_start_x"],
            df["trk2_start_y"],
            df["trk2_start_z"],
        ),
        np.nan,
    )
    #
    trk_dir_x_v = up.array("trk_dir_x_v")
    trk_dir_y_v = up.array("trk_dir_y_v")
    trk_dir_z_v = up.array("trk_dir_z_v")
    df["trk1_dir_x"] = get_elm_from_vec_idx(trk_dir_x_v, trk_id)
    df["trk2_dir_x"] = get_elm_from_vec_idx(trk_dir_x_v, trk2_id)
    df["trk1_dir_y"] = get_elm_from_vec_idx(trk_dir_y_v, trk_id)
    df["trk2_dir_y"] = get_elm_from_vec_idx(trk_dir_y_v, trk2_id)
    df["trk1_dir_z"] = get_elm_from_vec_idx(trk_dir_z_v, trk_id)
    df["trk2_dir_z"] = get_elm_from_vec_idx(trk_dir_z_v, trk2_id)

    df["tk1sh1_angle"] = np.where(
        (df["n_tracks_contained"] < 1) | (df["n_showers_contained"] < 1),
        np.nan,
        cos_angle_two_vecs(
            df["trk1_dir_x"], df["trk1_dir_y"], df["trk1_dir_z"], df["shr_px"], df["shr_py"], df["shr_pz"]
        ),
    )
    df["tk2sh1_angle"] = np.where(
        (df["n_tracks_contained"] < 2) | (df["n_showers_contained"] < 1),
        np.nan,
        cos_angle_two_vecs(
            df["trk2_dir_x"], df["trk2_dir_y"], df["trk2_dir_z"], df["shr_px"], df["shr_py"], df["shr_pz"]
        ),
    )
    df["tk1tk2_angle"] = np.where(
        df["n_tracks_contained"] < 2,
        np.nan,
        cos_angle_two_vecs(
            df["trk1_dir_x"], df["trk1_dir_y"], df["trk1_dir_z"], df["trk2_dir_x"], df["trk2_dir_y"], df["trk2_dir_z"]
        ),
    )
    #
    # todo: update also other variables, not used in the selection
    #
    # try to recover cases where the 2nd shower is split from the main one
    # note: we do not remake the shower pfp, so we ignore differences on
    # shr_score, shr_tkfit_dedx_max, trkfit since they are negligible
    # note2: in principle this can be done for 0p as well, but we focus only on Np for now
    #
    df["is_shr2splt"] = np.zeros_like(df["n_tracks_contained"])
    shr2splt = (
        (df["n_tracks_contained"] > 0)
        & (df["n_showers_contained"] > 1)
        & (df["shr12_cos_p1_dstart"] > 0.95)
        & (df["tk1sh2_distance"] > 60)
        & (df["shr_score"] < 0.1)
        & ((df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]) > 3)
    )
    df.loc[shr2splt, "is_shr2splt"] = 1
    df.loc[
        shr2splt, "n_showers_contained"
    ] = 1  # assume this happens to nues only! previously: = df["n_showers_contained"]-1
    pfpplanesubclusters_U_v = up.array("pfpplanesubclusters_U")
    pfpplanesubclusters_V_v = up.array("pfpplanesubclusters_V")
    pfpplanesubclusters_Y_v = up.array("pfpplanesubclusters_Y")
    df["shr2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v, shr2_id, 0)
    df["shr2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v, shr2_id, 0)
    df["shr2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v, shr2_id, 0)
    df.loc[shr2splt, "shrsubclusters0"] = df["shrsubclusters0"] + df["shr2subclusters0"]
    df.loc[shr2splt, "shrsubclusters1"] = df["shrsubclusters1"] + df["shr2subclusters1"]
    df.loc[shr2splt, "shrsubclusters2"] = df["shrsubclusters2"] + df["shr2subclusters2"]
    # We also have to update the subcluster variable
    df.loc[shr2splt, "subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    df.loc[shr2splt & (df["shr1shr2moliereavg"] > 0), "shrmoliereavg"] = df["shr1shr2moliereavg"]
    #
    # try to recover cases where the leading track is spurious (more than 30 cm away from nu vtx)
    # note: we redefine all track-related variables from trk2 (except pt and p for now),
    # and remove the contribution of trk1 from hit counting and energy calculation
    #
    df["is_trk1bad"] = np.zeros_like(df["n_tracks_contained"])
    trk1bad = (df["n_tracks_contained"] > 1) & (df["trk_distance"] > 30.0) & (df["is_shr2splt"] == 0)
    df.loc[trk1bad, "is_trk1bad"] = 1
    df.loc[trk1bad, "trkpid"] = df["trk2_llr_pid"]
    df.loc[trk1bad, "tksh_distance"] = df["tk2sh1_distance"]
    df.loc[trk1bad, "tksh_angle"] = df["tk2sh1_angle"]
    df.loc[trk1bad, "hits_ratio"] = df["shr_hits_tot"] / (df["shr_hits_tot"] + df["trk_hits_tot"] - df["trk1_nhits"])
    df.loc[trk1bad, "trk_len"] = df["trk2_len"]
    df.loc[trk1bad, "trk_distance"] = df["trk2_distance"]
    trk_score_v = up.array("trk_score_v")
    df["trk2_score"] = get_elm_from_vec_idx(trk_score_v, trk2_id)
    trk_energy_proton_v = up.array("trk_energy_proton_v")
    df["trk2_protonenergy"] = get_elm_from_vec_idx(trk_energy_proton_v, trk2_id)
    trk_theta_v = up.array("trk_theta_v")
    df["trk2_theta"] = get_elm_from_vec_idx(trk_theta_v, trk2_id)
    trk_phi_v = up.array("trk_phi_v")
    df["trk2_phi"] = get_elm_from_vec_idx(trk_phi_v, trk2_id)
    df.loc[trk1bad, "trk_score"] = df["trk2_score"]
    df.loc[trk1bad, "protonenergy"] = df["trk2_protonenergy"]
    df.loc[trk1bad, "trk_theta"] = df["trk2_theta"]
    df.loc[trk1bad, "trk_phi"] = df["trk2_phi"]
    df.loc[trk1bad, "trkshrhitdist2"] = df["trk2shrhitdist2"]
    df.loc[trk1bad, "n_tracks_contained"] = df["n_tracks_contained"] - 1

    df.loc[trk1bad, "trk_energy_tot"] = df["trk_energy_tot"] - df["trk_energy"]
    # note: we should redefine also pt, p
    #
    # try to recover cases where the 2nd track is actually the start of the shower
    # we need to redefine almost all shower variables (including dedx, which is tricky)
    #
    df["is_trk2srtshr"] = np.zeros_like(df["n_tracks_contained"])
    trk2srtshr = (
        (df["n_tracks_contained"] > 1)
        & (df["tk2sh1_angle"] > 0.98)
        & (df["tk1tk2_distance"] < df["tksh_distance"])
        & (df["shr_score"] < 0.1)
        & (df["is_shr2splt"] == 0)
        & (df["is_trk1bad"] == 0)
    )
    df.loc[trk2srtshr, "is_trk2srtshr"] = 1
    #
    shr_tkfit_dedx_u_v = up.array("shr_tkfit_dedx_u_v")
    shr_tkfit_dedx_v_v = up.array("shr_tkfit_dedx_v_v")
    shr_tkfit_dedx_y_v = up.array("shr_tkfit_dedx_y_v")
    shr_tkfit_nhits_u_v = up.array("shr_tkfit_dedx_nhits_u_v")
    shr_tkfit_nhits_v_v = up.array("shr_tkfit_dedx_nhits_v_v")
    shr_tkfit_nhits_y_v = up.array("shr_tkfit_dedx_nhits_y_v")
    df["trk2_tkfit_dedx_u"] = get_elm_from_vec_idx(shr_tkfit_dedx_u_v, trk2_id)
    df["trk2_tkfit_dedx_v"] = get_elm_from_vec_idx(shr_tkfit_dedx_v_v, trk2_id)
    df["trk2_tkfit_dedx_y"] = get_elm_from_vec_idx(shr_tkfit_dedx_y_v, trk2_id)
    df["trk2_tkfit_nhits_u"] = get_elm_from_vec_idx(shr_tkfit_nhits_u_v, trk2_id)
    df["trk2_tkfit_nhits_v"] = get_elm_from_vec_idx(shr_tkfit_nhits_v_v, trk2_id)
    df["trk2_tkfit_nhits_y"] = get_elm_from_vec_idx(shr_tkfit_nhits_y_v, trk2_id)
    df["trk2_tkfit_nhits_tot"] = df["trk2_tkfit_nhits_u"] + df["trk2_tkfit_nhits_v"] + df["trk2_tkfit_nhits_y"]
    df["trk2subclusters0"] = get_elm_from_vec_idx(pfpplanesubclusters_U_v, trk2_id, 0)
    df["trk2subclusters1"] = get_elm_from_vec_idx(pfpplanesubclusters_V_v, trk2_id, 0)
    df["trk2subclusters2"] = get_elm_from_vec_idx(pfpplanesubclusters_Y_v, trk2_id, 0)
    #
    df.loc[trk2srtshr, "tksh_distance"] = df["tk1tk2_distance"]
    df.loc[trk2srtshr, "tksh_angle"] = df["tk1tk2_angle"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_dedx_U"] = df["trk2_tkfit_dedx_u"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_dedx_V"] = df["trk2_tkfit_dedx_v"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_dedx_Y"] = df["trk2_tkfit_dedx_y"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_nhits_U"] = df["trk2_tkfit_nhits_u"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_nhits_V"] = df["trk2_tkfit_nhits_v"]
    df.loc[trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0), "shr_tkfit_nhits_Y"] = df["trk2_tkfit_nhits_y"]
    df.loc[trk2srtshr, "hits_ratio"] = (df["shr_hits_tot"] + df["trk2_nhits"]) / (
        df["shr_hits_tot"] + df["trk_hits_tot"]
    )
    #
    df.loc[trk2srtshr, "shr_tkfit_npointsvalid"] = df["shr_tkfit_npointsvalid"] + df["trk2_nhits"]  # patched!
    # other option... taking the track fit npoints for both (results do not change)
    df.loc[
        trk2srtshr
        & (df["trk1trk2hitdist2"] > 0)
        & (df["trkshrhitdist2"] > 0)
        & (df["trk1trk2hitdist2"] < df["trkshrhitdist2"]),
        "trkshrhitdist2",
    ] = df["trk1trk2hitdist2"]
    df.loc[trk2srtshr & (df["trk1trk2hitdist2"] > 0) & (df["trkshrhitdist2"] < 0), "trkshrhitdist2"] = df[
        "trk1trk2hitdist2"
    ]
    df.loc[trk2srtshr, "shrsubclusters0"] = df["shrsubclusters0"] + df["trk2subclusters0"]
    df.loc[trk2srtshr, "shrsubclusters1"] = df["shrsubclusters1"] + df["trk2subclusters1"]
    df.loc[trk2srtshr, "shrsubclusters2"] = df["shrsubclusters2"] + df["trk2subclusters2"]
    # If we make this correction, we must also correct the sum of the subclusters
    df.loc[trk2srtshr, "subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    df.loc[trk2srtshr & (df["shr1trk2moliereavg"] > 0), "shrmoliereavg"] = df["shr1trk2moliereavg"]
    df.loc[trk2srtshr, "n_tracks_contained"] = df["n_tracks_contained"] - 1
    df.loc[trk2srtshr, "trk_energy_tot"] = df["trk_energy_tot"] - df["trk2_protonenergy"]
    df.loc[trk2srtshr & (df["trk2_energy"] < 0.0), "trk2_energy"] = 0.0
    df["trk2_energy_cali"] = 0.001 * df["trk2_energy"] * df["shr_energy_tot_cali"] / df["shr_energy_tot"]

    #
    # try to recover cases where the 2nd shower is actually an attached proton
    #
    df["is_shr2prtn"] = np.zeros_like(df["n_tracks_contained"])
    shr2prtn = (
        (df["n_showers_contained"] > 1)
        & (df["tk1sh2_distance"] < 6.0)
        & (df["subcluster2tmp"] <= 4)
        & (df["shr2pid"] < 0.02)
    )
    df.loc[shr2prtn, "is_shr2prtn"] = 1
    df.loc[shr2prtn, "n_showers_contained"] = df["n_showers_contained"] - 1
    df.loc[shr2prtn, "n_tracks_contained"] = df["n_tracks_contained"] + 1
    df["shr2_protonenergy"] = get_elm_from_vec_idx(trk_energy_proton_v, shr2_id)
    df.loc[shr2prtn, "trk_energy_tot"] = df["trk_energy_tot"] + df["shr2_protonenergy"]
    df.loc[shr2prtn & (df["shr2_energy"] < 0.0), "shr2_energy"] = 0.0
    df["shr2_energy_cali"] = 0.001 * df["shr2_energy"] * df["shr_energy_tot_cali"] / df["shr_energy_tot"]

    df.loc[shr2prtn, "shr_energy_tot_cali"] = df["shr_energy_tot_cali"] - df["trk2_energy_cali"]

    #
    # try to recover cases where the leading track is embedded in the shower
    # todo: check that the two overlap, i.e. the shower is not downstream the track
    # todo: use distance/angle/dedx from the object closest to trk2
    # todo: in principle we should update also moliere angle and subcluster
    # FOR NOW WE JUST TAG THEM AND DO NOT TRY TO RECOVER
    #
    df["is_trk1embd"] = np.zeros_like(df["n_tracks_contained"])
    trk1embd = (df["n_tracks_contained"] > 1) & (df["tksh_angle"] > 0.99) & (df["is_trk1bad"] == 0)
    df.loc[trk1embd, "is_trk1embd"] = 1
    # Let's save memory by dropping some stuff we just used and won't use anymore
    #
    df.drop(columns=["shr1_start_x", "shr1_start_y", "shr1_start_z"], inplace=True)
    df.drop(columns=["shr2_start_x", "shr2_start_y", "shr2_start_z"], inplace=True)
    df.drop(columns=["shr12_start_dx", "shr12_start_dy", "shr12_start_dz"], inplace=True)
    # df.drop(columns=['shr2_energy'])
    df.drop(columns=["trk1_len", "trk2_len"], inplace=True)
    df.drop(columns=["trk1_distance", "trk2_distance"], inplace=True)
    df.drop(columns=["trk1_llr_pid", "trk2_llr_pid"], inplace=True)
    df.drop(columns=["trk1_nhits", "trk2_nhits"], inplace=True)
    df.drop(columns=["trk1_start_x", "trk1_start_y", "trk1_start_z"], inplace=True)
    df.drop(columns=["trk2_start_x", "trk2_start_y", "trk2_start_z"], inplace=True)
    df.drop(columns=["trk1_dir_x", "trk1_dir_y", "trk1_dir_z"], inplace=True)
    df.drop(columns=["trk2_dir_x", "trk2_dir_y", "trk2_dir_z"], inplace=True)
    df.drop(columns=["shr2subclusters0", "shr2subclusters1", "shr2subclusters2"], inplace=True)
    df.drop(columns=["trk2_score", "trk2_protonenergy"], inplace=True)
    df.drop(columns=["trk2_theta", "trk2_phi"], inplace=True)
    df.drop(columns=["trk2_tkfit_dedx_u", "trk2_tkfit_dedx_v", "trk2_tkfit_dedx_y"], inplace=True)
    df.drop(columns=["trk2_tkfit_nhits_u", "trk2_tkfit_nhits_v", "trk2_tkfit_nhits_y"], inplace=True)
    df.drop(columns=["trk2_tkfit_nhits_tot"], inplace=True)
    df.drop(columns=["trk2subclusters0", "trk2subclusters1", "trk2subclusters2"], inplace=True)
    df.drop(columns=["trk2_energy", "trk2_energy_cali"], inplace=True)

    df["n_tracks_cont_attach"] = df["n_tracks_contained"]
    df.loc[((df["tk2sh1_distance"] > 3) & (np.isfinite(df["tk2sh1_distance"]))), "n_tracks_cont_attach"] = (
        df["n_tracks_contained"] - 1
    )

    return

# Recovery vars calculation with the buggy energy estimator

def process_uproot_recoveryvars_old(up, df):

    print("WARNING: You are running the old (buggy) neutrino energy estimator")

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
                                   cos_angle_two_vecs(df["shr12_start_dx"],df["shr12_start_dy"],df["shr12_start_dz"],\
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
                            cos_angle_two_vecs(df["trk1_dir_x"],df["trk1_dir_y"],df["trk1_dir_z"],\
                                            df["shr_px"],    df["shr_py"],    df["shr_pz"]))
    df["tk2sh1_angle"] = np.where((df['n_tracks_contained']<2)|(df['n_showers_contained']<1),-9999.,
                            cos_angle_two_vecs(df["trk2_dir_x"],df["trk2_dir_y"],df["trk2_dir_z"],\
                                            df["shr_px"],    df["shr_py"],    df["shr_pz"]))
    df["tk1tk2_angle"] = np.where(df['n_tracks_contained']<2,-9999.,
                            cos_angle_two_vecs(df["trk1_dir_x"],df["trk1_dir_y"],df["trk1_dir_z"],\
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


def process_uproot_numu(up, df):
    #
    trk_llr_pid_v = up.array("trk_llr_pid_score_v")
    trk_score_v = up.array("trk_score_v")
    trk_len_v = up.array("trk_len_v")
    trk_end_x_v = up.array("trk_sce_end_x_v")
    trk_end_y_v = up.array("trk_sce_end_y_v")
    trk_end_z_v = up.array("trk_sce_end_z_v")
    trk_start_x_v = up.array("trk_sce_start_x_v")
    trk_start_y_v = up.array("trk_sce_start_y_v")
    trk_start_z_v = up.array("trk_sce_start_z_v")
   

    trk_energy_proton_v = up.array("trk_energy_proton_v")  # range-based proton kinetic energy
    trk_range_muon_mom_v = up.array("trk_range_muon_mom_v")  # range-based muon momentum
    trk_mcs_muon_mom_v = up.array("trk_mcs_muon_mom_v")
    trk_theta_v = up.array("trk_theta_v")
    trk_phi_v = up.array("trk_phi_v")
    pfp_generation_v = up.array("pfp_generation_v")
    trk_distance_v = up.array("trk_distance_v")
    trk_calo_energy_y_v = up.array("trk_calo_energy_y_v")
    trk_pfp_id_v = up.array("trk_pfp_id_v")
    pfp_pdg_v = up.array("backtracked_pdg")

    # CT: Adding track starts to the dataframe
    # df["trk_sce_start_x_v"] = trk_start_x_v
    # df["trk_sce_start_y_v"] = trk_start_y_v
    # df["trk_sce_start_z_v"] = trk_start_z_v
    # df["trk_sce_end_x_v"] = trk_end_x_v
    # df["trk_sce_end_y_v"] = trk_end_y_v
    # df["trk_sce_end_z_v"] = trk_end_z_v
    # df["trk_range_muon_mom_v"] = trk_range_muon_mom_v
    # df["trk_mcs_muon_mom_v"] = trk_mcs_muon_mom_v

    # CT: Adding pfp info to the dataframe
    # df["pfp_generation_v"] = pfp_generation_v
    # df["trk_score_v"] = trk_score_v
    # df["trk_distance_v"] = trk_distance_v
    # df["trk_len_v"] = trk_len_v

    trk_mask = trk_score_v > 0.0
    proton_mask = (trk_score_v > 0.5) & (trk_llr_pid_v < 0.0)

    df["proton_range_energy"] = ak.fromiter(
        [
            vec[vid.argsort()[-1]] if len(vid) > 0 else 0.0
            for vec, vid in zip(trk_energy_proton_v[proton_mask], trk_len_v[proton_mask])
        ]
    )

    # get element-wise reconstructed neutrino energy (for each index the value will be the neutrino energy assuming the track at that index is the muon)
    df["trk_energy_tot"] = trk_energy_proton_v.sum()
    muon_energy_correction_v = np.sqrt(trk_range_muon_mom_v**2 + 0.105**2) - trk_energy_proton_v
    # get element-wise MCS consistency
    muon_mcs_consistency_v = (trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v
    muon_calo_consistency_v = (trk_calo_energy_y_v - trk_range_muon_mom_v) / trk_range_muon_mom_v
    proton_calo_consistency_v = (trk_calo_energy_y_v * 0.001 - trk_energy_proton_v) / trk_energy_proton_v

    shr_mask = trk_score_v < 0.5
    trk_mask = trk_score_v > 0.5

    muon_candidate_idx = get_idx_from_vec_sort(-1, trk_len_v, trk_mask)

    # apply numu selection as defined by Ryan
    trk_score_v = up.array("trk_score_v")
    #'''
    muon_mask = (
        (trk_score_v > 0.8)
        & (trk_llr_pid_v > 0.2)
        & (trk_start_x_v > 5.0)
        & (trk_start_x_v < 251.0)
        & (trk_end_x_v > 5.0)
        & (trk_end_x_v < 251.0)
        & (trk_start_y_v > -110.0)
        & (trk_start_y_v < 110.0)
        & (trk_end_y_v > -110.0)
        & (trk_end_y_v < 110.0)
        & (trk_start_z_v > 20.0)
        & (trk_start_z_v < 986.0)
        & (trk_end_z_v > 20.0)
        & (trk_end_z_v < 986.0)
        & (trk_len_v > 10)
        & (trk_distance_v < 4.0)
        & (pfp_generation_v == 2)
        & (((trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v) > -0.5)
        & (((trk_mcs_muon_mom_v - trk_range_muon_mom_v) / trk_range_muon_mom_v) < 0.5)
    )

    muon_idx = get_idx_from_vec_sort(-1, trk_len_v, muon_mask)

    df["muon_length"] = get_elm_from_vec_idx(trk_len_v, muon_idx)
    df["muon_momentum"] = get_elm_from_vec_idx(trk_range_muon_mom_v, muon_idx)
    df["muon_phi"] = get_elm_from_vec_idx(trk_phi_v, muon_idx)
    df["muon_theta"] = get_elm_from_vec_idx(np.cos(trk_theta_v), muon_idx)
    df["muon_proton_energy"] = get_elm_from_vec_idx(np.cos(trk_energy_proton_v), muon_idx)
    df["muon_energy"] = np.sqrt(df["muon_momentum"] ** 2 + 0.105**2)
    # df['neutrino_energy'] = df['trk_energy_tot'] + df['muon_energy'] - df['muon_proton_energy']
    df["neutrino_energy"] = df["trk_energy_tot"] + get_elm_from_vec_idx(muon_energy_correction_v, muon_idx)
    df["muon_mcs_consistency"] = get_elm_from_vec_idx(muon_mcs_consistency_v, muon_idx)

    trk_score_v = up.array("trk_score_v")
    df["n_muons_tot"] = muon_mask.sum()
    df["n_tracks_tot"] = trk_mask.sum()
    # df['n_tracks_contained'] = contained_track_mask.sum()
    df["n_protons_tot"] = proton_mask.sum()
    df["n_showers_tot"] = shr_mask.sum()

    return

def drop_vector_columns(df):
    drop_columns = [
            "mc_pdg",
            "mc_E",
            "mc_px",
            "mc_py",
            "mc_pz",
            "trk_theta_v",
            "trk_end_z_v",
            "trk_len_v",
            "shr_tkfit_dedx_nhits_v_v",
            "trk_phi_v",
            "shr_tkfit_dedx_y_v",
            "trk_end_y_v",
            "shr_tkfit_dedx_v_v",
            "trk_end_x_v",
            "shr_tkfit_dedx_u_v",
            "shr_tkfit_dedx_nhits_y_v",
            "shr_tkfit_dedx_nhits_u_v",
            "trk_range_muon_mom_v",
            "trk_mcs_muon_mom_v" 
            "pfp_generation_v",
            "trk_dir_x_v",
            "trk_dir_y_v",
            "trk_dir_z_v",
            "trk_sce_start_x_v",
            "trk_sce_start_y_v",
            "trk_sce_start_z_v",
            "trk_sce_end_x_v",
            "trk_sce_end_y_v",
            "trk_sce_end_z_v",
            "trk_llr_pid_score_v",
            "trk_distance_v",
            "trk_score_v",
        ]
    drop_columns = [col for col in drop_columns if col in df.columns]
    df.drop(
        columns=drop_columns,
        inplace=True,
    )

# The following function should be aplied to the R3 CCPi0 sample when USEBDT and
# loadtruthfilters are both set to True.
# TODO: Use this function in the appropriate place in the code.
def apply_bdt_truth_filters(df):
    dfcsv = pd.read_csv(ls.ntuple_path + ls.RUN3 + "ccpi0nontrainevents.csv")
    dfcsv["identifier"] = dfcsv["run"] * 100000 + dfcsv["evt"]
    df["identifier"] = df["run"] * 100000 + df["evt"]
    Npre = float(df.shape[0])
    df = pd.merge(df, dfcsv, how="inner", on=["identifier"], suffixes=("", "_VAR"))
    Npost = float(df.shape[0])
    print("fraction of R3 CCpi0 sample after split : %.02f" % (Npost / Npre))


def get_rundict(run_number, category):
    thisfile_path = os.path.dirname(os.path.realpath(__file__))
   
    print(run_number)
 
    # Old ntuple paths
    #with open(os.path.join(thisfile_path, "data_paths.yml"), "r") as f:

    # New ntuple paths!
    with open(os.path.join(thisfile_path, "data_paths_2023.yml"), "r") as f:
        pathdefs = yaml.safe_load(f)

    runpaths = pathdefs[category]

    if verbose: print("get_rundict: run_number=",run_number)

    # runpaths is a list of dictionaries that each contain the 'run_id' and 'path' keys
    # Search for the dictionary where 'run_id' matches the run_number
    rundict = next((d for d in runpaths if d["run_id"] == str(run_number)), None)
    if rundict is None:
        raise ValueError(f"Run {run_number} not found in data_paths.yml for category {category}")

    return rundict

def get_pot_trig(run_number, category, dataset):
    rundict = get_rundict(run_number, category)
    # POT is the same for all detvar sets, taking it from CV
    dataset_dict = rundict[dataset] if category != "detvar" else rundict["cv"][dataset]
    pot = dataset_dict.pop("pot", None)
    trig = dataset_dict.pop("trig", None)
    if trig is not None:
        trig = int(trig)
    if pot is not None:
        pot = float(pot)
    return pot, trig


@cache_dataframe
def load_sample(
    run_number,
    category,
    dataset,
    append="",
    variation="cv",
    loadsystematics=False,
    loadpi0variables=False,
    loadshowervariables=False,
    loadrecoveryvars=False,
    loadnumuvariables=False,
    use_lee_weights=False,
    use_bdt=True,
    pi0scaling=0,
    load_crt_vars=False,
    load_numu_tki=False,
    full_path="",
    keep_columns=None,
):
    # Load the file from data_path.yml
    if full_path == "":

        if verbose: print("Using data_paths.yml to locate ntuple file")

        """Load one sample of one run for a particular kind of events."""
        
        #assert category in ["runs", "nearsidebands", "farsidebands", "fakedata", "numupresel","detvar"]
        assert category in ["runs","numupresel","detvar"]
        
        if use_bdt:
            assert loadshowervariables, "BDT requires shower variables"
        
        if use_lee_weights:
            assert category == "runs" and dataset == "nue", "LEE weights only available for nue runs"
        
        # CT: Slightly hacky way to ensure run number is >= 3 (assume first letter of string is >= 3)
        if load_crt_vars:
            assert int(run_number[0]) >= 3, "CRT variables only available for R3 and up"
        
        # The path to the actual ROOT file
        if category != "detvar":
            rundict = get_rundict(run_number, category)
            data_path = os.path.join(ls.ntuple_path, rundict["path"], rundict[dataset]["file"] + append + ".root")
            
        else: 
            rundict = get_rundict(run_number, category)
            subdir = "numupresel" if loadnumuvariables else "nuepresel"
            data_path = os.path.join(ls.ntuple_path, rundict["path"], subdir, rundict[variation][dataset]["file"] + append + ".root")
       
        if verbose: print("Loading ntuple file",data_path)
        if dataset in datasets:
            print("Dataset",dataset,"is a data or EXT file")       

 
        # try returning an empty dataframe
        if os.path.basename(data_path) == "dummy.root":
            if verbose: print("Using dummy file for run",run_number,"dataset",dataset)
            return None 

    # Load the data from its full path
    else: 
        if verbose: print("Loading file",full_path,"instead of using data_paths.yml")
        data_path = full_path 

    fold = "nuselection"
    tree = "NeutrinoSelectionFilter"

    with uproot.open(data_path) as up_file:
        up = up_file[fold][tree]

        variables = get_run_variables(
            run_number,
            category,
            dataset,
            loadsystematics=loadsystematics,
            loadpi0variables=loadpi0variables,
            loadshowervariables=loadshowervariables,
            loadrecoveryvars=loadrecoveryvars,
            loadnumuvariables=loadnumuvariables,
            use_lee_weights=use_lee_weights,
            load_crt_vars=load_crt_vars,
        )

        df = up.pandas.df(variables, flatten=False)

        df["bnbdata"] = dataset in datasets 
        df["extdata"] = dataset == "ext"

        # trk_energy_tot agrees here
        # For runs before 3, we put zeros for the CRT variables
        if int(run_number[0]) < 3:
            vardict = get_variables()
            crtvars = vardict["CRTVARS"]
            for var in crtvars:
                df[var] = 0.0

        # We also add some "one-hot" variables for the run number
        # TODO: Do we need this?
        df["run1"] = True if run_number == 1 else False
        df["run2"] = True if run_number == 2 else False
        df["run3"] = True if run_number == 3 else False
        df["run30"] = True if run_number == 3 else False
        df["run12"] = True if run_number in [1, 2] else False
        # TODO: In the old code, the 'run2' flag was set to True for all runs for
        # certain event types. Why??
        if dataset in ["drt", "nc_pi0", "cc_pi0", "cc_nopi", "cc_cpi", "nc_nopi", "nc_cpi"]:
            df["run2"] = True

        if dataset in ["nue", "drt", "lee", "nu", "ext"]:
            df["pot_scale"] = 1.0

        # If needed, load additional variables
            
        # new signal model weights 'leeweight_shwmodel'
        if use_lee_weights:
            df["leeweight_shwmodel"] = 0.
            df["e_bin1"]  = (df["elec_e"]>=0.15)   & (df["elec_e"]<0.3)
            df["e_bin2"]  = (df["elec_e"]>=0.3)    & (df["elec_e"]<0.45)
            df["e_bin3"]  = (df["elec_e"]>=0.45)   & (df["elec_e"]<0.62)
            df["e_bin4"]  = (df["elec_e"]>=0.62)   & (df["elec_e"]<0.8)
            df["e_bin5"]  = (df["elec_e"]>=0.8)    & (df["elec_e"]<1.2)
            df["pz_bin1"] = (df["elec_pz"]>=-1.)   & (df["elec_pz"]<-2./3)
            df["pz_bin2"] = (df["elec_pz"]>=-2./3) & (df["elec_pz"]<-1./3)
            df["pz_bin3"] = (df["elec_pz"]>=-1./3) & (df["elec_pz"]<0.)
            df["pz_bin4"] = (df["elec_pz"]>=0.)    & (df["elec_pz"]<0.3)
            df["pz_bin5"] = (df["elec_pz"]>=0.3)   & (df["elec_pz"]<0.72)
            df["pz_bin6"] = (df["elec_pz"]>=0.72)  & (df["elec_pz"]<=1)
                
            df["bin1"]=(df["e_bin1"]==True)  & (df["pz_bin1"]==True)
            df["bin2"]=(df["e_bin1"]==True)  & (df["pz_bin2"]==True)
            df["bin3"]=(df["e_bin1"]==True)  & (df["pz_bin3"]==True)
            df["bin4"]=(df["e_bin2"]==True)  & (df["pz_bin3"]==True)
            df["bin5"]=(df["e_bin1"]==True)  & (df["pz_bin4"]==True)
            df["bin6"]=(df["e_bin2"]==True)  & (df["pz_bin4"]==True)
            df["bin7"]=(df["e_bin1"]==True)  & (df["pz_bin5"]==True)
            df["bin8"]=(df["e_bin2"]==True)  & (df["pz_bin5"]==True)
            df["bin9"]=(df["e_bin3"]==True)  & (df["pz_bin5"]==True)
            df["bin10"]=(df["e_bin1"]==True) & (df["pz_bin6"]==True)
            df["bin11"]=(df["e_bin2"]==True) & (df["pz_bin6"]==True)
            df["bin12"]=(df["e_bin3"]==True) & (df["pz_bin6"]==True)
            df["bin13"]=(df["e_bin4"]==True) & (df["pz_bin6"]==True)
            df["bin14"]=(df["e_bin5"]==True) & (df["pz_bin6"]==True)

            df.loc[df['bin1']  == True, 'leeweight_shwmodel'] = 1.083131
            df.loc[df['bin2']  == True, 'leeweight_shwmodel'] = 1.548606
            df.loc[df['bin3']  == True, 'leeweight_shwmodel'] = 0.986919
            df.loc[df['bin4']  == True, 'leeweight_shwmodel'] = 0.174497
            df.loc[df['bin5']  == True, 'leeweight_shwmodel'] = 1.139981
            df.loc[df['bin6']  == True, 'leeweight_shwmodel'] = 0.587143
            df.loc[df['bin7']  == True, 'leeweight_shwmodel'] = 1.234858
            df.loc[df['bin8']  == True, 'leeweight_shwmodel'] = 0.141145
            df.loc[df['bin9']  == True, 'leeweight_shwmodel'] = 0.055200
            df.loc[df['bin10'] == True, 'leeweight_shwmodel'] = 10.707822
            df.loc[df['bin11'] == True, 'leeweight_shwmodel'] = 1.735584
            df.loc[df['bin12'] == True, 'leeweight_shwmodel'] = 0.367058
            df.loc[df['bin13'] == True, 'leeweight_shwmodel'] = 0.231248
            df.loc[df['bin14'] == True, 'leeweight_shwmodel'] = 0.112103

            df.drop(columns=["bin1", "bin2", "bin3", "bin4", "bin5", "bin6", "bin7", "bin8", "bin9",
                             "bin10", "bin11", "bin12", "bin13", "bin14"], inplace=True)
            df.drop(columns=["e_bin1", "e_bin2", "e_bin3", "e_bin4", "e_bin5"], inplace=True)
            df.drop(columns=["pz_bin1", "pz_bin2", "pz_bin3", "pz_bin4", "pz_bin5", "pz_bin6"], inplace=True)

        if loadnumuvariables:
            process_uproot_numu(up, df)
        if loadshowervariables:
            process_uproot_shower_variables(up, df)
        if loadrecoveryvars:
            if use_buggy_energy_estimator:
                process_uproot_recoveryvars_old(up, df)
            else:
                process_uproot_recoveryvars(up, df)
        if loadpi0variables:
            process_uproot_ccncpi0vars(up, df)
        if loadshowervariables:
            # Some variables have to be calculated after the recovery has been done
            post_process_shower_vars(up, df)
        if load_numu_tki:
            df = signal_1muNp.set_Signal1muNp(up,df)
            df = selection_1muNp.apply_selection_1muNp(up,df) 

    if use_bdt:
        add_bdt_scores(df)

    # Add the is_signal flag
    df["is_signal"] = df["category"] == 11
    is_mc = category == "runs" and dataset not in datasets and dataset != "ext" 
    print("is_mc=",is_mc)
    if is_mc:
        # The following adds MC weights and also the "flux" key.
        add_mc_weight_variables(df, pi0scaling=pi0scaling)
    # Add special LEE category
    if use_lee_weights:
        # If the category is 1, 10 or 11, we set it to 111
        df.loc[df["category"].isin([1, 10, 11]), "category"] = 111
        df["flux"] = 111

    # set EXT and DIRT contributions to 0 for fake-data studies
    if category == "fakedata":
        if dataset in ["ext", "drt"]:
            df["nslice"] = 0

    # add back the cosmic category, for background only
    df.loc[
        (df["category"] != 1)
        & (df["category"] != 10)
        & (df["category"] != 11)
        & (df["category"] != 111)
        & (df["slnunhits"] / df["slnhits"] < 0.2),
        "category",
    ] = 4

    add_paper_categories(df, dataset)

    # CT: For some reason this only run over the EXT and data in the old code
    if dataset == "ext" or dataset == "bnb":
        df = remove_duplicates(df)

    if keep_columns is not None:
        # We have to keep certain variables in order for everything to even function
        vardict = get_variables()
        minimum_columns = vardict["WEIGHTS"] + vardict["SYSTVARS"] + vardict["WEIGHTSLEE"] + ["leeweight_shwmodel"]
        minimum_columns += ["category", "paper_category", "paper_category_xsec", "category_1e1p", "interaction"]
        keep_columns = set(keep_columns) | set(minimum_columns)
        # drop all columns that are not in keep_columns in place
        df.drop(columns=set(df.columns) - set(keep_columns), inplace=True)
    return df

# CT: plotter currently requires the pot weights to be passed in as another dictionary
# Adding to this function for now, discuss when refactoring plotter.py
def _load_run(
    run_number,
    data="bnb",
    truth_filtered_sets=["nue", "drt"],
    blinded=True,
    load_lee=False,
    use_new_signal_model=False,
    numupresel=False,
    **load_sample_kwargs,
):

    category = "numupresel" if numupresel else "runs"
    # As a preparation step, we find out which variables we will need in order to do the truth-filtering
    rundict = get_rundict(1, category)
    filter_vars = set()
    for truth_set in truth_filtered_sets:
        if truth_set == "drt":
            continue
        filter_vars.update(extract_variables_from_query(rundict[truth_set]["filter"]))
    keep_columns = load_sample_kwargs.pop("keep_columns", None)
    if keep_columns is not None:
        keep_columns = set(keep_columns) | filter_vars
        print(f"Updating keep_columns with truth-filtering variables: {filter_vars}")
        load_sample_kwargs["keep_columns"] = keep_columns
    output = {}
    weights = {}
    # At a minimum, we always need data, ext and nu (mc)
    if blinded:
        data_df = None
    else:
        data_df = load_sample(run_number, category, data, **load_sample_kwargs)
        data_df["weights"] = 1.0
    data_pot, data_trig = get_pot_trig(run_number, category, data)
    weights["data"] = 1.0
    output["data"] = data_df
    ext_df = load_sample(run_number, category, "ext", **load_sample_kwargs)
    _, ext_trigger = get_pot_trig(run_number, category, "ext")  # ext has no POT
    ext_df["weights"] = data_trig / ext_trigger
    weights["ext"] = data_trig / ext_trigger
    output["ext"] = ext_df
    mc_sets = [
        "mc"
    ] + truth_filtered_sets  # CT: The existing plotter.py looks for a sample labelled "mc" mfor the numu component
    if load_lee:
        mc_sets.append("lee")
    # We get the number of expected multisim universes from the first mc set.
    # Then, we can check that every truth-filtered mc set has the same number of universes.
    # Also, we can fix the issue where multisim universes are missing for the drt sample.
    expected_multisim_universes = {"weightsGenie": None, "weightsFlux": None, "weightsReint": None}
    for mc_set in mc_sets:
        if mc_set == "lee":
            print("Loading lee sample")
            mc_df = load_sample(run_number, category, "nue", **load_sample_kwargs, use_lee_weights=True)
            mc_pot, _ = get_pot_trig(run_number, category, "nue")  # nu has no trigger number
        else:
            mc_df = load_sample(run_number, category, mc_set, **load_sample_kwargs)
            mc_pot, _ = get_pot_trig(run_number, category, mc_set)  # nu has no trigger number
        mc_df["dataset"] = mc_set
        # For better performance, we want to convert the "dataset" column into a categorical column
        # where the categories are all the entries in mc_sets
        mc_df["dataset"] = pd.Categorical(mc_df["dataset"], categories=mc_sets + ["data", "ext"])
        mc_df["weights"] = mc_df["weightSplineTimesTune"] * data_pot / mc_pot
        # For some calculations, specifically the multisim error calculations for GENIE, we need the
        # weights without the tune. We add this as a separate column here.
        mc_df["weights_no_tune"] = mc_df["weightSpline"] * data_pot / mc_pot
        if mc_set == "lee":
            mc_df["weights_oldmodel"] = mc_df["weightSplineTimesTune"] * data_pot / mc_pot
            mc_df["weights_no_tune_oldmodel"] = mc_df["weightSpline"] * data_pot / mc_pot
            mc_df["weights_shwmodel"] = mc_df["weightSplineTimesTune"] * data_pot / mc_pot
            mc_df["weights_no_tune_shwmodel"] = mc_df["weightSpline"] * data_pot / mc_pot
        
        if mc_set == "lee":
            if use_new_signal_model:
                mc_df["weights"] *= mc_df["leeweight_shwmodel"]
                mc_df["weights_no_tune"] *= mc_df["leeweight_shwmodel"]
            else:
                mc_df["weights"] *= mc_df["leeweight"]
                mc_df["weights_no_tune"] *= mc_df["leeweight"]
            mc_df["weights_oldmodel"] *= mc_df["leeweight"]
            mc_df["weights_no_tune_oldmodel"] *= mc_df["leeweight"]
            mc_df["weights_shwmodel"] *= mc_df["leeweight_shwmodel"]
            mc_df["weights_no_tune_shwmodel"] *= mc_df["leeweight_shwmodel"]
        for ms_column in expected_multisim_universes:
            multisim_weights = mc_df[ms_column].values
            n_universes = len(multisim_weights[0])
            # First, check that the number of universes is the same for every event
            assert (
                np.all([len(weights) == n_universes for weights in multisim_weights])
            ), f"Multisim weights for {mc_set} have different numbers of universes"
            if expected_multisim_universes[ms_column] is None:
                expected_multisim_universes[ms_column] = n_universes
            if n_universes != expected_multisim_universes[ms_column]:
                if mc_set == "drt" and n_universes == 0:
                    # For missing multisim universes, we replace them with a list of ones (stored as integer 1000) of the
                    # correct length
                    print(f"WARNING: {mc_set} has no {ms_column} universes, replacing with ones")
                    mc_df[ms_column] = [[1000] * expected_multisim_universes[ms_column]] * len(mc_df)
                else:
                    raise ValueError(
                        f"Multisim weights for {mc_set} have inconsistent or missing multisim universes for {ms_column}"
                    )
        weights[mc_set] = data_pot / mc_pot
        output[mc_set] = mc_df

    # Remove the truth filtered events from "mc" to avoid double-counting
    for truth_set in truth_filtered_sets:
        if truth_set == "drt":
            continue
        else:
            # The filters are all the same, so we just take them from run 1 here
            rundict = get_rundict(1, category)
            df_temp = output["mc"].query(rundict[truth_set]["filter"], engine="python")
            output["mc"].drop(index=df_temp.index, inplace=True)

    # If using one of the sideband datasets, apply the same query to the MC as well
    datadict = get_rundict(run_number,category)[data] 
    if "sideband_def" in datadict:
        sdb_def = datadict["sideband_def"]
        if verbose:
            print("The sideband data you're using had the following query applied:")
            print(sdb_def)
            print("I will also apply this query to the MC you're loading")
        for key in output:
            df_temp = output[key].query(sdb_def)
            output[key] = df_temp

    return output, weights, data_pot  # CT: Return the weight dict and data pot

def load_runs_detvar( 
    run_numbers,
    variation,
    **load_run_detvar_kwargs,
):

    """Load detector variation samples for a several runs."""

    runsdata = {}  # dictionary containing each run dictionary
    weights = {}  # dictionary containing each weights dictionary
    data_pots = np.zeros(len(run_numbers))  # array to store the POTs for each run
    # Output variables:
    output = {}  # same format as load_run output dictionary but with each dataset dataframe concatenated by run
    weights_combined = (
        {}
    )  # same format as load_run weights dictionary but with the weights combined for each dataset by run
    for run in run_numbers:
        print(type(run))
        runsdata[f"{run}"], weights[f"{run}"], data_pots[run_numbers.index(run)] = _load_run_detvar(run, variation, **load_run_detvar_kwargs)

    pot_sum = np.sum(data_pots)
    rundict = runsdata[f"{run_numbers[0]}"]
    data_sets = rundict.keys()  # get the names of the datasets that have been loaded
    for dataset in data_sets:
        if np.any([rundata[dataset] is None for rundata in runsdata.values()]):
            df = None
        else:
            df = pd.concat([rundata[dataset] for run_key, rundata in runsdata.items()])
        output[dataset] = df
        weights_arr = np.array([])  # temporary array to store the weights of a particular dataset for each run
        for run_key, weight_dict in weights.items():
            weights_arr = np.append(weights_arr, weight_dict[dataset])
        mc_pots = data_pots / weights_arr  # this is an array
        weight_sum = pot_sum / np.sum(mc_pots)  # this is a single value of the combined weight over the runs
        weights_combined[dataset] = weight_sum  # which gets stored in this output weights dictionary
    return output, weights_combined, pot_sum

def _load_run_detvar( 
    run_number,
    var,
    truth_filtered_sets=["nue"],
    load_lee=False,
    **load_sample_kwargs,
):

    """Load detector variation samples for a given run.
    
    This function is not compatible with the standard load_run function.
    It is intended to be used with the `make_detsys.py` script.
    A specialty of this function is that it returns the filter queries for
    the input truth filtered samples. This is required to apply the same
    filters when the uncertainties are calculated later when running the 
    analysis, because the RunHistGenerator might now load the same 
    truth-filtered sets.
    """
    assert var in detector_variations 
   
    assert isinstance(run_number,str), "You my only generate detector uncertainties for one run at a time"

    if verbose and run_number == 1 and var == "lydown":
        print("LY Down uncertainties is not used in run 1, loading CV sample as a dummy")
 
    output = {}
    filter_queries = {}
    assert "mc" not in truth_filtered_sets, "Unfiltered MC should not be passed in truth_filtered_sets"
    mc_sets = ["mc"] + truth_filtered_sets

    rundict = get_rundict(run_number, "detvar")
    data_pot = rundict["data_pot"]
    weights = dict()

    if load_lee:
        mc_sets.append("lee")
    for mc_set in mc_sets:
        mc_df = load_sample(run_number, "detvar", mc_set, variation=var, **load_sample_kwargs)
        mc_pot, _ = get_pot_trig(run_number, "detvar", mc_set)  # nu has no trigger number
        output[mc_set] = mc_df
        mc_df["dataset"] = mc_set
        # For detector systematics, we are using unweighted MC events. The uncertainty will be
        # calculated as the relative difference between the CV and the detector variation.
        mc_df["weights"] = np.ones(len(mc_df)) / mc_pot
        mc_df["weights_no_tune"] = np.ones(len(mc_df)) / mc_pot
        weights[mc_set] = data_pot / mc_pot

    # Remove the truth filtered events from "mc" to avoid double-counting
    for truth_set in truth_filtered_sets:
        if truth_set == "drt":
            continue
        else:
            filter_query = rundict[var][truth_set]["filter"]
            df_temp = output["mc"].query(filter_query, engine="python")
            logging.debug(f"Removing {df_temp.shape[0]} truth filtered events from {truth_set} in {var}")
            output["mc"].drop(index=df_temp.index, inplace=True)

    # output["data"] = None  # Key required by other code
    # output["ext"] = None  # Key required by other code

    return output, weights, data_pot  # CT: Return the weight dict and data pot


# CT: Keeing this in here as a separate function for the time being
def load_run_detvar( 
    run_number,
    var,
    truth_filtered_sets=["nue"],
    load_lee=False,
    **load_sample_kwargs,
):

    """Load detector variation samples for a given run.
    
    This function is not compatible with the standard load_run function.
    It is intended to be used with the `make_detsys.py` script.
    A specialty of this function is that it returns the filter queries for
    the input truth filtered samples. This is required to apply the same
    filters when the uncertainties are calculated later when running the 
    analysis, because the RunHistGenerator might now load the same 
    truth-filtered sets.
    """
    assert var in detector_variations 
   
    assert isinstance(run_number,str), "You my only generate detector uncertainties for one run at a time"

    if verbose and run_number == 1 and var == "lydown":
        print("LY Down uncertainties is not used in run 1, loading CV sample as a dummy")
 
    output = {}
    filter_queries = {}
    assert "mc" not in truth_filtered_sets, "Unfiltered MC should not be passed in truth_filtered_sets"
    mc_sets = ["mc"] + truth_filtered_sets
    if load_lee:
        mc_sets.append("lee")
    for mc_set in mc_sets:
        mc_df = load_sample(run_number, "detvar", mc_set, variation=var, **load_sample_kwargs)
        mc_pot, _ = get_pot_trig(run_number, "detvar", mc_set)  # nu has no trigger number
        output[mc_set] = mc_df
        mc_df["dataset"] = mc_set
        # For detector systematics, we are using unweighted MC events. The uncertainty will be
        # calculated as the relative difference between the CV and the detector variation.
        mc_df["weights"] = np.ones(len(mc_df)) / mc_pot
        mc_df["weights_no_tune"] = np.ones(len(mc_df)) / mc_pot

    # Remove the truth filtered events from "mc" to avoid double-counting
    for truth_set in truth_filtered_sets:
        if truth_set == "drt":
            continue
        else:
            # The filters are all the same, so we just take them from run 1 here
            rundict = get_rundict(1, "detvar")
            filter_query = rundict[var][truth_set]["filter"]
            df_temp = output["mc"].query(filter_query, engine="python")
            logging.debug(f"Removing {df_temp.shape[0]} truth filtered events from {truth_set} in {var}")
            output["mc"].drop(index=df_temp.index, inplace=True)
            filter_queries[truth_set] = filter_query
    # output["data"] = None  # Key required by other code
    # output["ext"] = None  # Key required by other code
    return output, filter_queries


def load_runs(run_numbers, **load_run_kwargs):

    # Can't use run 3 and run 3_crt at the same time - they're the same data!
    if "3" in run_numbers and "3_crt" in run_numbers:
        raise ValueError("You cannot use run 3 and run 3_crt at the same time. They contain overlapping data.")

    runsdata = {}  # dictionary containing each run dictionary
    weights = {}  # dictionary containing each weights dictionary
    data_pots = np.zeros(len(run_numbers))  # array to store the POTs for each run
    # Output variables:
    output = {}  # same format as load_run output dictionary but with each dataset dataframe concatenated by run
    weights_combined = (
        {}
    )  # same format as load_run weights dictionary but with the weights combined for each dataset by run
    for run in run_numbers:
        runsdata[f"{run}"], weights[f"{run}"], data_pots[run_numbers.index(run)] = _load_run(run, **load_run_kwargs)
    pot_sum = np.sum(data_pots)
    rundict = runsdata[f"{run_numbers[0]}"]
    data_sets = rundict.keys()  # get the names of the datasets that have been loaded
    for dataset in data_sets:
        if np.any([rundata[dataset] is None for rundata in runsdata.values()]):
            df = None
        else:
            df = pd.concat([rundata[dataset] for run_key, rundata in runsdata.items()])
        output[dataset] = df
        weights_arr = np.array([])  # temporary array to store the weights of a particular dataset for each run
        for run_key, weight_dict in weights.items():
            weights_arr = np.append(weights_arr, weight_dict[dataset])
        mc_pots = data_pots / weights_arr  # this is an array
        weight_sum = pot_sum / np.sum(mc_pots)  # this is a single value of the combined weight over the runs
        weights_combined[dataset] = weight_sum  # which gets stored in this output weights dictionary
    return output, weights_combined, pot_sum


def filter_pi0_events(df):
    # This filter was applied in the original code to all truth-filtered pi0 events.
    # The explanation was to avoid recycling unbiased ext events, i.e. selecting
    # a slice with little nue content.
    # TODO: Unterstand this.
    df = df.query("(nslice == 0 | (slnunhits / slnhits) > 0.1)")


# TODO: Use this function where appropriate (used to be when updatedProtThresh > 0)
def update_proton_threshold(df, threshold):
    df.loc[(df["category"] == 11) & (df["proton_ke"] < threshold), "category"] = 10
    df.loc[(df["nproton"] > 0) & (df["proton_ke"] < threshold), "nproton"] = 0


def add_bdt_scores(df):
    TRAINVAR = [
        "shr_score",
        "tksh_distance",
        "tksh_angle",
        "shr_tkfit_dedx_max",
        "trkfit",
        "trkpid",
        "subcluster",
        "shrmoliereavg",
        "trkshrhitdist2",
        "hits_ratio",
        "secondshower_Y_nhit",
        "secondshower_Y_vtxdist",
        "secondshower_Y_dot",
        "anglediff_Y",
        "CosmicIPAll3D",
        "CosmicDirAll3D",
    ]

    LABELS = ["pi0", "nonpi0"]
    for label, bkg_query in zip(LABELS, nue_booster.bkg_queries):
        with open(ls.pickle_path + "booster_%s_0304_extnumi.pickle" % label, "rb") as booster_file:
            booster = pickle.load(booster_file)
            df[label + "_score"] = booster.predict(xgb.DMatrix(df[TRAINVAR]), ntree_limit=booster.best_iteration)

    # 0p BDT

    TRAINVARZP = [
        "shrmoliereavg",
        "shr_score",
        "trkfit",
        "subcluster",
        "CosmicIPAll3D",
        "CosmicDirAll3D",
        "secondshower_Y_nhit",
        "secondshower_Y_vtxdist",
        "secondshower_Y_dot",
        "anglediff_Y",
        "secondshower_V_nhit",
        "secondshower_V_vtxdist",
        "secondshower_V_dot",
        "anglediff_V",
        "secondshower_U_nhit",
        "secondshower_U_vtxdist",
        "secondshower_U_dot",
        "anglediff_U",
        "shr_tkfit_2cm_dedx_U",
        "shr_tkfit_2cm_dedx_V",
        "shr_tkfit_2cm_dedx_Y",
        "shr_tkfit_gap10_dedx_U",
        "shr_tkfit_gap10_dedx_V",
        "shr_tkfit_gap10_dedx_Y",
        "shrMCSMom",
        "DeltaRMS2h",
        "shrPCA1CMed_5cm",
        "CylFrac2h_1cm",
    ]

    LABELSZP = ["bkg"]

    for label, bkg_query in zip(LABELSZP, nue_booster.bkg_queries):
        with open(ls.pickle_path + "booster_%s_0304_extnumi_vx.pickle" % label, "rb") as booster_file:
            booster = pickle.load(booster_file)
            df[label + "_score"] = booster.predict(xgb.DMatrix(df[TRAINVARZP]), ntree_limit=booster.best_iteration)


def add_mc_weight_variables(df, pi0scaling=0):
    # add MCC8-style weights
    df["weightMCC8"] = 1.0

    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.0) & (df["nu_e"] < 0.1)), "weightMCC8"] = 1.0 / 0.05
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.1) & (df["nu_e"] < 0.2)), "weightMCC8"] = 1.0 / 0.1
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.2) & (df["nu_e"] < 0.3)), "weightMCC8"] = 1.0 / 0.25
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.3) & (df["nu_e"] < 0.4)), "weightMCC8"] = 1.0 / 0.4
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.4) & (df["nu_e"] < 0.5)), "weightMCC8"] = 1.0 / 0.5
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.5) & (df["nu_e"] < 0.6)), "weightMCC8"] = 1.0 / 0.65
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.6) & (df["nu_e"] < 0.7)), "weightMCC8"] = 1.0 / 0.65
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.7) & (df["nu_e"] < 0.8)), "weightMCC8"] = 1.0 / 0.7
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.8) & (df["nu_e"] < 0.9)), "weightMCC8"] = 1.0 / 0.8
    df.loc[((df["nu_pdg"] == 12) & (df["nu_e"] > 0.9) & (df["nu_e"] < 1.0)), "weightMCC8"] = 1.0 / 0.85

    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.0) & (df["nu_e"] < 0.1)), "weightMCC8"] = 1.0 / 0.05
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.1) & (df["nu_e"] < 0.2)), "weightMCC8"] = 1.0 / 0.1
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.2) & (df["nu_e"] < 0.3)), "weightMCC8"] = 1.0 / 0.2
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.3) & (df["nu_e"] < 0.4)), "weightMCC8"] = 1.0 / 0.35
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.4) & (df["nu_e"] < 0.5)), "weightMCC8"] = 1.0 / 0.45
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.5) & (df["nu_e"] < 0.6)), "weightMCC8"] = 1.0 / 0.55
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.6) & (df["nu_e"] < 0.7)), "weightMCC8"] = 1.0 / 0.65
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.7) & (df["nu_e"] < 0.8)), "weightMCC8"] = 1.0 / 0.73
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.8) & (df["nu_e"] < 0.9)), "weightMCC8"] = 1.0 / 0.75
    df.loc[((df["nu_pdg"] == 14) & (df["nu_e"] > 0.9) & (df["nu_e"] < 1.0)), "weightMCC8"] = 1.0 / 0.8

    df.loc[df["weightTune"] <= 0, "weightTune"] = 1.0
    df.loc[df["weightTune"] == np.inf, "weightTune"] = 1.0
    df.loc[df["weightTune"] > 100, "weightTune"] = 1.0
    df.loc[np.isnan(df["weightTune"]) == True, "weightTune"] = 1.0
    df.loc[df["weightSplineTimesTune"] <= 0, "weightSplineTimesTune"] = 1.0
    df.loc[df["weightSplineTimesTune"] == np.inf, "weightSplineTimesTune"] = 1.0
    df.loc[df["weightSplineTimesTune"] > 100, "weightSplineTimesTune"] = 1.0
    df.loc[np.isnan(df["weightSplineTimesTune"]) == True, "weightSplineTimesTune"] = 1.0

    # flux parentage
    df["flux"] = np.zeros_like(df["nslice"])
    df.loc[(((df["nu_pdg"] == 12) | (df["nu_pdg"] == -12)) & (df["nu_decay_mode"] < 11)), "flux"] = 10
    df.loc[(((df["nu_pdg"] == 12) | (df["nu_pdg"] == -12)) & (df["nu_decay_mode"] > 10)), "flux"] = 1
    df.loc[(((df["nu_pdg"] == 14) | (df["nu_pdg"] == -14)) & (df["nu_decay_mode"] < 11)), "flux"] = 10
    df.loc[(((df["nu_pdg"] == 14) | (df["nu_pdg"] == -14)) & (df["nu_decay_mode"] > 10)), "flux"] = 1
    df["pi0weight"] = df["weightSpline"]
    # pi0 scaling
    if pi0scaling == 1:
        df.loc[df["npi0"] > 0, "weightSplineTimesTune"] = df["weightSpline"] * df["weightTune"] * 0.759
        df.loc[df["npi0"] > 0, "weightSpline"] = df["weightSpline"] * 0.759
    elif pi0scaling == 2:
        pi0emax = 0.6
        df.loc[(df["pi0_e"] > 0.1) & (df["pi0_e"] < pi0emax), "weightSplineTimesTune"] = df[
            "weightSplineTimesTune"
        ] * (1.0 - 0.4 * df["pi0_e"])
        df.loc[(df["pi0_e"] > 0.1) & (df["pi0_e"] >= pi0emax), "weightSplineTimesTune"] = df[
            "weightSplineTimesTune"
        ] * (1.0 - 0.4 * pi0emax)
        df.loc[(df["pi0_e"] > 0.1) & (df["pi0_e"] < pi0emax), "weightSpline"] = df["weightSpline"] * (
            1.0 - 0.4 * df["pi0_e"]
        )
        df.loc[(df["pi0_e"] > 0.1) & (df["pi0_e"] >= pi0emax), "weightSpline"] = df["weightSpline"] * (
            1.0 - 0.4 * pi0emax
        )
    elif pi0scaling == 3:
        df.loc[df["npi0"] == 1, "weightSplineTimesTune"] = df["weightSpline"] * df["weightTune"] * 0.835
        df.loc[df["npi0"] == 1, "weightSpline"] = df["weightSpline"] * 0.835
        df.loc[df["npi0"] == 2, "weightSplineTimesTune"] = df["weightSpline"] * df["weightTune"] * 0.795
        df.loc[df["npi0"] == 2, "weightSpline"] = df["weightSpline"] * 0.795

    df["dx"] = df["reco_nu_vtx_x"] - df["true_nu_vtx_sce_x"]
    df["dy"] = df["reco_nu_vtx_y"] - df["true_nu_vtx_sce_y"]
    df["dz"] = df["reco_nu_vtx_z"] - df["true_nu_vtx_sce_z"]
    df["dr"] = np.sqrt(df["dx"] * df["dx"] + df["dy"] * df["dy"] + df["dz"] * df["dz"])


def get_run_variables(
    run_number,
    category,
    dataset,
    loadsystematics=False,
    loadpi0variables=False,
    loadshowervariables=False,
    loadrecoveryvars=False,
    loadnumuvariables=False,
    use_lee_weights=False,
    load_crt_vars=False,
):
    assert category in ["runs","numupresel","detvar"]

    VARDICT = get_variables()
    VARIABLES = VARDICT["VARIABLES"]
    WEIGHTS = VARDICT["WEIGHTS"]
    WEIGHTSLEE = VARDICT["WEIGHTSLEE"]
    SYSTVARS = VARDICT["SYSTVARS"]
    NUEVARS = VARDICT["NUEVARS"]
    NUMUVARS = VARDICT["NUMUVARS"]
    RCVRYVARS = VARDICT["RCVRYVARS"]
    PI0VARS = VARDICT["PI0VARS"]

    if loadsystematics:
        WEIGHTS += SYSTVARS
        WEIGHTSLEE += SYSTVARS
    if loadpi0variables:
        VARIABLES += PI0VARS
    if loadshowervariables:
        VARIABLES += NUEVARS
    if loadrecoveryvars:
        VARIABLES += RCVRYVARS
    if loadnumuvariables:
        VARIABLES += NUMUVARS

    ALLVARS = VARIABLES

    # Weights are only available in MC runs.
    if category in ["runs", "numupresel", "detvar"] and dataset not in datasets + ["ext"]:
        if use_lee_weights:
            assert dataset == "nue", "LEE weights are only available for nue runs"
            ALLVARS += WEIGHTSLEE
        else:
            ALLVARS += WEIGHTS

    # Starting in Run 3, the CRT veto has been implemented.
    if load_crt_vars and int(run_number[0]) >= 3:
        ALLVARS += VARDICT["CRTVARS"]

    # There are some additional variables that are only used for baseline "nu"
    # MC runs.
    if dataset == "nu":
        ALLVARS += VARDICT["MCFVARS"]

    return list(set(ALLVARS))


def remove_duplicates(df):
    return df.drop_duplicates(subset=["run", "evt"], keep="last")


# Adding these functions to use in the filtterig code
def get_path(run_number, category, dataset, append=""):
    """Load one sample of one run for a particular kind of events."""

    assert category in ["runs", "nearsidebands", "farsidebands", "fakedata","detvar"]

    rundict = get_rundict(run_number, category)

    # The path to the actual ROOT file
    return os.path.join(ls.ntuple_path, rundict["path"])


def get_filename(run_number, category, dataset, variation="cv", append=""):
    """Load one sample of one run for a particular kind of events."""

    assert category in ["runs", "nearsidebands", "farsidebands", "fakedata","detvar"]

    rundict = get_rundict(run_number, category)

    # The path to the actual ROOT file
    if(category == "detvar"):
        return rundict[dataset][variation]["file"] + append + ".root"
    else:
        return rundict[dataset]["file"] + append + ".root"
