import sys
import os
import numpy as np
import pandas as pd
import pickle
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections as sel

# from microfit import detsys
# from microfit import xsec_covariances as xs
# from microfit.xsec_signal_generator import XsecCovarHistGenerator

#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
RUN = ["1","3"]
blinded = True
data="bnb"

run_combo = "Run"
for run in RUN:
    run_combo += run


#detector_variations = ["cv","lydown","lyatt","lyrayleigh","sce","recomb2","wiremodx","wiremodyz","wiremodthetaxz","wiremodthetayz"]
detector_variations = ["cv","lydown"]
        
# #for variation in detector_variations:
# for variation in dl.detector_variations:
#     sel_DVrundata = {}
#     DVrundata, DVweights, DVdata_pot = dl.load_runs_detvar(
#         run_numbers=RUN,
#         dataset=data,
#         variation=variation,
#         blinded=True,
#         loadsystematics=False,
#         loadpi0variables=False,
#         loadshowervariables=True,
#         loadrecoveryvars=False,
#         loadnumuvariables=False,
#         use_lee_weights=False,
#         use_bdt=True,
#         pi0scaling=0,
#         load_crt_vars=False,
#         load_numu_tki=False,
#         load_nue_tki=True,
#         full_path="",
#         )
    
#     for k, df in DVrundata.items():
#         sel_DVrundata[k] = df.query(query, engine='python')

#     summed_sel_DVrundata = pd.concat([df for k, df in sel_DVrundata.items()])
#     print("Length of selected dataframe: ", len(summed_sel_DVrundata))

            

def use_detvar_cache(
        *, 
        load_runs_detvar_args,
        verbose = False, 
        mc_sets = ["mc", "nue"],
        #overwrite: bool = False,
        detvar_cache_path = "",
        ): 
    
    # type: ignore

    run_combo = "Run"
    for run in load_runs_detvar_args["run_numbers"]:
        run_combo += run

    rundata = {}

    #base_fn = f"DetVar_{load_runs_detvar_args["variation"]}_{run_combo}.pkl"
    variation = load_runs_detvar_args["variation"]
    base_fn = f"DetVar_{variation}_{run_combo}.pkl"

    if not os.path.exists(detvar_cache_path):
        os.mkdir(detvar_cache_path)

    count_no_pkl = 0

    for mc_set in mc_sets:
        mc_fn = f"{mc_set}_{base_fn}"
        pkl_filepath = os.path.join(detvar_cache_path, mc_fn)

        if os.path.exists(pkl_filepath):
            if verbose:
                print("Loading cached dataframe: \n")
            rundata[mc_set] = pd.read_pickle(pkl_filepath)
        else:
            count_no_pkl += 1

    if count_no_pkl == 0:
        return rundata
    else:
        if verbose:
            print("Cache empty. Loading from scratch: \n")
        rundata, _, _ = dl.load_runs_detvar(**load_runs_detvar_args)

        for mc_set, df in rundata.items():
            mc_fn = f"{mc_set}_{base_fn}"
            pkl_filepath = os.path.join(detvar_cache_path, mc_fn)
            pd.to_pickle(df, pkl_filepath)

        return rundata
    

for variation in detector_variations:

    params = {
        #
        "run_numbers": RUN,
        "dataset": data,
        "variation": variation,
        "blinded": True,
        "loadsystematics": False,
        "loadpi0variables": False,
        "loadshowervariables": True,
        "loadrecoveryvars": False,
        "loadnumuvariables": False,
        "use_lee_weights": False,
        "use_bdt": True,
        "pi0scaling": 0,
        "load_crt_vars": False,
        "load_numu_tki": False,
        "load_nue_tki": True,
        "full_path": ""
    }

    print("Loading for the first time:")
    print()
    rundata = use_detvar_cache(
        load_runs_detvar_args = params, 
        verbose = True, 
        detvar_cache_path = "/exp/uboone/data/users/mmoudgal/PELEE/my_cached_detvars/",
        )
    
    print(f"{variation}:", rundata.keys())
    print()

    print("Loading for the second time time:")
    print()
    cached_rundata = use_detvar_cache(
        load_runs_detvar_args = params,
        verbose = True, 
        detvar_cache_path = "/exp/uboone/data/users/mmoudgal/PELEE/my_cached_detvars/",
        )
    
    print(f"{variation}:", cached_rundata.keys())
    print()
    
print("Done :)")
    



