import sys
import os
import numpy as np
import pandas as pd
import pickle
from typing import Optional, Tuple, Union
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

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

