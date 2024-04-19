# run this script using python3LEE environment

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections

#RUN = ["3"]
RUN = ["1","2","3","4b","4c","4d","5"] #important that it's a string 1) new format to include latest runs 2) to include 'mc_pdg' otherwise it gets dropped

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=False,
    loadrecoveryvars=False,
    loadsystematics=True,
    use_bdt=False,
#    load_lee=True,
    load_nue_tki=True,
    blinded=True,
    enable_cache=True,
)

print(rundata.keys())
print("trk_llr_pid_score_v" in rundata["mc"].columns)
print("backtracked_pdg" in rundata["mc"].columns)
print(type(rundata["mc"]["trk_llr_pid_score_v"][0]))
print(data_pot)

SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]

for key, df in rundata.items():
    if key not in ['data', 'ext']:
        print(key)
        print(type(df))
        df.drop(SYSTVARS, axis=1)
    if key != 'data':
        df.to_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}.pkl')
        
print('Done loading data, removing syst vars, and then storing as pickle files.')