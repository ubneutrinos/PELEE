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

RUN = ["1"]
#RUN = ["1","2","3","4b","4c","4d","5"] #important that it's a string 1) new format to include latest runs 2) to include 'mc_pdg' otherwise it gets dropped

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=True,
    loadnumuvariables=True,
    use_bdt=False,
    load_lee=False,
    blinded=True,
    load_crt_vars=False,
    load_oLEE=True,
    enable_cache=True,
)

print('Data POT:', data_pot)

SYSTVARS = ["weightsGenie", "weightsFlux", "weightsReint"]

for key, df in rundata.items():
    if key not in ['data', 'ext']:
        print(key)
        print(type(df))
        df.drop(SYSTVARS, axis=1)
    if key != 'data':
        df.to_pickle(f'/exp/uboone/data/users/jbtyler1/PELEE/tmp_pickles/{key}.pkl')
        
print('Done loading data, removing syst vars, and then storing as pickle files.')
