import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

RUN = ["3"]
#RUN = ["1","2","3_crt","4b","4c","4d","5"]
#RUN = ["1","2","3","4b","4c","4d","5"] #important that it's a string 1) new format to include latest runs 2) to include 'mc_pdg' otherwise it gets dropped

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=False,
    load_lee=False,
    load_nue_tki=True,
    blinded=True,
    load_crt_vars=False,
    enable_cache=True,
)

for key, df in rundata.items():
    if key != 'data':
        df.to_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/stv_{key}.pkl')
        
print('Done loading data and storing as pickle files.')