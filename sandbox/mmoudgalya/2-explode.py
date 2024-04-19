# run this script using the mimLEE environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections

data_pot = 1.0862e+21

rundata = {
    'data': None,
    'ext': None,
    'mc': None,
    'nue': None,
    'drt': None
}

for key, df in rundata.items():
    if key != 'data':
        print(key)
        rundata[key] = pd.read_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}.pkl')
        
# flatten the dataframes in rundata wrt trk_llr_pid_score_v and backtracked_pdg

for key, df in rundata.items():
    if key not in ['data']:
        print(key)
        
        if key == 'ext':
            rundata[key] = df.explode(['trk_llr_pid_score_v']).reset_index(drop=True)
            rundata[key]['backtracked_pdg'] = 0
        else:
            rundata[key] = df.explode(['trk_llr_pid_score_v', 'backtracked_pdg']).reset_index(drop=True)
        
        rundata[key].to_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}_exploded.pkl')
        rundata[key].to_csv(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}_exploded.csv')
        
print('Done.')