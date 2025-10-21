# run this script using the mimLEE environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
#from microfit import run_plotter as rp
#from microfit import histogram as hist

#from microfit import variable_definitions as vdef
#from microfit import selections

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
        rundata[key] = pd.read_pickle(f'/exp/uboone/data/users/jbtyler1/PELEE/tmp_pickles/{key}.pkl')
        
# flatten the dataframes in rundata

for key, df in rundata.items():
    if key not in ['data']:
        print(key)
        
        if key == 'ext':
            rundata[key] = df.explode(['mc_E','mc_pdg','true_vel_vector','true_KE_vector']).reset_index(drop=True)
            #'backtracked_pdg','trk_llr_pid_score_v'
            rundata[key]['mc_pdg'] = 0
            #rundata[key]['backtracked_pdg'] = 0
        else:
            rundata[key] = df.explode(['mc_E','mc_pdg','true_vel_vector','true_KE_vector']).reset_index(drop=True)
            #'backtracked_pdg','trk_llr_pid_score_v'
        
        rundata[key].to_pickle(f'/exp/uboone/data/users/jbtyler1/PELEE/tmp_pickles/{key}_exploded.pkl')
        rundata[key].to_csv(f'/exp/uboone/data/users/jbtyler1/PELEE/csv_files/{key}_exploded.csv')
        
print('Done exploding the dataframes.')
