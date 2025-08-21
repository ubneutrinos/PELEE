import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")

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
        rundata[key] = pd.read_pickle(f'/exp/uboone/data/users/mmoudgal/PELEE/stv_{key}.pkl')

for key, df in rundata.items():
    if key != 'data':
        file = uproot.recreate(f"/exp/uboone/data/users/mmoudgal/PELEE/stv_{key}.root")
        #file["summed_pot"] = data_pot
        file["stv_tree"] = rundata[key]
        
print('Done saving as root files.')