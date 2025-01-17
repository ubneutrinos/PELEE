# run this script using python3LEE environment
# make sure to adjust the data_pot and output file name as required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections


data_pot = 1.0862e+21 # Runs 1-5 nue
#data_pot = 1.1076e+21 # Runs 1-5 numu
#data_pot = 6.848e+20 # Runs 1-3 nue & numu

rundata = {
    'data': None,
    'ext': None,
    'mc': None,
    'nue': None,
    'drt': None
}

for key, df in rundata.items():
    if key != 'data':
        df = pd.read_csv(f'/exp/uboone/data/users/mmoudgal/PELEE/{key}_exploded.csv')
        #df["backtracked_pdg"] = df["backtracked_pdg"].abs()  
        rundata[key] = df

selection = "OnePL_new"
preselection = "OneP_new"

for binning_def in vdef.variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    binning = hist.Binning.from_config(*binning_def[:4])
    #binning = hist.Binning.from_config(*binning_def)
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning,
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
    )
    plotter = rp.RunHistPlotter(signal_generator)
    axes = plotter.plot(
        category_column="backtracked_pdg",
        include_multisim_errors=False,
        add_ext_error_floor=False,
#        show_data_mc_ratio=True,
    )
    plt.savefig(f'plots/investigation/{binning_def[0]}_Run12345_nue_{preselection}_{selection}.pdf', bbox_inches='tight')
    
print('Done making LLR PID plot.')