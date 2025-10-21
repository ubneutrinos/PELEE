# run this script using python3LEE environment
# make sure to adjust the data_pot and output file name as required

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections

data_pot = 1.67e+20 # Run 1 bnb (numu) OG Test00
#data_pot = 1.67e+20 # Run 1 bnb (numu) numupresel off Test01
#data_pot = 4.55e+19 # Run 1 opendata_bnb numupresel on Test02
#data_pot = 4.56e+19 # Run 1 opendata_bnb numupresel off Test03

#data_pot = 1.0862e+21 # Runs 1-5 nue
#data_pot = 1.1076e+21 # Runs 1-5 numu
#data_pot = 6.848e+20 # Runs 1-3 nue & numu

#c = 1
#n = 1.4620
#critB = c/n

rundata = {
    'data': None,
    'ext': None,
    'mc': None,
    'nue': None,
    'drt': None
}

# Get rid of NaN
def isNaN(num):
    return num != num

def convert_to_int(mc_pdg):
    if isNaN(mc_pdg):
        return 
    else:
        return int(mc_pdg)

    
for key, df in rundata.items():
    if key != 'data':
        df = pd.read_csv(f'/exp/uboone/data/users/jbtyler1/PELEE/csv_files/{key}_exploded.csv')
        df.drop(df.query('abs(mc_pdg) >= 1000000000').index, inplace=True) #drop odd elements
        df.drop(df.query('abs(mc_pdg) == 2112').index, inplace=True) #drop neutrons
        df.drop(df.loc[df["mc_pdg"].isin([12, -12, 14, -14])].index, inplace=True) #drop neutrinos (nue->12, numu->14)
        print('Done removing neutrinos and elements in', key)
        #df.drop(df.query('true_vel_vector < 0.68399').index, inplace=True) #Note: Need to not hardcode value
        #print('Done removing particles that did not pass the critical velocity test in', key)
        df.drop(df.query('true_vel_vector > 0.68399').index, inplace=True) #Note: Need to not hardcode value
        print('Done removing particles that passed the critical velocity test in', key)
        rundata[key] = df
        
preselection = "None"
selection = "oLEEtrue" #Choose either "oLEEtrue" or "oLEEreco"

for binning_def in vdef.OLEE_truth_variables: #Choose either "OLEE_truth_variables" or "OLEE_reco_variables"
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    binning = hist.Binning.from_config(*binning_def[:4])
    print(binning_def)
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
    axes1 = plotter.plot(
        category_column="mc_pdg", #Choose either "mc_pdg" (for truth vars) or "backtracked_pdg" (for reco vars)
        include_multisim_errors=False,           #systematic errors
        add_ext_error_floor=False,               #?
        show_data_mc_ratio=False,                #ratio plot on bottom
        show_chi_square=False,                   #statistical test useful for data & reco
    )
    
    #plt.savefig(f'Plots/Research/{preselection}_{selection}_Run{RUNstr}_{binning_def[0]}.pdf', bbox_inches='tight')
    #plt.savefig(f'Plots/Research/{preselection}_{selection}_Run{RUNstr}_{binning_def[0]}.png', bbox_inches='tight')
    plt.savefig(f'Plots/Research/{preselection}_{selection}_{binning_def[0]}_SubChrnkov_Run1Test.pdf', bbox_inches='tight')
    plt.savefig(f'Plots/Research/{preselection}_{selection}_{binning_def[0]}_SubChrnkov_Run1Test.png', bbox_inches='tight')
    plt.show()
    
print('Done making plots! :D')
