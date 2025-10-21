# Picking out muons and plotting their energy
# Original code written by Maitreyee Moudgalya
# Edited by Jennifer Tyler
    # 2024-05-29 : Edited to pick out muons and plot their energy
    # 2024-05-30 : Added in NUMU preselection and created new selection (NUMU1SH) to plot muons with standard numu cc selection
    #              and have exactly 1 shower nearby.
    #              Plotted shower energy for corresponding events. Added code block to calculate correct scaling for this.
    # 2024-05-31 : Added in code to put histogram data into csv files.
    # 2024-06-06 : Commented out code for csv files
    # 2024-xx-xx : 

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
RUN = ["1","2","3_nocrt","3_crt","4b","4c","4d","5"]
#RUN = ["1","2","3","4b","4c","4d","5"] #important that it's a string 1) new format to include latest runs 2) to include 'mc_pdg' otherwise it gets dropped
print()

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
    enable_cache=True,
)


#Set up string for runs selected
RUNstr = ""
for run in RUN:
    RUNstr = RUNstr + run
print()
print(RUNstr)


#Calculate scaling for shower energies
for key, df in rundata.items():
    if key!='data':        
        df['shr_energy_cali'] = df['shr_energy_cali'] * 1/0.83
print()
print('Done calculating shower energies!')

print()
print('Now making histograms ...')

preselection = "NUMU"
selection = "NUMU1SHTOT"


#Making Histograms
for binning_def in vdef.OLEE_variables:
    # some binning definitions have more than 4 elements, we ignore the last ones for now
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
        category_column="category",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )
    
    plt.savefig(f'Plots/Research/{preselection}_{selection}_Run{RUNstr}_{binning_def[0]}.pdf', bbox_inches='tight')
    plt.savefig(f'Plots/Research/{preselection}_{selection}_Run{RUNstr}_{binning_def[0]}.png', bbox_inches='tight')
    plt.show()
    
print()
print('Done making histograms!')
    

#Save data in csv file
#print()
#print('Saving data as csv files ...')
#for key, df in rundata.items():
#    if key != 'data':
#        df.to_csv(f'/exp/uboone/data/users/jbtyler1/PELEE/csv_files/{key}.csv')

print()
print('All Done :D')
