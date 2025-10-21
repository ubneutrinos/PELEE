#Program for Plotting Muons
#Written by Maitreyee Moudgalya (mmoudal)
#Edited by Jennifer Tyler (jbtyler1)

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

RUN = ["1"]#,"2","3_nocrt","3_crt","4b","4c","4d","5"]

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    truth_filtered_sets=["nue"],
    loadpi0variables=True,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    load_lee=True,
    blinded=False,
    load_crt_vars=True,
    enable_cache=True,
)

#Appendix A1 1eNp
print('Plotting for Np vars with NP presel')

selection = "None"                                          # Make own selection rules
preselection = "NP"                                         # Make own selection rules

for binning_def in vdef.NP_presel_none_1eNp:                # Variable definition name (name for list of variables
    # some binning definitions have more than 4 elements,   # in variable_defs.py) goes here
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
    axes = plotter.plot(
        category_column="paper_category",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=True,
        show_chi_square=True,
    )
    
    plt.savefig(f'Plots/Research/appendix_{preselection}_{selection}_{binning_def[0]}.pdf', bbox_inches='tight')
    plt.show()

print()
print('Done :D')
