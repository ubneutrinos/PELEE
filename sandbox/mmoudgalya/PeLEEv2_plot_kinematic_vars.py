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

#RUN = ["1","2","3_nocrt","3_crt","4b","4c","4d","5"]
#RUN = ["1","2","3_nocrt","3_crt"]
RUN = ["4b","4c","4d","5"]

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

print('Plotting for Np vars at all energies')

selection = "NPBDT"
preselection = "NP"

for binning_def in vdef.kinevars_1eNp:
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
    axes = plotter.plot(
        category_column="paper_category",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=True,
        show_chi_square=True,
    )
    
    plt.savefig(f'plots/PeLEEv2_plots/{selection}_{binning_def[0]}_Runs45.pdf', bbox_inches='tight')
    plt.savefig(f'plots/PeLEEv2_plots/{selection}_{binning_def[0]}_Runs45.png', bbox_inches='tight')
    plt.show()

print('Plotting for 0p vars at all energies')

selection = "ZPBDT_CRT"
preselection = "ZP"

for binning_def in vdef.kinevars_1e0p:
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
    axes2 = plotter.plot(
        category_column="paper_category",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=True,
        show_chi_square=True,
    )
    
    plt.savefig(f'plots/PeLEEv2_plots/{selection}_{binning_def[0]}_Runs45.pdf', bbox_inches='tight')
    plt.savefig(f'plots/PeLEEv2_plots/{selection}_{binning_def[0]}_Runs45.png', bbox_inches='tight')
    plt.show()

# print('Filter the dataframes to keep events with reco_e < 0.65')

# filtered_rundata = {}
# for key in rundata.keys():
#     filtered_rundata[key] = rundata[key].query('reco_e < 0.65')

# print('Plotting for Np vars at low energy')

# selection = "NPBDT"
# preselection = "NP"

# for binning_def in vdef.kinevars_1eNp_low_energy:
#     # some binning definitions have more than 4 elements,
#     # we ignore the last ones for now
#     binning = hist.Binning.from_config(*binning_def[:4])
#     print(binning_def)
#     signal_generator = hist.RunHistGenerator(
#         filtered_rundata,
#         binning,
#         data_pot=data_pot,
#         selection=selection,
#         preselection=preselection,
#         sideband_generator=None,
#         uncertainty_defaults=None,
#     )
#     plotter = rp.RunHistPlotter(signal_generator)
#     axes3 = plotter.plot(
#         category_column="paper_category",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=True,
#         show_chi_square=True,
#     )
    
#     plt.savefig(f'plots/PeLEEv2_plots/low_energy_{selection}_{binning_def[0]}_Runs45.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/PeLEEv2_plots/low_energy_{selection}_{binning_def[0]}_Runs45.png', bbox_inches='tight')
#     plt.show()

# print('Plotting for 0p vars at low energy')

# selection = "ZPBDT_CRT"
# preselection = "ZP"

# for binning_def in vdef.kinevars_1e0p_low_energy:
#     # some binning definitions have more than 4 elements,
#     # we ignore the last ones for now
#     binning = hist.Binning.from_config(*binning_def[:4])
#     print(binning_def)
#     signal_generator = hist.RunHistGenerator(
#         filtered_rundata,
#         binning,
#         data_pot=data_pot,
#         selection=selection,
#         preselection=preselection,
#         sideband_generator=None,
#         uncertainty_defaults=None,
#     )
#     plotter = rp.RunHistPlotter(signal_generator)
#     axes4 = plotter.plot(
#         category_column="paper_category",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=True,
#         show_chi_square=True,
#     )
    
#     plt.savefig(f'plots/PeLEEv2_plots/low_energy_{selection}_{binning_def[0]}_Runs45.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/PeLEEv2_plots/low_energy_{selection}_{binning_def[0]}_Runs45.png', bbox_inches='tight')
#     plt.show()

print()
print('Done :)')