import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_detvar_cache import use_detvar_cache
sys.path.append("../../")
import data_loading as dl
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections as sel

# from microfit import detsys
# from microfit import xsec_covariances as xs
# from microfit.xsec_signal_generator import XsecCovarHistGenerator

RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
blinded = True
data="bnb"

run_combo = "Run"
for run in RUN:
    run_combo += run

selection = "OnePBDT"
preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
query_title = f"{sel.selection_categories[selection]['title']}"

#detector_variations = ["cv","lydown","lyatt","lyrayleigh","sce","recomb2","wiremodx","wiremodyz","wiremodthetaxz","wiremodthetayz"]
detector_variations = ["cv"]

#wtypes = ["weightSplineTimesTune", "pot_weight"]
wtypes = ["pot_weight"]

for w in wtypes:
        #for variation in detector_variations:
        for variation in dl.detector_variations:
            sel_DVrundata = {}
            # DVrundata, DVweights, DVdata_pot = dl.load_runs_detvar(
            #     RUN,
            #     dataset=data,
            #     variation=variation,
            #     blinded=True,
            #     loadsystematics=False,
            #     loadpi0variables=False,
            #     loadshowervariables=True,
            #     loadrecoveryvars=False,
            #     loadnumuvariables=False,
            #     use_lee_weights=False,
            #     use_bdt=True,
            #     pi0scaling=0,
            #     load_crt_vars=False,
            #     load_numu_tki=False,
            #     load_nue_tki=True,
            #     full_path="",
            #     )

            params = {
                 
                "run_numbers": RUN,
                "dataset": data,
                "variation": variation,
                "blinded": True,
                "loadsystematics": False,
                "loadpi0variables": False,
                "loadshowervariables": True,
                "loadrecoveryvars": False,
                "loadnumuvariables": False,
                "use_lee_weights": False,
                "use_bdt": True,
                "pi0scaling": 0,
                "load_crt_vars": False,
                "load_numu_tki": False,
                "load_nue_tki": True,
                "full_path": ""
            }

            DVrundata = use_detvar_cache(
                load_runs_detvar_args = params, 
                verbose = True, 
                detvar_cache_path = "/exp/uboone/data/users/mmoudgal/PELEE/my_cached_detvars_nuepresel/",
                )
            
            for k, df in DVrundata.items():
                # retrieve the POT weights (data_pot/mc_pot scaling) from the loaded in dataframes
                DVrundata[k]["pot_weight"] = DVrundata[k]["weights"]
                # make the selection
                sel_DVrundata[k] = DVrundata[k].query(query, engine='python')

            summed_sel_DVrundata = pd.concat([df for k, df in sel_DVrundata.items()])
            print("Length of selected dataframe: ", len(summed_sel_DVrundata))

            #title = f"{variation}" + "\n" + query_title
            # pot_in_sci_notation = "{:.2e}".format(DVdata_pot)
            # base, exponent = pot_in_sci_notation.split("e")
            # pot_label = f"${base} \\times 10^{{{int(exponent)}}}$ POT"

            plt.hist(summed_sel_DVrundata[w], 200, range=(0, 0.2), histtype='step', label=variation)

        plt.legend()
        # #plt.yscale('log')
        plt.xlabel(f'{w}')
        plt.ylabel(f'Frequency')
        plt.yscale('log')
        plt.title(query_title)
        #plt.text(0.98, 0.98, pot_label, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
        plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_detvar_{w}_zoomed_{preselection}_{selection}_{data}_{run_combo}.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_detvar_{w}_zoomed_{preselection}_{selection}_{data}_{run_combo}.png', bbox_inches='tight')
        plt.show()
        plt.clf()

print()
print("Done :)")
    
