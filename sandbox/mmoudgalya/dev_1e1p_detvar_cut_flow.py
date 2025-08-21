import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_detvar_cache import use_detvar_cache
sys.path.append("../../")
import data_loading as dl
import cut_flow as cf
from importlib import reload
reload(dl)

from microfit import run_plotter as rp
from microfit import histogram as hist

from microfit import variable_definitions as vdef
from microfit import selections as sel

# from microfit import detsys
# from microfit import xsec_covariances as xs
# from microfit.xsec_signal_generator import XsecCovarHistGenerator

#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
RUN = ["5"]

data_pot = 1.477E+20
mc_pot = {
    "mc": 1.02685E+21,
    "nue": 1.59E+23,
}


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
#detector_variations = ["cv","sce","recomb2","wiremodx"]
detector_variations = ("cv","sce","recomb2","wiremodx")

rundata = {}
for variation in detector_variations:
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
    
    # for key, df in DVrundata.items():
    #     # # Using just the POT weights
    #     # DVrundata[key]["weights"] = 1.0 * data_pot/mc_pot[key]
    #     # Using just the GENIE spline * tune weights
    #     DVrundata[key]["weights"] = 1.0 * DVrundata[key]["weightSplineTimesTune"]

    summed_DVrundata = pd.concat([df for k, df in DVrundata.items()])
    rundata[variation] = summed_DVrundata

cut_dictionary = cf.do_cut_flow(rundata, preselection, selection, *detector_variations, printed=True, weighted=True)

print()
print("Done :)")
    
