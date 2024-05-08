import sys
import numpy as np
sys.path.append("../") #original: "../"
import data_loading as dl
import os
from microfit import run_plotter as rp
from microfit import histogram as hist
import matplotlib.pyplot as plt
from microfit import variable_definitions as vdef
from microfit import selections
from SidebandDraw import draw_sideband

#RUN_COMBOS_vv = [["1","2","3_crt"],["1","2","3_crt","4b","4c","4d","5"],["4b","4c","4d","5"]]
RUN_COMBOS_vv = [["1","2","3_crt","4b","4c","4d","5"]]

sideband = "bnb"
title = None

variables = [vdef.NP_opendata_variables]
    #Kinematic Variables:   1eNp0pi -> , 1e0p0pi -> 
    #Selection Variables:   1eNp0pi preselection -> loosesel_variables_1eNp + evtsel_variabls,
    #                       1eNp0pi Loose selection -> ,
    #                       1eNp0pi BDT selection -> , 1e0p0pi preselection -> ,
    #                       1e0p0pi loose selection -> , 1e0p0pi BDT selection -> ,
selections = ["NPBDT"]
preselections = ["NP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            sideband,
            add_detsys=True,
            loadpi0variables=True,
            loadshowervariables=True,
            loadrecoveryvars=True,
            loadsystematics=True,
            load_lee=True,
            blinded=False,
            enable_cache=True,
            numupresel=False,
            loadnumuvariables=False,
            use_bdt=True,
            load_crt_vars=True,
            load_numu_tki=False)

variables = [vdef.ZP_opendata_variables]
selections = ["ZPBDT","ZPBDT_CRT"]
preselections = ["ZP","ZP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            sideband,
            add_detsys=True,
            loadpi0variables=True,
            loadshowervariables=True,
            loadrecoveryvars=True,
            loadsystematics=True,
            load_lee=True,
            blinded=False,
            enable_cache=True,
            numupresel=False,
            loadnumuvariables=False,
            use_bdt=True,
            load_crt_vars=True,
            load_numu_tki=False)

print("Finished!")
