import sys
import numpy as np
sys.path.append("../")
import data_loading as dl
import os
from microfit import run_plotter as rp
from microfit import histogram as hist
import matplotlib.pyplot as plt
from microfit import variable_definitions as vdef
from microfit import selections
from SidebandDraw import draw_sideband

#RUN_COMBOS_vv = [["1","2","3_crt"],["1","2","3_crt","4b","4c","4d","5"],["4b","4c","4d","5"]]
RUN_COMBOS_vv = [["1","2","3_nocrt","3_crt","4b","4c","4d","5"]]

sideband = "bnb"
title = None

variables = [vdef.basic_variables[1:3]]
    #To test, use [:1] at end for only first variable. OR use vdef.VariableList[number]
    #1eNp:  loosesel_variables_1eNp[0:5,6:7,9:13], evtsel_variabls[2:3], shrsel_variables[2:3,4:8], trksel_variables[1:2]
    #1e0p:  bdt_common_variables_1e0p[0:10], bdt_1e0p_variables[0:18], shrsel_variables[17:20], evtsel_variabls[3:4],
    #       bdtscore_variables[4:6], basic_variables[1:3]

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

variables = [vdef.basic_variables[1:3]]
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
