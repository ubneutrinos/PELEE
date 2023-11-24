#!/usr/bin/env python
# coding: utf-8

# ## Produce Plots for the Pi0, NuMU, 2+ Shower, Near Sideband, Far Sideband ##

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


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


# In[3]:


from SidebandDraw import draw_sideband


# #### Select the Runs to be plotted ####

# In[4]:


RUN_COMBOS_vv = [["1"],["2"],["3"],["4b"],["4c"],["4d"],["1","2","3"],["1","2","3","4b","4c","4d"]]


# ### NuMu Sideband ###

# In[5]:


variables = [vdef.numusel_variables]

draw_sideband(RUN_COMBOS_vv,
            ["NUMU"],
            ["NUMU"],
            variables,
            "muon_sideband",
            loadpi0variables=False,
            loadshowervariables=False,
            loadrecoveryvars=False,
            loadsystematics=True,
            load_lee=False,
            blinded=False,
            enable_cache=True,
            numupresel=True,
            loadnumuvariables=True,
            use_bdt=False,
            load_numu_tki=False)


# ### High Energy Sideband ###

# In[6]:


variables = [vdef.NP_far_sideband_variables,vdef.ZP_far_sideband_variables]
selections = ["None","None","NPVL","NPL","ZPLOOSESEL","NPBDT","ZPBDT"]
preselections = ["NP","ZP","NP","NP","ZP","NP","ZP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            "shr_energy_sideband",
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
            load_numu_tki=False)


# ### Pi0 Sideband ###

# In[7]:


variables = [vdef.pi0_variables]
selections = ["PI0","ZPLOOSETWOSHR","ZPBDTTWOSHR"]
preselections = ["PI0","ZP","ZP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            "two_shr_sideband",
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
            load_numu_tki=False)


# ### Near/Far Sidebands ###

# In[8]:


variables = [vdef.NP_near_sideband_variables]
selections = ["NP_NEAR_SIDEBAND","NP_FAR_SIDEBAND"]
preselections = ["NP","NP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            "bdt_sideband",
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
            load_numu_tki=False)

variables = [vdef.ZP_near_sideband_variables]
selections = ["ZP_NEAR_SIDEBAND","ZP_FAR_SIDEBAND"]
preselections = ["ZP","ZP"]

draw_sideband(RUN_COMBOS_vv,
            selections,
            preselections,
            variables,
            "bdt_sideband",
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
            load_numu_tki=False)


# In[ ]:





# In[ ]:




