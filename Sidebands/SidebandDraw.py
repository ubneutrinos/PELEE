import data_loading as dl
import os
from microfit import run_plotter as rp
from microfit import histogram as hist
import matplotlib.pyplot as plt
from microfit import variable_definitions as vdef
from microfit import selections

# Draw a stack of plots for lots of variable/selection/dataset combinations
# Author: C Thorpe (U of Manchester)

def draw_sideband(RUN_COMBOS_vv,SELECTION_v,PRESELECTION_v,VARIABLE_v,DATASET,**dl_kwargs): 

  for run_combo in RUN_COMBOS_vv:
      
      # Make a directory to store the plots
      runcombo_str=""
      for i_r in range(0,len(run_combo)):
          runcombo_str = runcombo_str + run_combo[i_r]
      os.system("mkdir -p Plots/run_"+runcombo_str)
      
      print("Making plots for runs",run_combo)
     
      # Load the data 
      rundata, mc_weights, data_pot = dl.load_runs(
          run_combo,
          data=DATASET,
          truth_filtered_sets=["nue","drt"],
          **dl_kwargs
      ) 
         
      # Choose a preselection/selection/variable list combination, draw
      # plots for each
      for i in range(0,len(SELECTION_v)):
          selection = SELECTION_v[i]
          preselection = PRESELECTION_v[i]
          variables = VARIABLE_v[i]
          
          for binning_def in variables:
              
              # some binning definitions have more than 4 elements,
              # we ignore the last ones for now
              binning = hist.Binning.from_config(*binning_def[:4])
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
                  add_ext_error_floor=True,
                  show_data_mc_ratio=True,
              )

              # Form a unique name for each plot
              pltname = "Plots/run_" + runcombo_str + "/" + DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + selection + ".pdf"
              print(pltname)
              print(type(plt))
              plt.savefig(pltname)
              
      del rundata
      del mc_weights
      del data_pot
