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
             
          os.system("mkdir -p Plots/png/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)
          os.system("mkdir -p Plots/pdf/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)

          for variables in VARIABLE_v:

              for binning_def in variables:

                  # Check the variable exists in the dataframes, skip this var if it doesn't
                  found_variable=True
                  for key in rundata.keys():                  
                      if binning_def[0] not in rundata[key].columns: found_variable=False
                  
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
                      include_multisim_errors=False,
                      add_ext_error_floor=True,
                      show_data_mc_ratio=True,
                      show_chi_square=True,
                  )
                  
                  # Form a unique name for each plot
                  #pltname = "Plots/pdf/run_" + runcombo_str + "/" + DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + selection + ".pdf"
                  pltname = "Plots/pdf/run_" + runcombo_str + "/" + DATASET + "/" + preselection + "_" + selection + "/" +\
                            DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + preselection + "_" + selection + ".pdf"
                  plt.savefig(pltname)
                  pltname = "Plots/png/run_" + runcombo_str + "/" + DATASET + "/" + preselection + "_" + selection + "/" +\
                            DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + preselection + "_" + selection + ".png"
                  plt.savefig(pltname)
                  plt.close()     
              
      del rundata
      del mc_weights
      del data_pot
 
