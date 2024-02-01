import data_loading as dl
import os
from microfit import run_plotter as rp
from microfit import histogram as hist
import matplotlib.pyplot as plt
from microfit import variable_definitions as vdef
from microfit import selections
import make_detsys as detsys

# Draw a stack of plots for lots of variable/selection/dataset combinations
# Author: C Thorpe (U of Manchester)

verb=True

def draw_sideband(RUN_COMBOS_vv,SELECTION_v,PRESELECTION_v,VARIABLE_v,DATASET,sideband_title=None,add_detsys=False,**dl_kwargs): 

  VARIABLE_v = VARIABLE_v + [vdef.normalization]

  for run_combo in RUN_COMBOS_vv:
      
      # Make a directory to store the plots
      runcombo_str=""
      for i_r in range(0,len(run_combo)):
          runcombo_str = runcombo_str + run_combo[i_r]
      
      print("Making plots for runs",run_combo)
      
      # Load the data 
      if verb: print("Loading data")
      rundata, mc_weights, data_pot = dl.load_runs(
          run_combo,
          data=DATASET,
          truth_filtered_sets=["nue","drt"],
          **dl_kwargs
      ) 
      if verb: print("Finished loading data")
         
      # Choose a preselection/selection/variable list combination, draw
      # plots for each
      for i in range(0,len(SELECTION_v)):

          selection = SELECTION_v[i]
          preselection = PRESELECTION_v[i]

          if verb: print("Making plots with preselection",preselection,"and selection",selection)
               
          if sideband_title != None and selection+"_"+DATASET not in selections.selection_categories.keys():
              sel = selections.selection_categories[selection]
              sel["title"] = sel["title"] + ", " + sideband_title 
              selections.selection_categories[selection+"_"+DATASET] = sel

            
          os.system("mkdir -p Plots/png/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)
          os.system("mkdir -p Plots/pdf/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)
          os.system("mkdir -p Plots/no_detvar_png/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)
          os.system("mkdir -p Plots/no_detvar_pdf/run_"+runcombo_str+"/"+DATASET+"/"+preselection+"_"+selection)

          for variables in VARIABLE_v:

              for binning_def in variables:

                  if verb: print("Making plot of",binning_def[0])
    
                  # Check the variable exists in the dataframes, skip this var if it doesn't
                  found_variable=True
                  for key in rundata.keys():                  
                      if binning_def[0] not in rundata[key].columns: found_variable=False
       
                  if found_variable == False:
                      print("Variable",binning_def[0],"missing frome dataframes, not drawing")
 
                  # some binning definitions have more than 4 elements,
                  # we ignore the last ones for now
                  binning = hist.Binning.from_config(*binning_def[:4])

                  # Make the detvar histogram
                  if add_detsys:
                      if verb: print("Adding detsys")
                      detsys_file=detsys.make_variations(
                          run_combo,
                          DATASET,         
                          selection,
                          preselection,
                          binning,
                          make_plots=False,
                          truth_filtered_sets=["nue"],
                          **dl_kwargs
                      )
                      if verb: print("Detector variation histograms saved as",detsys_file)
                  
                  signal_generator = hist.RunHistGenerator(
                      rundata,
                      binning,
                      data_pot=data_pot,
                      selection=selection,
                      preselection=preselection,
                      sideband_generator=None,
                      uncertainty_defaults=None,
                      detvar_data_path=detsys_file if add_detsys else None 
                  )


                  plotter = rp.RunHistPlotter(signal_generator)
                  axes = plotter.plot(
                      category_column="paper_category",
                      include_multisim_errors=True,
                      add_ext_error_floor=False,
                      show_data_mc_ratio=True,
                      show_chi_square=True,
                      smooth_ext_histogram=False,
                      add_precomputed_detsys=add_detsys
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

                  # Also make a copy of the plot without the detector uncertainties
                  if add_detsys:
                      plotter = rp.RunHistPlotter(signal_generator)
                      axes = plotter.plot(
                          category_column="paper_category",
                          include_multisim_errors=True,
                          add_ext_error_floor=False,
                          show_data_mc_ratio=True,
                          show_chi_square=True,
                          smooth_ext_histogram=False,
                          add_precomputed_detsys=False
                      )

                      pltname = "Plots/no_detvar_pdf/run_" + runcombo_str + "/" + DATASET + "/" + preselection + "_" + selection + "/" +\
                                DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + preselection + "_" + selection + ".pdf"
                      plt.savefig(pltname)
                      pltname = "Plots/no_detvar_png/run_" + runcombo_str + "/" + DATASET + "/" + preselection + "_" + selection + "/" +\
                                DATASET + "_" + binning_def[0] + "_run" + runcombo_str + "_" + preselection + "_" + selection + ".png"
                      plt.savefig(pltname)
                      plt.close()     

              
      del rundata
      del mc_weights
      del data_pot
 
