import sys, os
sys.path.append("../../")
from data_loading import load_runs
import matplotlib.pyplot as plt
import numpy as np
from microfit.histogram import Binning, Histogram, RunHistGenerator
import pandas as pd
from microfit.run_plotter import RunHistPlotter
from colorama import Fore,Style
import csv


if __name__ == "__main__":
    print(Style.RESET_ALL)
    print("Hello, welcome")
    RUN = ["1","3","4a","4b","4c","4d","5"]  # this can be a list of several runs, i.e. [1,2,3]
    RUN = ["3"]  # this can be a list of several runs, i.e. [1,2,3]
    rundata, mc_weights, data_pot = load_runs(
        RUN,
        data="bnb",  # which data to load
        # truth_filtered_sets=["nue", "drt", "nc_pi0", "cc_pi0", "cc_nopi", "cc_cpi", "nc_nopi", "nc_cpi"],
        # Which truth-filtered MC sets to load in addition to the main MC set. At least nu_e and dirt
        # are highly recommended because the statistics at the final level of the selection are very low.
        truth_filtered_sets=["nue", "drt"],
        #truth_filtered_sets=[],
        # Choose which additional variables to load. Which ones are required may depend on the selection
        # you wish to apply.
        loadpi0variables=True,
        loadshowervariables=True,
        loadrecoveryvars=True,
        loadsystematics=True,
        # Load the nu_e set one more time with the LEE weights applied
        load_lee=True,
        # With the cache enabled, by default the loaded dataframes will be stored as HDF5 files
        # in the 'cached_dataframes' folder. This will speed up subsequent loading of the same data.
        enable_cache=True,
        blinded=False,
    )

    preselectionsZ = ["ZP","ZPLowE"]
    selectionsZ = ["ZPBDT", "ZPLOOSESEL","None"]

    preselectionsN = ["NP","NPLowE"]
    selectionsN = ["NPBDT", "NPL","None"]


    #binning_def = ("ccnc", 1, (-0., 1), r"ccnc")
    binning_def = ("reco_e", 1, (0.0, 30), r"Reconstructed Energy [GeV]")
    binning = Binning.from_config(*binning_def)
    filename = "Results_rundep_run{}_test.csv".format(RUN[0])
    rows = []

    for i in [1]:
        if i == 0:
            preselections = preselectionsN
            selections = selectionsN
        else:
            preselections = preselectionsZ
            selections = selectionsZ
        for preselection in preselections:
            for selection in selections:
                row = ['{}'.format(preselection),'{}'.format(selection),' ','']
                rows.append(row)
                print(Fore.RED + preselection, selection)
                print(Style.RESET_ALL)
                signal_generator = RunHistGenerator(
                    rundata,
                    binning,
                    data_pot=data_pot,
                    selection=selection,
                    preselection=preselection,
                )
                plotter = RunHistPlotter(signal_generator)
                ax = plotter.plot(
                category_column="category",
                include_multisim_errors=True,
                add_ext_error_floor=False,
                show_data_mc_ratio=True,
                #show_counts = False,
                )
                #print("predictions ",signal_generator.get_total_prediction())
                data_hist = (signal_generator.get_data_hist())
                POT = plotter.run_hist_generator.data_pot
                Ndata = data_hist.sum()
                NoPOT = float(data_hist.sum()/(plotter.run_hist_generator.data_pot*pow(10,-20)))
               # error = float(data_hist.std_devs[0])
                error = float(np.sqrt(Ndata)/(POT*pow(10,-20)))
                print("POT = ", POT)
                print(f"Data: {Ndata:.1f}")
                print("N/POT ", NoPOT)
                print("Data y err: ", error)
                #plt.savefig("RecoePlotwithData.png")
                row = ['{}'.format(Ndata),'{}'.format(POT),'{}'.format(NoPOT),'{}'.format(error)]
                rows.append(row)
    
    # writing to csv file
    fields = ['N', 'POT', 'N/POT', 'error']
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
    
        # writing the fields
        csvwriter.writerow(fields)
    
        # writing the data rows
        csvwriter.writerows(rows)

    #plt.savefig("RecoePLotwithData.png")

 

