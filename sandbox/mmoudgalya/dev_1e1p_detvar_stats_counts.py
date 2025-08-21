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

from microfit.histogram import Binning, HistogramGenerator, MultiChannelBinning

# from microfit import detsys
# from microfit import xsec_covariances as xs
# from microfit.xsec_signal_generator import XsecCovarHistGenerator

#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"]
blinded = True
data="bnb"

run_combos = [
    #  ["1"],
    #  ["2"],
    #  ["3"],
    #  ["3_nocrt"],
    #  ["3_crt"],
    #  ["4a"],
    #  ["4b"],
    #  ["4c"],
    #  ["4d"],
    #  ["5"],
    #  ["1A_OT"],
    #  ["1B_OT"],
    #  ["1","2","3","4b","4c","4d","5"],
     ["1","2","3_nocrt","3_crt","4b","4c","4d","5"],
    #  ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"],
]

# run_combo = "Run"
# for run in RUN:
#     run_combo += run

# selection = "OnePBDT"
# preselection = "OneP_new"
selection = "NPBDT"
preselection = "NP"
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"
query_title = f"{sel.selection_categories[selection]['title']}"

#detector_variations = ["cv","lydown","lyatt","lyrayleigh","sce","recomb2","wiremodx","wiremodyz","wiremodthetaxz","wiremodthetayz"]
detector_variations = ["cv"]

for RUN in run_combos:
    run_combo = "Run"
    for run in RUN:
        run_combo += run
    for binning_def in vdef.TKI_variables_1e1p:
            # some binning definitions have more than 4 elements,
            # we ignore the last ones for now
            #binning = hist.Binning.from_config(*binning_def[:4])
            binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
            print(binning_def)
            print()
            label = binning_def[0].lstrip("Reco")
            
            summed_variations = {}
            fig, ax = plt.subplots()
            fig2, ax2 = plt.subplots()
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
                    sel_DVrundata[k] = df.query(query, engine='python')

                summed_sel_DVrundata = pd.concat([df for k, df in sel_DVrundata.items()])
                # print("Length of selected dataframe: ", len(summed_sel_DVrundata))

                # title = f"{variation}" + "\n" + query_title
                # pot_in_sci_notation = "{:.2e}".format(DVdata_pot)
                # base, exponent = pot_in_sci_notation.split("e")
                # pot_label = f"${base} \\times 10^{{{int(exponent)}}}$ POT"


                # plt.hist(sel_DVrundata["nue"][binning.variable], binning.bin_edges, histtype='step', label='Intrinsic nue')
                # plt.hist(sel_DVrundata["mc"][binning.variable], binning.bin_edges, histtype='step', label='MC - nu overlay')

                if variation == "cv":
                     plt.hist(summed_sel_DVrundata[binning.variable], binning.bin_edges, weights=summed_sel_DVrundata["weights"], histtype='step', label="CV", color="k", lw=3)
                    #  ax.hist(sel_DVrundata["nue"][binning.variable], binning.bin_edges, weights=sel_DVrundata["nue"]["weights"], histtype='step', label="CV", color="k", lw=3)
                    #  ax2.hist(sel_DVrundata["mc"][binning.variable], binning.bin_edges, weights=sel_DVrundata["mc"]["weights"], histtype='step', label="CV", color="k", lw=3)
                else:
                     plt.hist(summed_sel_DVrundata[binning.variable], binning.bin_edges, weights=summed_sel_DVrundata["weights"], histtype='step', label=variation)
                    #  ax.hist(sel_DVrundata["nue"][binning.variable], binning.bin_edges, weights=sel_DVrundata["nue"]["weights"], histtype='step', label=variation)
                    #  ax2.hist(sel_DVrundata["mc"][binning.variable], binning.bin_edges, weights=sel_DVrundata["mc"]["weights"], histtype='step', label=variation)
                
                # n, bins, patches = plt.hist(summed_sel_DVrundata[binning.variable], binning.bin_edges, histtype='step', label='Summed')
                # print("Sum of bin counts: ", sum(n))

                # Wn, Wbins, Wpatches = plt.hist(summed_sel_DVrundata[binning.variable], binning.bin_edges, weights=summed_sel_DVrundata["weights"], histtype='step', label='Summed')
                # print("Sum of weighted bin counts: ", sum(Wn))

        #         # PLotting using PELEE objects
        #         hist_dict = {}
        #         use_kde_smoothing = False
        #         options = {}
        #         for dataset in sel_DVrundata:
        #             generator = HistogramGenerator(sel_DVrundata[dataset], binning)
        #             hist_dict[dataset] = generator.generate(
        #                 use_kde_smoothing=use_kde_smoothing, options=options, extra_query=None
        # )
        #         print(hist_dict.keys())
        #         for dataset, h in hist_dict.items():
        #              if variation not in summed_variations:
        #                   summed_variations[variation] = h
        #              else:
        #                   summed_variations[variation] += h

        #         if variation == "cv":
        #              summed_variations[variation].draw(ax=ax, label="CV", color="k", show_errors=False, lw=3)
        #         else:
        #              summed_variations[variation].draw(ax=ax, label=variation, show_errors=False)

            plt.legend()
            # #plt.yscale('log')
            plt.xlabel(f'{binning.variable_tex}')
            plt.ylabel(f'Events')
            plt.title(f"Summed \n {query_title}")
            #plt.text(0.98, 0.98, pot_label, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
            plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_summed_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.pdf', bbox_inches='tight')
            plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_summed_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.png', bbox_inches='tight')
            
            # ax.legend()
            # ax2.legend()
            # ax.set_title(f"Intrinsic nue \n {query_title}")
            # ax2.set_title(f"mc (numu) \n {query_title}")
            # ax.set_xlabel(f'{binning.variable_tex}')
            # fig.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_nue_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.pdf', bbox_inches='tight')
            # fig.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_nue_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.png', bbox_inches='tight')
            # fig2.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_mc_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.pdf', bbox_inches='tight')
            # fig2.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_weighted_detvars_mc_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}.png', bbox_inches='tight')
            
            plt.show()
            plt.clf()

print("Done :)")
    
