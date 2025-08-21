import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

for variation in dl.detector_variations:
    sel_DVrundata = {}
    DVrundata, DVweights, DVdata_pot = dl.load_runs_detvar(
        RUN,
        dataset=data,
        variation=variation,
        blinded=True,
        loadsystematics=False,
        loadpi0variables=False,
        loadshowervariables=True,
        loadrecoveryvars=False,
        loadnumuvariables=False,
        use_lee_weights=False,
        use_bdt=True,
        pi0scaling=0,
        load_crt_vars=False,
        load_numu_tki=False,
        load_nue_tki=True,
        full_path="",
        )
    
    for k, df in DVrundata.items():
        sel_DVrundata[k] = df.query(query, engine='python')

    summed_sel_DVrundata = pd.concat([df for k, df in sel_DVrundata.items()])

    title = f"{variation}" + "\n" + query_title
    pot_in_sci_notation = "{:.2e}".format(DVdata_pot)
    base, exponent = pot_in_sci_notation.split("e")
    pot_label = f"${base} \\times 10^{{{int(exponent)}}}$ POT"

    for binning_def in vdef.TKI_variables_1e1p:
        # some binning definitions have more than 4 elements,
        # we ignore the last ones for now
        #binning = hist.Binning.from_config(*binning_def[:4])
        binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
        print(binning_def)
        print()
        label = binning_def[0].lstrip("Reco")

        # plt.hist(sel_DVrundata["nue"][binning.variable], binning.bin_edges, histtype='step', label='Intrinsic nue')
        # plt.hist(sel_DVrundata["mc"][binning.variable], binning.bin_edges, histtype='step', label='MC - nu overlay')
        plt.hist(summed_sel_DVrundata[binning.variable], binning.bin_edges, histtype='step', label='Summed')
        #plt.legend()
        #plt.yscale('log')
        plt.xlabel(f'{binning.variable_tex}')
        plt.ylabel(f'Unweighted MC Events')
        plt.title(title)
        plt.text(0.98, 0.98, pot_label, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
        plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_detvar_{variation}_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}_zoomed.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_1e1p/analysis_plots/detsys/investigate/stats_detvar_{variation}_{binning_def[0]}_{preselection}_{selection}_{data}_{run_combo}_zoomed.png', bbox_inches='tight')
        plt.show()
        plt.clf()

print("Done :)")
    
