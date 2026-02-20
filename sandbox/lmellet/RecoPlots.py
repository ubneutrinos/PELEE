# Make sure the local settings ntuple path points to the unfiltered ntuples and adjust data_loading.py

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
from microfit import selections

from microfit import detsys
from microfit import xsec_covariances as xs
from microfit.xsec_signal_generator import XsecCovarHistGenerator


def repeated_nom_values(hist):
    # repeat the last bin count
    y = hist.bin_counts
    y = np.append(y, y[-1])
    return y

RUN = ["5"]
blinded = True
#data="nuwro_fd"
data="bnb"
closure_test = False
use_detvar = False

# Choose the selection cuts

selection = "None"
preselection = "None"
#selection = "OneP_NPBDTXS"
#preselection = "NUE"
category_column = "category_fixed"
sig_code = 12

#signal_query = category_column + f" == {sig_code}"
signal_query = "category_fixed == 12"

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data=data,
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=True,
    use_bdt=True,
    load_lee=False,
    load_numu_tki=False,
    load_nue_tki=False,
    keep_columns=None,
    blinded=blinded,
    load_crt_vars=True,
    enable_cache=True,
)

run_combo = "Run"
for run in RUN:
    run_combo += run
    


all_mc = pd.concat([df for k, df in rundata.items() if k not in ['data','ext']])
all_sig = all_mc.query(signal_query, engine='python')
'''
for binning_def in vdef.variables_ratio:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")
    true_var_name = "True" + label
'''

binning = 'digitized_bin_cos_trk_theta', np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5], 'digitized_bin_cos_trk_theta', r'Bin numbers for $cos\theta_{p}^{reco}$')

if blinded == False:
    show_data_mc_ratio = True
    show_chi_square = True
else:
    show_data_mc_ratio = False
    show_chi_square = False
    
if data == "nuwro_fd":
    include_multisim_errors = False
    add_precomputed_detsys = False
    show_errorband = False
    uncertainties = np.sqrt(np.diagonal(plot_cov))
    uncertainties = np.append(uncertainties, uncertainties[-1])
    override_data_cov = True
else:
    include_multisim_errors = True
    add_precomputed_detsys = use_detvar
    show_errorband = True
    override_data_cov = False

    # Total error
signal_generator = hist.RunHistGenerator(
    rundata,
    binning.copy(),
    data_pot=data_pot,
    selection=selection,
    preselection=preselection,
    sideband_generator=None,
    uncertainty_defaults=None,
    detvar_data=None,
    extra_background_fractional_error = "",
    # mc_hist_generator_cls = XsecCovarHistGenerator,
    true_var_name=None, 
    signal_query=signal_query, 
    # uncut_signal_df=rundata["nue"],
    normalization_uncertainty=[0.01,0.02]
    )
    
include_multisim_errors = False
show_errorband = False
plotter = rp.RunHistPlotter(signal_generator)
axes = plotter.plot(
    category_column=category_column,
    signal_category_num=sig_code,
    include_multisim_errors=include_multisim_errors,
    stat_variance_method="data",
    add_ext_error_floor=False,
    show_data_mc_ratio=show_data_mc_ratio,
    show_chi_square=show_chi_square,
    show_total_unconstrained=False,
    add_precomputed_detsys=add_precomputed_detsys,
    show_errorband=show_errorband,
    override_data_cov = override_data_cov
)

if data == "nuwro_fd":
    ax = axes[0]
    ax.fill_between(
        binning.bin_edges,
        np.clip(repeated_nom_values(total_prediction) - uncertainties, 0, None),
        repeated_nom_values(total_prediction) + uncertainties,
        alpha=1.0, #0.5,
        step="post",
        label="My uncertainty",
        #color="gray",
        linewidth=0.0,
        hatch="///////",
        facecolor="none",
        edgecolor=(0.1, 0.1, 0.1),
        )
ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    
plt.savefig(f'analysis_plots/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
plt.savefig(f'analysis_plots/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
plt.show()
plt.clf()

print('Done :)')