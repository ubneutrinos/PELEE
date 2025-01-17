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

print('Loaded packages')

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial",
    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", "RecoECal", "Reco_mag_q", "RecoPL",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "trkshrhitdist2", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len", "reco_e",
    "RecoLeadProton_trk_len", "RecoLeadProton_trk_trunk_dEdx_y", "RecoLeadProton_dEdx_y_per_trklen",
    "RecoLeadProtonCandidate_trk_pid", "RecoElectronCandidate_shr_pid", "RecoElectron_conversion_dist",
    "pi0_radlen1", "pi0_radlen2", "pi0_score", "nonpi0_score", "bkg_score",
    "RecoElecE", "RecoElecModMom", "RecoElecMomX", "RecoElecMomY", "RecoElecMomZ",
    "RecoLeadProtonKE", "RecoLeadProtonModMom", "RecoLeadProtonMomX", "RecoLeadProtonMomY", "RecoLeadProtonMomZ",
]

#RUN = ["5"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"]
#RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
RUN = ["1","2","3","4a","4b","4c","4d","5"]
blinded = True

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=True,
    load_lee=False,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=True,
    load_crt_vars=False,
    enable_cache=False,
)

print('Loaded dataframes')

run_combo = "Run"
for run in RUN:
    run_combo += run

from microfit import selections as sel
selection = "None"
preselection = "OneP_new"
query = f"{sel.preselection_categories[preselection]['query']}" # and {sel.selection_categories[selection]['query']}"
sel_title = f"{sel.selection_categories[selection]['title']}"

# All available MC (plus EXT) events
all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])

all_sig = all_mc['category_1e1p'] == 12
tot_sig = np.sum(all_mc.loc[all_sig, 'weights'])
tot_bkg = np.sum(all_mc.loc[~all_sig, 'weights'])
print('Total candidate signal events:', tot_sig)

# Selected events - total prediction
sel_mc = all_mc.query(query, engine='python')

is_sig = sel_mc['category_1e1p'] == 12
sel_sig = np.sum(sel_mc.loc[is_sig, 'weights'])
sel_bkg = np.sum(sel_mc.loc[~is_sig, 'weights'])
sel_evt = np.sum(sel_mc['weights'])
print('After cuts:')
print('Total signal events:', sel_sig)
print('Total background events:', sel_bkg)
print('Total events:', sel_evt)
print('sig + bkg =', sel_sig+sel_bkg)
print()
efficiency = (sel_sig / tot_sig) * 100
purity = (sel_sig / sel_evt) * 100
print('Efficiency:', efficiency)
print('Purity:', purity)
print()


# # Loose selection variables
# for binning_def in vdef.loosesel_variables_1e1p:
    
#     x_plot = np.linspace(binning_def[2][0], binning_def[2][1], 100)
#     eff = []
#     pur = []
#     product = []
    
#     for x in x_plot:
#         sig_above = np.sum(sel_mc.loc[is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
#         bkg_above = np.sum(sel_mc.loc[~is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
#         tot_sig_above = np.sum(all_mc.loc[all_sig & (all_mc[binning_def[0]] > x), 'weights'])
        
#         eff.append(sig_above/tot_sig_above)
#         pur.append(sig_above/(sig_above + bkg_above))
#         product.append((sig_above/tot_sig_above) * (sig_above/(sig_above + bkg_above)))
        
#     plt.plot(x_plot, pur, label='signal purity')
#     plt.plot(x_plot, eff, label='signal efficiency')
#     plt.plot(x_plot, product, label='purity $\\times$ efficiency')
#     plt.legend()
#     plt.grid()
#     plt.xlabel(binning_def[3])
#     plt.title(f'Purity and Efficiency Curves \n {sel_title}')
#     plt.savefig(f'plots/pur_eff/without_mom_bounds/metrics_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/pur_eff/without_mom_bounds/metrics_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
#     plt.show()
#     plt.clf()
    
    
#     #binning = hist.Binning.from_config(*binning_def[:4])
#     binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
#     print(binning_def)
#     signal_generator = hist.RunHistGenerator(
#         rundata,
#         binning,
#         data_pot=data_pot,
#         selection=selection,
#         preselection=preselection,
#         sideband_generator=None,
#         uncertainty_defaults=None,
#     )
#     plotter = rp.RunHistPlotter(signal_generator)
#     axes = plotter.plot(
#         category_column="category_1e1p",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )
    
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')

#     axes2 = plotter.plot(
#         category_column="interaction",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )

#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_selvars_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
#     plt.show()
#     plt.clf()
    

# TKI variables
for binning_def in vdef.TKI_variables_1e1p:
    
    x_plot = np.linspace(binning_def[2][0], binning_def[2][1], 100)
    eff = []
    pur = []
    product = []
    
    for x in x_plot:
        sig_above = np.sum(sel_mc.loc[is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
        bkg_above = np.sum(sel_mc.loc[~is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
        tot_sig_above = np.sum(all_mc.loc[all_sig & (all_mc[binning_def[0]] > x), 'weights'])
        
        eff.append(sig_above/tot_sig_above)
        pur.append(sig_above/(sig_above + bkg_above))
        product.append((sig_above/tot_sig_above) * (sig_above/(sig_above + bkg_above)))
        
    plt.plot(x_plot, pur, label='signal purity')
    plt.plot(x_plot, eff, label='signal efficiency')
    plt.plot(x_plot, product, label='purity $\\times$ efficiency')
    plt.legend()
    plt.grid()
    plt.xlabel(binning_def[3])
    plt.title(f'Purity and Efficiency Curves \n {sel_title}')
    plt.savefig(f'plots/pur_eff/with_mom_bounds/metrics_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
    plt.savefig(f'plots/pur_eff/with_mom_bounds/metrics_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
#     #binning = hist.Binning.from_config(*binning_def[:4])
#     binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
#     print(binning_def)
#     signal_generator = hist.RunHistGenerator(
#         rundata,
#         binning,
#         data_pot=data_pot,
#         selection=selection,
#         preselection=preselection,
#         sideband_generator=None,
#         uncertainty_defaults=None,
#     )
#     plotter = rp.RunHistPlotter(signal_generator)
#     axes = plotter.plot(
#         category_column="category_1e1p",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )
    
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')

#     axes2 = plotter.plot(
#         category_column="interaction",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )

#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_TKI_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
#     plt.show()
#     plt.clf()
    

# # TKI ingredients
# for binning_def in vdef.variables_1e1p:
    
#     x_plot = np.linspace(binning_def[2][0], binning_def[2][1], 100)
#     eff = []
#     pur = []
#     product = []
    
#     for x in x_plot:
#         sig_above = np.sum(sel_mc.loc[is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
#         bkg_above = np.sum(sel_mc.loc[~is_sig & (sel_mc[binning_def[0]] > x), 'weights'])
#         tot_sig_above = np.sum(all_mc.loc[all_sig & (all_mc[binning_def[0]] > x), 'weights'])
        
#         eff.append(sig_above/tot_sig_above)
#         pur.append(sig_above/(sig_above + bkg_above))
#         product.append((sig_above/tot_sig_above) * (sig_above/(sig_above + bkg_above)))
        
#     plt.plot(x_plot, pur, label='signal purity')
#     plt.plot(x_plot, eff, label='signal efficiency')
#     plt.plot(x_plot, product, label='purity $\\times$ efficiency')
#     plt.legend()
#     plt.grid()
#     plt.xlabel(binning_def[3])
#     plt.title(f'Purity and Efficiency Curves \n {sel_title}')
#     plt.savefig(f'plots/pur_eff/without_mom_bounds/metrics_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/pur_eff/without_mom_bounds/metrics_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
#     plt.show()
#     plt.clf()
    
    
#     #binning = hist.Binning.from_config(*binning_def[:4])
#     binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
#     print(binning_def)
#     signal_generator = hist.RunHistGenerator(
#         rundata,
#         binning,
#         data_pot=data_pot,
#         selection=selection,
#         preselection=preselection,
#         sideband_generator=None,
#         uncertainty_defaults=None,
#     )
#     plotter = rp.RunHistPlotter(signal_generator)
#     axes = plotter.plot(
#         category_column="category_1e1p",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )
    
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/topo_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')

#     axes2 = plotter.plot(
#         category_column="interaction",
#         include_multisim_errors=True,
#         add_ext_error_floor=False,
#         show_data_mc_ratio=False,
#         show_chi_square=False,
#     )

#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.pdf', bbox_inches='tight')
#     plt.savefig(f'plots/reco_study/without_mom_bounds/unfiltered/int_ingredients_{preselection}_{selection}_{binning_def[0]}_{run_combo}.png', bbox_inches='tight')
#     plt.show()
#     plt.clf()


print('Done :)')