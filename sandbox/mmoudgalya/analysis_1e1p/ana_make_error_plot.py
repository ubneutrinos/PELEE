# Make sure the local settings ntuple path points to the unfiltered ntuples and adjust data_loading.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../../")
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

def plot_cov_matrix(cov, binning_def, binning):

    if binning_def[1] is None:
        edges = binning.bin_edges
    else:
        edges = binning.bin_centers
            
    X, Y = np.meshgrid(edges,edges)
    max_val = np.max(np.abs(cov))
    plt.pcolormesh(X, Y, cov, cmap="RdBu_r", vmin=-max_val, vmax=max_val, shading='flat') #, norm=LogNorm())
    plt.colorbar(label = "Covariance")
    plt.xlabel(f'{binning.variable_tex}')
    plt.ylabel(f'{binning.variable_tex}')
    plt.title(f"Covariance Matrix")

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
    "ccnc",
]

#RUN = ["3"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"] # use this if using CRT
RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"] # for detvars with bnb or for closure test
#RUN = ["1","2","3","4a","4c","5"] # for nuwro_fd, no run 4b and 4d available
blinded = True
#data="nuwro_fd"
data="bnb"
closure_test = True

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data=data,
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
    blinded=blinded,
    load_crt_vars=False,
    enable_cache=True,
)
print('Loaded data')

run_combo = "Run"
for run in RUN:
    run_combo += run

# Calculating the integrated flux:

# Zarko Pavlovic, Jun 22 2020
# /pnfs/uboone/persistent/uboonebeam/bnb_gsimple/bnb_gsimple_fluxes_01.09.2019_463_hist/readme.txt

# Nominal_UB_XY_Surface = 256.35*233. # cm2
# SoftFidSurface = 236. * 210.  # cm2
# POTPerSpill = 4997.*5e8
# HistoFlux_int = 593641.00 # retrieved from the root file itself by running hEnue_cv->Integral()
# IntegratedFlux = (HistoFlux_int * data_pot / POTPerSpill / Nominal_UB_XY_Surface)
# print('Integrated flux:', IntegratedFlux)

selection = "OnePBDT"
preselection = "OneP_new"

#detector_variations = ["cv","lydown","lyatt","lyrayleigh","sce","recomb2","wiremodx","wiremodyz","wiremodthetaxz","wiremodthetayz"]

IntegratedFlux = 1

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    
    # Load detvars
    detvar_data = detsys.make_variations(
    run_numbers=RUN,
    data="bnb",
    binning=binning.copy(),
    selection=selection,
    preselection=preselection,
    use_kde_smoothing=False,
    make_plots=True,
    plot_output_dir= "/exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya/analysis_1e1p/analysis_plots/detsys/investigate/",
    enable_detvar_cache=True,
    detvar_cache_dir="/exp/uboone/data/users/mmoudgal/PELEE/detvar_cached_dataframes/",
    extra_selection_query=None,
    show_plots=True,
    )
    
    #plt.clf()
    
    # Total error
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=detvar_data,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=[0.01,0.02]
    )
    total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True, add_precomputed_detsys=True, smooth_detsys_variations=True)
    bin_counts = total_prediction.bin_counts
    bin_edges = binning.bin_edges
    n_bins = len(bin_edges) - 1
    print(f"binning_def[0] bin counts:", bin_counts)
    flux_norm_total_prediction = total_prediction / IntegratedFlux
    total_cov = flux_norm_total_prediction.covariance_matrix
    total_error = np.sqrt(np.diagonal(total_cov)) / bin_counts
    print('total error:', total_error)
    print()

    fig, ax = plt.subplots()
    flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_total_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_total_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False, as_fractional=True)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_frac_total_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_frac_total_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # Stat error 
    #(included by default - just need to turn flags off for syst errors)
    # Hence, will need to subtract stat_cov when calculating all the individual syst error contributions
    signal_generator_stat = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=None
    )
    
    total_prediction_stat = signal_generator_stat.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
    flux_norm_total_prediction_stat = total_prediction_stat / IntegratedFlux
    stat_cov = flux_norm_total_prediction_stat.covariance_matrix
    stat_error = np.sqrt(np.diagonal(stat_cov)) / bin_counts
    print('stat error:', stat_error)
    print()

    plot_cov_matrix(stat_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_stat_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_stat_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # Ntargets error
    signal_generator_Ntargets = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=[0.01]
    )
    
    total_prediction_Ntargets = signal_generator_Ntargets.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
    flux_norm_total_prediction_Ntargets = total_prediction_Ntargets / IntegratedFlux
    Ntargets_cov = flux_norm_total_prediction_Ntargets.covariance_matrix - stat_cov
    Ntargets_error = np.sqrt(np.diagonal(Ntargets_cov)) / bin_counts
    print('Ntargets error:', Ntargets_error)
    print()

    plot_cov_matrix(Ntargets_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_Ntargets_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_Ntargets_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # POT error
    signal_generator_POT = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=[0.02]
    )
    
    total_prediction_POT = signal_generator_POT.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
    flux_norm_total_prediction_POT = total_prediction_POT / IntegratedFlux
    POT_cov = flux_norm_total_prediction_POT.covariance_matrix - stat_cov
    POT_error = np.sqrt(np.diagonal(POT_cov)) / bin_counts
    print('POT error:', POT_error)
    print()

    plot_cov_matrix(POT_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_POT_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_POT_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # detsys error
    signal_generator_detsys = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=detvar_data,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=None
    )
    
    total_prediction_detsys = signal_generator_detsys.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=True, smooth_detsys_variations=True)
    flux_norm_total_prediction_detsys = total_prediction_detsys / IntegratedFlux
    detsys_cov = flux_norm_total_prediction_detsys.covariance_matrix - stat_cov
    detsys_error = np.sqrt(np.diagonal(detsys_cov)) / bin_counts
    print('detsys error:', detsys_error)
    print()

    plot_cov_matrix(detsys_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_detsys_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_detsys_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # Multisim errors:
    signal_generator_multisim = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        mc_hist_generator_cls = XsecCovarHistGenerator,
        true_var_name=None, 
        signal_query="category_1e1p == 12", 
        uncut_signal_df=rundata["nue"],
        normalization_uncertainty=None
    )
    
    mc_hist_generator = signal_generator_multisim.mc_hist_generator
    
    # GENIE Multisim error
    genie_multisim_cov = (mc_hist_generator.calculate_multisim_uncertainties(multisim_weight_column="weightsGenie")) / IntegratedFlux**2
    genie_multisim_error = np.sqrt(np.diagonal(genie_multisim_cov)) / bin_counts
    print('genie error:', genie_multisim_error)
    print()

    plot_cov_matrix(genie_multisim_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_genie_multisim_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_genie_multisim_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # GENIE Unisim error
    genie_unisim_cov = (mc_hist_generator.calculate_unisim_uncertainties()) / IntegratedFlux**2
    genie_unisim_error = np.sqrt(np.diagonal(genie_unisim_cov)) / bin_counts
    print('genie unisim error:', genie_unisim_error)
    print()

    genie_total_cov = genie_multisim_cov + genie_unisim_cov
    genie_total_error = np.sqrt(np.diagonal(genie_total_cov)) / bin_counts
    print('genie total error:', genie_total_error)
    print()

    plot_cov_matrix(genie_unisim_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_genie_unisim_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_genie_unisim_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # Flux error
    flux_cov = (mc_hist_generator.calculate_multisim_uncertainties(multisim_weight_column="weightsFlux")) / IntegratedFlux**2
    flux_error = np.sqrt(np.diagonal(flux_cov)) / bin_counts
    print('flux error:', flux_error)
    print()

    plot_cov_matrix(flux_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_flux_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_flux_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # Reinteraction error
    reint_cov = (mc_hist_generator.calculate_multisim_uncertainties(multisim_weight_column="weightsReint")) / IntegratedFlux**2
    reint_error = (np.sqrt(np.diagonal(reint_cov)) / bin_counts)
    print('reint error:', reint_error)
    print()

    plot_cov_matrix(reint_cov, binning_def, binning)
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_reint_cov_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/cov_breakdown/{binning_def[0]}_reint_cov_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    # All errors except stats (total systematics error)
    only_syst_error = np.sqrt(np.diagonal(Ntargets_cov + POT_cov + detsys_cov + genie_multisim_cov + genie_unisim_cov + flux_cov + reint_cov)) / bin_counts
    all_except_stat_error = np.sqrt(np.diagonal(total_cov - stat_cov)) / bin_counts
    print("Only syst error:", only_syst_error)
    print()
    print("All except stat error:", all_except_stat_error)
    print()
    
    diff = only_syst_error - all_except_stat_error
    print('diff:', diff)
    print()

    
    # Plotting the fractional errors
    
    fig, ax2 = plt.subplots()
        
    ax2.stairs(genie_total_error, bin_edges, label='GENIE (multisim + unisim)', linestyle='dashdot')
    ax2.stairs(genie_multisim_error, bin_edges, label='GENIE multisim', linestyle='dashdot')
    ax2.stairs(genie_unisim_error, bin_edges, label='GENIE unisim', linestyle='dashdot')
    ax2.stairs(flux_error, bin_edges, label='Flux', linestyle='dashdot')
    ax2.stairs(reint_error, bin_edges, label='Reinteractions', linestyle='dashdot')
    ax2.stairs(detsys_error, bin_edges, label='DetSyst', linestyle='dashdot')
    ax2.stairs(POT_error, bin_edges, label='POT', linestyle='dashdot')
    ax2.stairs(Ntargets_error, bin_edges, label='NTargets', linestyle='dashdot')
    ax2.stairs(stat_error, bin_edges, label='MC Stat', linestyle='dashdot')
        
    ax2.stairs(only_syst_error, bin_edges, label='Total Syst Errors', linestyle='dashdot', color='black')
    ax2.stairs(total_error, bin_edges, label='Total Errors (Syst + Stat)', linestyle='solid', color='black', lw=1.7)
    
    ax2.set_xlabel(binning.variable_tex)
    ax2.set_ylabel('Fractional uncertainty on total predicted events')
    ax2.legend(bbox_to_anchor=(0, 1.03, 1, 0.3), loc="lower left", mode="expand", ncol=2)
    plt.savefig(f'analysis_plots/errors/investigate/total_errors_{data}_{run_combo}_{binning_def[0]}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/errors/investigate/total_errors_{data}_{run_combo}_{binning_def[0]}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
print('Done :)')
