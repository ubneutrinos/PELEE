# Make sure the local settings ntuple path points to the filtered ntuples

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

#RUN = ["5"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"] # use this if using CRT
RUN = ["1","2","3","4a","4b","4c","4d","5"] # for detvars with bnb or for closure test
#RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
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
print()

run_combo = "Run"
for run in RUN:
    run_combo += run
    
# Calculating the integrated flux:

# Zarko Pavlovic, Jun 22 2020
# /pnfs/uboone/persistent/uboonebeam/bnb_gsimple/bnb_gsimple_fluxes_01.09.2019_463_hist/readme.txt

Nominal_UB_XY_Surface = 256.35*233. # cm2
SoftFidSurface = 236. * 210.  # cm2
POTPerSpill = 4997.*5e8
HistoFlux_int = 593641.00 # retrieved from the root file itself by running hEnue_cv->Integral()
IntegratedFlux = (HistoFlux_int * data_pot / POTPerSpill / Nominal_UB_XY_Surface)
print('Integrated flux:', IntegratedFlux)

with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'w') as f:
    f.writelines(["Data POT = ", f"{data_pot} \n"])
    f.writelines(["Integrated flux = ", f"{IntegratedFlux} \n"])
    #f.write("\nCovariance Matrices:\n")

# Calculating the covariance matrix:

selection = "OnePBDT"
preselection = "OneP_new"

all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])
all_sig = all_mc.query("category_1e1p == 12", engine='python')

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")
    #################################################################################
    # Getting the covariance matrix
    
    if data == "nuwro_fd":
        # Getting only xsec cov matrix and stats matrix for NuWro fakedata studies
        
        # Stat cov
        #(included by default - just need to turn flags off for syst errors)
        # Hence, will need to subtract stat_cov when calculating all the individual syst error contributions
        signal_generator = hist.RunHistGenerator(
            rundata,
            binning.copy(),
            data_pot=data_pot,
            selection=selection,
            preselection=preselection,
            sideband_generator=None,
            uncertainty_defaults=None,
            detvar_data=None,
            normalization_uncertainty=None
        )
    
        total_prediction = signal_generator.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
        flux_norm_total_prediction_stat = total_prediction #/ IntegratedFlux
        stat_cov = flux_norm_total_prediction_stat.covariance_matrix
        #stat_error = np.sqrt(np.diagonal(stat_cov)) / bin_counts
        
        # GENIE cov
        
        mc_hist_generator = signal_generator.mc_hist_generator
    
        # GENIE multisim
        genie_cov = (mc_hist_generator.calculate_multisim_uncertainties(multisim_weight_column="weightsGenie")) #/ IntegratedFlux**2
        #genie_error = np.sqrt(np.diagonal(genie_cov)) / bin_counts
        #print('genie error:', genie_error)
        #print()

        # GENIE Unisim
        genie_unisim_cov = (mc_hist_generator.calculate_unisim_uncertainties()) #/ IntegratedFlux**2
        #genie_unisim_error = np.sqrt(np.diagonal(genie_unisim_cov)) / bin_counts
        #print('genie unisim error:', genie_unisim_error)
        #print()
        
        cov = genie_cov + genie_unisim_cov + stat_cov
        
        # Plotting the cov matrix 
        
        if binning_def[1] == None:
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
        
        
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()
    
    else:
    
        # Load detvars
        detvar_data = detsys.make_variations(
        run_numbers=RUN,
        data="bnb",
        binning=binning.copy(),
        selection=selection,
        preselection=preselection,
        use_kde_smoothing=True,
        make_plots=False,
        plot_output_dir= "/exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya/analysis_1e1p/analysis_plots/detsys/",
        enable_detvar_cache=True,
        detvar_cache_dir="/exp/uboone/data/users/mmoudgal/PELEE/detvar_cached_dataframes/",
        extra_selection_query=None,
        show_plots=True,
        #variations=detector_variations,
        #**dl_kwargs,
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
            normalization_uncertainty=[0.01,0.02]
        )
        total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True, add_precomputed_detsys=True, smooth_detsys_variations=True)

        print(f"binning_def[0] bin counts:", total_prediction.bin_counts)
        print()

        flux_norm_total_prediction = total_prediction / IntegratedFlux
        fig, ax = plt.subplots()
        flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False)
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()

        cov = total_prediction.covariance_matrix
        
    print('Covariance matrix for', binning_def[0], ':', cov)
    print()
    
    # writing these to a file
    cov_str = "{"
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            #print(cov[i][j])
            cov_str += f"{cov[i][j]},"
    cov_str = cov_str[:-1] # to remove the last comma
    cov_str += "};"
    # print(cov_str)
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.write(f"\n{label}:\n")
        f.writelines([f"covariance", " = ", f"{cov_str} \n"])

    if closure_test:
            signal_generator_stat = hist.RunHistGenerator(
            rundata,
            binning.copy(),
            data_pot=data_pot,
            selection=selection,
            preselection=preselection,
            sideband_generator=None,
            uncertainty_defaults=None,
            detvar_data=None,
            normalization_uncertainty=None
        )
    
            total_prediction = signal_generator_stat.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
            flux_norm_total_prediction_stat = total_prediction #/ IntegratedFlux
            stat_cov = flux_norm_total_prediction_stat.covariance_matrix
        
            # writing these to a file
            cov_str = "{"
            for i in range(stat_cov.shape[0]):
                for j in range(stat_cov.shape[1]):
                    #print(cov[i][j])
                    cov_str += f"{stat_cov[i][j]},"
            cov_str = cov_str[:-1] # to remove the last comma
            cov_str += "};"
            # print(cov_str)
            with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
                f.writelines([f"statistical covariance", " = ", f"{cov_str} \n"])    
        
    print(f'Finished calculating covariance matrix for {label}.')
    
    #################################################################################
    # Making reco distribution plots
    
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
    else:
        include_multisim_errors = True
        add_precomputed_detsys = True
        show_errorband = True
    
    plotter = rp.RunHistPlotter(signal_generator)
    axes = plotter.plot(
        category_column="category_1e1p",
        include_multisim_errors=include_multisim_errors,
        add_ext_error_floor=False,
        show_data_mc_ratio=show_data_mc_ratio,
        show_chi_square=show_chi_square,
        add_precomputed_detsys=add_precomputed_detsys,
        show_errorband=show_errorband,
    )
    
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=include_multisim_errors,
        add_ext_error_floor=False,
        show_data_mc_ratio=show_data_mc_ratio,
        show_chi_square=show_chi_square,
        add_precomputed_detsys=add_precomputed_detsys,
        show_errorband=show_errorband,
    )

    plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    #################################################################################
    # Getting uB tune signal bin counts
    
    genieUBsig, _ = np.histogram(all_sig[binning_def[0]], bins=binning_def[-1], weights=all_sig["weights"])
    
    # writing these to a file
    counts_str = "{"
    for i in genieUBsig:
        counts_str += f"{i},"
    counts_str = counts_str[:-1] # to remove the last comma
    counts_str += "};"
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.writelines([f"genieUBsig", " = ", f"{counts_str} \n"])
    
    mc_hists = signal_generator.get_mc_hists(
        category_column="category_1e1p",
    )
    
    mc_sig = mc_hists[12].bin_counts
    print(binning_def[0], 'bin counts:', mc_sig)
    print()
    
    # writing these to a file
    counts_str = "{"
    for i in mc_sig:
        counts_str += f"{i},"
    counts_str = counts_str[:-1] # to remove the last comma
    counts_str += "};"
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.writelines([f"Selected sig", " = ", f"{counts_str} \n"])

    print(f'Finished getting the predicted signal bin counts for {label}.')
    print()
    
    #################################################################################
    # Getting the background-subtracted data bin counts
    
    if blinded == False:
        data_hist = signal_generator.get_data_hist()
        data_counts = data_hist.bin_counts
        total_pred_counts = total_prediction.bin_counts
        total_bkg = total_pred_counts - mc_sig
#         bkg_mc_counts = {k: v.bin_counts for k, v in mc_hists.items() if k != 12}
#         bkg_mc_sum = [sum(items) for items in zip(*bkg_mc_counts.values())]
        
#         with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
#             f.writelines([f"{label} bkg sum", " = ", f"{str(bkg_mc_sum)} \n"])
#             f.writelines([f"{label} data counts", " = ", f"{str(data_counts)} \n"])
        
        measure = [a - b for a, b in zip(data_counts, total_bkg)]
        print(binning_def[0], 'bkg-subtracted data counts:', measure)
        print()
        # writing these to a file
        measure_str = "{"
        for i in measure:
            measure_str += f"{i},"
        measure_str = measure_str[:-1] # to remove the last comma
        measure_str += "};"
        print(measure_str)
        print()
        with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
            f.writelines([f"measure", " = ", f"{measure_str} \n"])

    print(f'Finished getting the background-subtracted data bin counts for {label}.')
    print()
    
print('Done :)')
    