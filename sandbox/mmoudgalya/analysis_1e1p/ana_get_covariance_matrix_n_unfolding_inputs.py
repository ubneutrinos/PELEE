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
#from microfit.statistics import chi_square as chi_square_func

def chi_square_func(observation, expectation, total_covar):
    """
    Calculate the chi-square value for a given observation, expectation, and total covariance.

    Parameters:
    observation (np.ndarray): The observed data.
    expectation (np.ndarray): The expected data.
    systematic_covariance (np.ndarray): The systematic covariance matrix.

    Returns:
    float: The chi-square value.

    """

    # TODO: Add a check to catch if the prediction/data histograms are empty.
    # CT has seen a crash caused by this that was annoying to debug

    n = observation
    mu = expectation

    covar_inv = np.linalg.inv(total_covar)
    chi2 = np.dot(n - mu, np.dot(covar_inv, n - mu))
    return chi2

def plot_cov_matrix(cov, binning_def, binning, title):

    if binning_def[1] is None:
        edges = binning.bin_edges
    else:
        edges = binning.bin_centers
            
    X, Y = np.meshgrid(edges,edges)
    max_val = np.max(np.abs(cov))
    plt.pcolormesh(X, Y, cov, cmap="RdBu_r", vmin=-max_val, vmax=max_val, shading='flat') #, norm=LogNorm())
    plt.colorbar(label = title)
    plt.xlabel(f'{binning.variable_tex}')
    plt.ylabel(f'{binning.variable_tex}')
    plt.title(f"{title} Matrix")

def repeated_nom_values(hist):
    # repeat the last bin count
    y = hist.bin_counts
    y = np.append(y, y[-1])
    return y

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons", "TrueNElec", "TrueNProt", 
    "TrueDeltaPT_1e1p", "TrueDeltaAlphaT_1e1p", "TruePN_1e1p", "TrueAlpha3D_1e1p",
    "TrueLeadProtonKE_1e1p", "TrueLeadProtonModMom_1e1p", "TrueLeadProtonE_1e1p", "TrueLeadProtonMomX_1e1p", "TrueLeadProtonMomY_1e1p", "TrueLeadProtonMomZ_1e1p",
    "TrueElecKE_1e1p", "TrueElecModMom_1e1p", "TrueElecE_1e1p", "TrueElecMomX_1e1p", "TrueElecMomY_1e1p", "TrueElecMomZ_1e1p",

    "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "mc_KE_prot", "mc_p_prot", "mc_E_prot", "mc_px_prot", "mc_py_prot", "mc_pz_prot",
    "mc_KE_elec", "mc_p_elec", "mc_E_elec", "mc_px_elec", "mc_py_elec", "mc_pz_elec",

    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT_1e1p", "RecoDeltaAlphaT_1e1p", "RecoPN_1e1p", "RecoAlpha3D_1e1p", #"RecoECal_1e1p", "Reco_mag_q_1e1p", "RecoPL_1e1p",
    "RecoLeadProtonKE_1e1p", "RecoLeadProtonModMom_1e1p", "RecoLeadProtonMomX_1e1p", "RecoLeadProtonMomY_1e1p", "RecoLeadProtonMomZ_1e1p",
    "RecoElecE", "RecoElecModMom", "RecoElecMomX", "RecoElecMomY", "RecoElecMomZ",

    "RecoLeadProton_trk_len", "RecoLeadProton_trk_trunk_dEdx_y", "RecoLeadProton_dEdx_y_per_trklen",
    "RecoLeadProtonCandidate_trk_pid", "RecoElectronCandidate_shr_pid", "RecoElectron_conversion_dist",

    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial", "elec_e", "proton_ke",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "trkshrhitdist2", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len", "reco_e",
    "trkpid", "trk_len", "n_showers_contained", "protonenergy_corr", "n_tracks_contained",
    "pi0_radlen1", "pi0_radlen2", "pi0_score", "nonpi0_score", "bkg_score", "trk_id",

    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", #"RecoECal", "Reco_mag_q", "RecoPL",
    "RecoLeadProtonKE", "RecoLeadProtonModMom", "RecoLeadProtonMomX", "RecoLeadProtonMomY", "RecoLeadProtonMomZ",

    # "InFV_1muNp", "TrueMuonIdx_1muNp", "TrueLeadProtonIdx_1muNp", "TrueNProt_1muNp", "TrueFSPions_1muNp", "Signal_1mu1p", 
    # "TrueDeltaPT_1mu1p", "TrueDeltaAlphaT_1mu1p", "TruePN_1mu1p", "TrueAlpha3D_1mu1p",
    # "TrueLeadProtonE_1muNp", "TrueLeadProtonMomX_1muNp", "TrueLeadProtonMomY_1muNp", "TrueLeadProtonMomZ_1muNp",
    # "TrueMuonE_1muNp", "TrueMuonMomX_1muNp", "TrueMuonMomY_1muNp", "TrueMuonMomZ_1muNp",

    # "sel_CC1p0pi", "InFV_reco_1muNp", "MuonCandidateIdx_1muNp", "LeadProtonIdx_1muNp", "LeadProtonPassMomentumCut_1muNp", "PFPStartsInPCV_1muNp", "PassTopoScoreCut_1muNp",
    # "PassNuMuCCSelection_1muNp", "NoRecoShowers_1muNp", "MuonContained_1muNp", "PassMuonMomentumCut_1muNp", "PassMuonQualCut_1muNp",
    # "LeadProtonPassMomentumCut_1muNp", "NProtons_1muNp",
    # "RecoDeltaPT_1mu1p", "RecoDeltaAlphaT_1mu1p", "RecoPN_1mu1p", "RecoAlpha3D_1mu1p", "RecoECal_1mu1p", "RecoPL_1mu1p",
    # "RecoLeadProtonE_1muNp", "RecoLeadProtonMomentum_1muNp", "RecoLeadProtonMomX_1muNp", "RecoLeadProtonMomY_1muNp", "RecoLeadProtonMomZ_1muNp", 
    # "RecoMuonE_1muNp", "RecoMuonMomentum_1muNp", "RecoMuonMomX_1muNp", "RecoMuonMomY_1muNp", "RecoMuonMomZ_1muNp",
]

keep_vars_detsys = keep_vars + ["ccnc", "nu_pdg",]

#RUN = ["3"]
#RUN = ["1","2","3_nocrt","3_crt","4a","4b","4c","4d","5"] # use this if using CRT
#RUN = ["1","2","3","4a","4b","4c","4d","5","1A_OT","1B_OT"] # for detvars with bnb or for closure test
RUN = ["1"] # for nuwro_fd
blinded = False
data="nuwro_fd"
#data="bnb"
closure_test = False
use_detvar = False

# Choose the selection cuts

# selection = "OnePBDT"
# preselection = "OneP_new"
# category_column="category_1e1p"
# sig_code = 12

# selection = "None"
# preselection = "None"
selection = "OneP_NPBDTXS"
preselection = "NUE"
category_column="category_1e1p_tki"
sig_code = 12

signal_query = category_column + f" == {sig_code}"

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data=data,
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=True,
    load_lee=False,
    load_numu_tki=False,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=blinded,
    load_crt_vars=False,
    enable_cache=True,
)

if data == "nuwro_fd":
    # Load in original NuWro FD and high-stats nue NuWro as MC to extract bin counts to be used as data bin counts
    nw_rundata, nw_mc_weights, nw_data_pot = dl.load_runs(
        ["1_nuwrofd"],
        data="nuwro_fd",
        loadpi0variables=False,
        loadshowervariables=True,
        loadrecoveryvars=True,
        loadsystematics=True,
        numupresel=False,
        loadnumuvariables=False,
        use_bdt=True,
        load_lee=False,
        load_numu_tki=False,
        load_nue_tki=True,
        keep_columns=keep_vars,
        blinded=True,
        load_crt_vars=False,
        enable_cache=True,
    )

    for k, df in nw_rundata.items():
        query_nue_cc = "abs(nu_pdg)==12 and ccnc==0"
        query_nue_nc = "abs(nu_pdg)==12 and ccnc==1"
        if k == "mc":
            nw_rundata[k] = nw_rundata[k].query(f"~({query_nue_cc})", engine='python') # Remove the cc nues (automatically done in data_loading.load_runs() but repeated here as a safety net)
        elif k == "nue":
            nw_rundata[k] = nw_rundata[k].query(f"~({query_nue_nc})", engine='python') # Remove the nc nues from the high-stats nue Nuwro sample
        else:
            nw_rundata[k] = None # EXT and dirt not needed for fake data studies. Data dataframe is not used here.

    nuwrofd_data = pd.concat([df for k, df in nw_rundata.items() if k in ["mc","nue"]])
    rundata["data"] = nuwrofd_data
    data_truth = rundata["data"].query(signal_query, engine='python')
    backup_dfs = {}
    for k, df in rundata.items():
        if k in ["drt", "ext"]:
            backup_dfs[k] = df
            rundata[k] = None

print('Loaded data')
print()

###########################################################################################
# If we want to filter the n-tuples for certain studies
# filtered_rundata = {}  
# for key, df in rundata.items():
#     print(key)
#     if rundata[key] is None:
#         print(key)
#         filtered_rundata[key] = None
#     elif key in ["data"] and blinded:
#         filtered_rundata["data"] = None
#     else:
#         filtered_rundata[key] = df.query(signal_query, engine='python')

# rundata = {}
# rundata = filtered_rundata.copy()
###########################################################################################

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
    f.writelines([f"Data POT = {data_pot} \n"])
    f.writelines([f"Integrated flux = {IntegratedFlux} \n"])
    #f.write("\nCovariance Matrices:\n")

# Calculating the covariance matrix:


# Needed to apply 20% flat uncertainty to bkg detvars
extra_selection_query = "(abs(nu_pdg) == 12)"
misc_background_query = "not (abs(nu_pdg) == 12)"
misc_background_error_frac = 0.2
extra_background_fractional_error = {misc_background_query: misc_background_error_frac}

# for k, df in rundata.items():
#     print(f"{k} \n: {rundata[k]}")

all_mc = pd.concat([df for k, df in rundata.items() if k not in ['data','ext']])
all_sig = all_mc.query(signal_query, engine='python')

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")
    true_var_name = "True" + label
    #################################################################################
    # Getting the covariance matrix
    
    plot_cov = None
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
            mc_hist_generator_cls = XsecCovarHistGenerator,
            true_var_name=None, 
            signal_query=signal_query, 
            uncut_signal_df=rundata["nue"],
            normalization_uncertainty=None
        )
    
        total_prediction = signal_generator.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
        bin_counts = total_prediction.bin_counts
        flux_norm_total_prediction_stat = total_prediction #/ IntegratedFlux
        pred_stat_cov = flux_norm_total_prediction_stat.covariance_matrix
        
        # Data stat cov
        data_hist = signal_generator.get_data_hist()
        data_counts = data_hist.bin_counts
        data_stat_cov = np.diag(data_counts)

        # bnb_pot = 1.30048e+21 
        # data_stat_cov *= data_pot / bnb_pot

        # GENIE multisim and unisim covs
        mc_hist_generator = signal_generator.mc_hist_generator
        genie_multisim_cov = (mc_hist_generator.calculate_multisim_uncertainties(multisim_weight_column="weightsGenie"))
        genie_unisim_cov = (mc_hist_generator.calculate_unisim_uncertainties())
    
        cov = genie_multisim_cov + genie_unisim_cov + pred_stat_cov + data_stat_cov
        plot_cov = genie_multisim_cov + genie_unisim_cov + pred_stat_cov # error band on reco distr. plot should include only GENIE error and MC stats error

        # Plotting the cov matrix 
        
        # if binning_def[1] is None:
        #     edges = binning.bin_edges
        # else:
        #     edges = binning.bin_centers
            
        # X, Y = np.meshgrid(edges,edges)
        # max_val = np.max(np.abs(cov))
        # plt.pcolormesh(X, Y, cov, cmap="RdBu_r", vmin=-max_val, vmax=max_val, shading='flat') #, norm=LogNorm())
        # plt.colorbar(label = "Covariance")
        # plt.xlabel(f'{binning.variable_tex}')
        # plt.ylabel(f'{binning.variable_tex}')
        # plt.title(f"Covariance Matrix")
            
        plot_cov_matrix(cov, binning_def, binning, "Covariance")
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()

        # Plotting fractional covariance matrices

        fig, ax = plt.subplots()
        total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False, as_fractional=True)
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_MCstat_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_MCstat_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()

        genie_multisim_frac_cov = genie_multisim_cov / np.outer(bin_counts, bin_counts)
        plot_cov_matrix(genie_multisim_frac_cov, binning_def, binning, "Fractional Covariance")
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_geniemultisim_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_geniemultisim_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()

        genie_unisim_frac_cov = genie_unisim_cov / np.outer(bin_counts, bin_counts)
        plot_cov_matrix(genie_unisim_frac_cov, binning_def, binning, "Fractional Covariance")
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_genieunisim_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_genieunisim_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()
    
    else:

        if use_detvar:
            # Load detvars
            detvar_data = detsys.make_variations(
            run_numbers=RUN,
            data="bnb",
            binning=binning.copy(),
            selection=selection,
            preselection=preselection,
            use_kde_smoothing=False,
            make_plots=True,
            plot_output_dir= "/exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya/analysis_1e1p/analysis_plots/detsys/",
            enable_detvar_cache=True,
            detvar_cache_dir="/exp/uboone/data/users/mmoudgal/PELEE/detvar_cached_dataframes/",
            extra_selection_query=extra_selection_query,
            show_plots=False,
            loadpi0variables=False,
            loadshowervariables=True,
            loadrecoveryvars=True,
            loadsystematics=True,
            numupresel=False,
            loadnumuvariables=False,
            use_bdt=True,
            load_lee=False,
            load_numu_tki=False,
            load_nue_tki=True,
            keep_columns=keep_vars_detsys,
            blinded=blinded,
            load_crt_vars=False,
            enable_cache=True,
            )
        else:
            detvar_data = None

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
            extra_background_fractional_error = extra_background_fractional_error,
            mc_hist_generator_cls = XsecCovarHistGenerator,
            true_var_name=None, 
            signal_query=signal_query, 
            uncut_signal_df=rundata["nue"],
            normalization_uncertainty=[0.01,0.02]
        )
        total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True, add_precomputed_detsys=use_detvar, smooth_detsys_variations=use_detvar)
        cov = total_prediction.covariance_matrix
    
        if blinded == False:
            data_hist = signal_generator.get_data_hist()
            data_counts = data_hist.bin_counts
            data_stat_cov = np.diag(data_counts)
            cov += data_stat_cov

        flux_norm_total_prediction = total_prediction / IntegratedFlux
        fig, ax = plt.subplots()
        flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False)
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()

        fig2, ax2 = plt.subplots()
        flux_norm_total_prediction.draw_covariance_matrix(ax=ax2, as_correlation=False, as_fractional=True)
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
        plt.savefig(f'analysis_plots/unfolding_inputs/frac_covariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
        plt.show()
        plt.clf()
        
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
        f.writelines([f"covariance = {cov_str} \n"])
    
    if data == "nuwro_fd":
        # Getting the NuWro FD truth prediction
        nuwroFDtruth, _ = np.histogram(data_truth[true_var_name], bins=binning.bin_edges, weights=data_truth["weights"])
        
        # writing these to a file
        counts_str = "{"
        for i in nuwroFDtruth:
            counts_str += f"{i},"
        counts_str = counts_str[:-1] # to remove the last comma
        counts_str += "};"
        with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
            f.writelines([f"nuwroFDtruth = {counts_str} \n"])

    # cov_str = "{"
    # for i in range(data_stat_cov.shape[0]):
    #     for j in range(data_stat_cov.shape[1]):
    #         #print(cov[i][j])
    #         cov_str += f"{data_stat_cov[i][j]},"
    # cov_str = cov_str[:-1] # to remove the last comma
    # cov_str += "};"
    # # print(cov_str)
    # with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
    #     f.writelines([f"data stat covariance", " = ", f"{cov_str} \n"])

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
            mc_hist_generator_cls = XsecCovarHistGenerator,
            true_var_name=None, 
            signal_query=signal_query, 
            uncut_signal_df=rundata["nue"],
            normalization_uncertainty=None
        )
    
            total_prediction = signal_generator_stat.get_total_prediction(include_multisim_errors=False, add_precomputed_detsys=False, smooth_detsys_variations=False)
            flux_norm_total_prediction = total_prediction / IntegratedFlux
            fig, ax = plt.subplots()
            flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False)
            plt.savefig(f'analysis_plots/unfolding_inputs/predstatcovariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
            plt.savefig(f'analysis_plots/unfolding_inputs/predstatcovariance_matrix_{data}_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
            plt.show()
            plt.clf()
            
            flux_norm_total_prediction_stat = total_prediction #/ IntegratedFlux
            pred_stat_cov = flux_norm_total_prediction_stat.covariance_matrix
        
            # writing these to a file
            cov_str = "{"
            for i in range(pred_stat_cov.shape[0]):
                for j in range(pred_stat_cov.shape[1]):
                    #print(cov[i][j])
                    cov_str += f"{pred_stat_cov[i][j]},"
            cov_str = cov_str[:-1] # to remove the last comma
            cov_str += "};"
            # print(cov_str)
            with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
                f.writelines([f"statistical covariance = {cov_str} \n"])    
        
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
        uncertainties = np.sqrt(np.diagonal(plot_cov))
        uncertainties = np.append(uncertainties, uncertainties[-1])
        override_data_cov = True
    else:
        include_multisim_errors = True
        add_precomputed_detsys = use_detvar
        show_errorband = True
        override_data_cov = False
    
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
    
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    # axes2 = plotter.plot(
    #     category_column="interaction",
    #     signal_category_num=0,
    #     include_multisim_errors=include_multisim_errors,
    #     add_ext_error_floor=False,
    #     show_data_mc_ratio=show_data_mc_ratio,
    #     show_chi_square=show_chi_square,
    #     show_total_unconstrained=False,
    #     add_precomputed_detsys=add_precomputed_detsys,
    #     show_errorband=show_errorband,
    # )

    # if data == "nuwro_fd":
    #     ax = axes2[0]
    #     ax.fill_between(
    #         binning.bin_edges,
    #         np.clip(repeated_nom_values(total_prediction) - uncertainties, 0, None),
    #         repeated_nom_values(total_prediction) + uncertainties,
    #         alpha=0.5,
    #         step="post",
    #         label="My uncertainty",
    #         #color="gray",
    #         linewidth=0.0,
    #         hatch="///////",
    #         facecolor="none",
    #         edgecolor=(0.1, 0.1, 0.1),
    #     )
    #     ax.set_ylim(0, ax.get_ylim()[1] * 1.5)

    # plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    # plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    # plt.show()
    # plt.clf()
    
    #################################################################################
    # Getting uB tune signal bin counts
    
    genieUBsig, _ = np.histogram(all_sig[true_var_name], bins=binning.bin_edges, weights=all_sig["weights"])
    
    # writing these to a file
    counts_str = "{"
    for i in genieUBsig:
        counts_str += f"{i},"
    counts_str = counts_str[:-1] # to remove the last comma
    counts_str += "};"
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.writelines([f"genieUBsig = {counts_str} \n"])
    
    mc_hists = signal_generator.get_mc_hists(
        category_column=category_column,
    )
    
    mc_sig = mc_hists[sig_code].bin_counts
    print(binning_def[0], 'bin counts:', mc_sig)
    print()
    
    # writing these to a file
    counts_str = "{"
    for i in mc_sig:
        counts_str += f"{i},"
    counts_str = counts_str[:-1] # to remove the last comma
    counts_str += "};"
    with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
        f.writelines([f"Selected sig = {counts_str} \n"])

    print(f'Finished getting the predicted signal bin counts for {label}.')
    print()
    
    #################################################################################
    # Getting the background-subtracted data bin counts
    
    if blinded == False:
        data_hist = signal_generator.get_data_hist()
        data_counts = data_hist.bin_counts
        total_pred_counts = total_prediction.bin_counts
        total_bkg = total_pred_counts - mc_sig
#         bkg_mc_counts = {k: v.bin_counts for k, v in mc_hists.items() if k != sig_code}
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
        chi_square = chi_square_func(
                np.array(measure),
                mc_sig,
                cov,
            )
        with open(f'unfolding_inputs_{data}_{run_combo}.txt', 'a') as f:
            f.writelines([f"measure = {measure_str} \n"])
            f.writelines([f"chi2 = {chi_square} \n"])

    print(f'Finished getting the background-subtracted data bin counts for {label}.')
    print()
    
print('Done :)')
    