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

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
    "nproton", "npion", "npi0", "nelec", "nmuon", "isVtxInFiducial",
    "Sel_1e1p", "sel_1e1p_w_cuts", "RecoElectronCandidateIdx", "RecoLeadProtonCandidateIdx", "InFV_reco",
    "RecoElecPassMomCut", "RecoLeadProtonPassMomCut", "n_reco_tracks", "n_reco_showers",
    "RecoDeltaPT", "RecoDeltaAlphaT", "RecoPN", "RecoAlpha3D", "RecoECal", "Reco_mag_q", "RecoPL",
    "nslice", "selected", "shr_energy_tot_cali", "_opfilter_pe_beam", "_opfilter_pe_veto", "bnbdata", "extdata",
    "CosmicIPAll3D", "hits_ratio", "shrmoliereavg", "subcluster", "trkfit", "tksh_distance",
    "shr_tkfit_nhits_tot", "shr_tkfit_dedx_max", "tksh_angle", "shr_trk_len"
]

#RUN = ["5"]
#RUN = ["1","2","3_nocrt","3_crt","4b","4c","4d","5"]
RUN = ["1","2","3","4c","5"] # for nuwro_fd, no run 4b and 4d available
blinded = False

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="nuwro_fd",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=False,
    use_bdt=False,
    load_lee=False,
    load_nue_tki=True,
    keep_columns=keep_vars,
    blinded=blinded,
    load_crt_vars=False,
    enable_cache=False,
)

print('Loaded data')

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

with open(f'unfolding_inputs_{run_combo}.txt', 'w') as f:
    f.writelines(["Data POT = ", f"{data_pot} \n"])
    f.writelines(["Integrated flux = ", f"{IntegratedFlux} \n"])
    f.write("\nCovariance Matrices:\n")

# Calculating the covariance matrix:

selection = "OnePL_new"
preselection = "OneP_new"

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning,
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
    )
    total_prediction = signal_generator.get_total_prediction(include_multisim_errors=True)
    print(f"binning_def[0] bin counts:", total_prediction.bin_counts)

    flux_norm_total_prediction = total_prediction / IntegratedFlux
    fig, ax = plt.subplots()
    flux_norm_total_prediction.draw_covariance_matrix(ax=ax, as_correlation=False)
    label = binning_def[0].lstrip("Reco")
    plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{run_combo}_{label}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/covariance_matrix_{run_combo}_{label}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    cov = flux_norm_total_prediction.covariance_matrix
    print('Covariance matrix for', binning_def[0], ':', cov)
    
    # writing these to a file
    cov_str = "{"
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            #print(cov[i][j])
            cov_str += f"{cov[i][j]},"
    cov_str = cov_str[:-1] # to remove the last comma
    cov_str += "};"
    # print(cov_str)
    with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
        f.writelines([f"{label}", " = ", f"{cov_str} \n"])

print('Finished calculating covariance matrices.')
        
# Calculating the response matrix:

from microfit import selections as sel
query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])
sel_sig = all_mc.query(query, engine='python')

variables = {
# #    '': {'reco': , 'truth': , 'nbins': , 'bounds': },
#     'Proton Kinetic Energy [GeV]': {'reco': 'trk_energy', 'truth': 'TrueLeadProtonKE', 'nbins': 10, 'bounds': (0, 1), 'range': [[0,1.3],[0,1.3]]},
#     'Proton Momentum [GeV/c]': {'reco': 'mod_trk_p', 'truth': 'TrueLeadProtonModMom', 'nbins': 20, 'bounds': (0, 1.5), 'range': [[0.1,2],[0.1,2]]},
#     'Proton X Momentum [GeV/c]': {'reco': 'trk_px', 'truth': 'TrueLeadProtonMomX', 'nbins': 20, 'bounds': (-1, 1), 'range': [[-1,1.25],[-1,1.25]]},
#     'Proton Y Momentum [GeV/c]': {'reco': 'trk_py', 'truth': 'TrueLeadProtonMomY', 'nbins': 20, 'bounds': (-1.5, 1.5), 'range': [[-1,1.5],[-1,1.5]]},
#     'Proton Z Momentum [GeV/c]': {'reco': 'trk_pz', 'truth': 'TrueLeadProtonMomZ', 'nbins': 20, 'bounds': (-1, 1.5),'range': [[-1,2],[-1,2]]},
#     'Electron Kinetic Energy [GeV]': {'reco': 'shr_energy_cali', 'truth': 'TrueElecKE', 'nbins': 16, 'bounds': (0, 4), 'range': [[0,4.6],[0,4.6]]},
#     'Electron Momentum [GeV/c]': {'reco': 'mod_shr_p', 'truth': 'TrueElecModMom', 'nbins': 20, 'bounds': (0, 5), 'range': [[0,4.6],[0,4.6]]},
#     'Electron X Momentum [GeV/c]': {'reco': 'shr_px', 'truth': 'TrueElecMomX', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1,1],[-1,1]]},
#     'Electron Y Momentum [GeV/c]': {'reco': 'shr_py', 'truth': 'TrueElecMomY', 'nbins': 12, 'bounds': (-1.5, 1.5), 'range': [[-1,1],[-1,1]]},
#     'Electron Z Momentum [GeV/c]': {'reco': 'shr_pz', 'truth': 'TrueElecMomZ', 'nbins': 12, 'bounds': (-1, 5), 'range': [[0,4.6],[0,4.6]]},
    '$\\delta p_T$ [GeV/c]': {'reco': 'RecoDeltaPT', 'truth': 'TrueDeltaPT', 'nbins': 20, 'bounds': (0, 2), 'range': [[0,1.75],[0,1.75]], 'bin_edges': [[0,0.3,1.75],[0,0.3,1.75]], 'bin_edges_1d': [0,0.3,1.75]},
    '$\\delta \\alpha_T$ [degrees]': {'reco': 'RecoDeltaAlphaT', 'truth': 'TrueDeltaAlphaT', 'nbins': 20, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,80,180],[0,80,180]], 'bin_edges_1d': [0,80,180]},
    '$p_n$ [GeV/c]': {'reco': 'RecoPN', 'truth': 'TruePN', 'nbins': 10, 'bounds': (0, 2), 'range': [[0,1.75],[0,1.75]], 'bin_edges': [[0,0.3,1.75],[0,0.3,1.75]], 'bin_edges_1d': [0,0.3,1.75]},
    '$\\alpha_{3D}$ [degrees]': {'reco': 'RecoAlpha3D', 'truth': 'TrueAlpha3D', 'nbins': 10, 'bounds': (0, 180), 'range': [[0,180],[0,180]], 'bin_edges': [[0,90,180],[0,90,180]], 'bin_edges_1d': [0,90,180]},
}

with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
    f.write("\nResponse Matrices:\n")

for k, var in variables.items():
    truth = variables[k]['truth']
    reco = variables[k]['reco']
    bounds = variables[k]['range']
    bin_edges = variables[k]['bin_edges']
    bin_edges_1d = variables[k]['bin_edges_1d']
    
    truth_hist, truth_edges = np.histogram(sel_sig[truth], bins=bin_edges_1d, weights=sel_sig["weights"], range=bounds)
    H, xedges, yedges = np.histogram2d(sel_sig[truth], sel_sig[reco], bins=bin_edges, weights=sel_sig["weights"], range=bounds)
    
    resp = H.T / truth_hist
    
    X, Y = np.meshgrid(xedges,yedges)
    plt.pcolormesh(X, Y, resp, shading='flat') #, norm=LogNorm())
    plt.colorbar()
    plt.xlabel(f'True {k}')
    plt.ylabel(f'Reco {k}')
    
    plt.title(f"Response Matrix")
    label = reco.lstrip("Reco")
    plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{run_combo}_{label}_2bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/response_matrix_{run_combo}_{label}_2bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    print('Response matrix for', k, ':', resp)
    
    # writing these to a file
    resp_str = "{"
    for i in range(resp.shape[0]):
        for j in range(resp.shape[1]):
            resp_str += f"{resp[i][j]},"
    resp_str = resp_str[:-1] # to remove the last comma
    resp_str += "};"
    print(label)
    with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
        f.writelines([f"{label}", " = ", f"{resp_str} \n"])

print('Finished calculating response matrices.')

# Getting the uB tune signal bin counts

with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
    f.write("\nuB Tune Signal Bin Counts:\n")

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
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
        category_column="category_1e1p",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )
    
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )

    plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'analysis_plots/unfolding_inputs/int_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    
    mc_hists = signal_generator.get_mc_hists(
        category_column="category_1e1p",
        include_multisim_errors=True,
    )
    
    mc_sig = mc_hists[12].bin_counts
    print(binning_def[0], 'bin counts:', mc_sig)
    
    label = binning_def[0].lstrip("Reco")
    
    # writing these to a file
    counts_str = "{"
    for i in mc_sig:
        counts_str += f"{i},"
    counts_str = counts_str[:-1] # to remove the last comma
    counts_str += "};"
    with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
        f.writelines([f"{label}", " = ", f"{counts_str} \n"])

print('Finished getting the predicted signal bin counts.')

# Getting the background-subtracted data bin counts

with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
    f.write("\nBackground-subtracted Data Bin Counts:\n")

for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning,
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
    )
    
    mc_hists = signal_generator.get_mc_hists(
        category_column="category_1e1p",
        include_multisim_errors=True,
    )
    
    label = binning_def[0].lstrip("Reco")
    
    if blinded == False:
        data_hist = signal_generator.get_data_hist()
        data_counts = data_hist.bin_counts
        bkg_mc_counts = {k: v.bin_counts for k, v in mc_hists.items() if k != 12}
        bkg_mc_sum = [sum(items) for items in zip(*bkg_mc_counts.values())]
        
#         with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
#             f.writelines([f"{label} bkg sum", " = ", f"{str(bkg_mc_sum)} \n"])
#             f.writelines([f"{label} data counts", " = ", f"{str(data_counts)} \n"])
        
        measure = [a - b for a, b in zip(data_counts, bkg_mc_sum)]
        print(binning_def[0], 'bkg-subtracted data counts:', measure)
        # writing these to a file
        measure_str = "{"
        for i in measure:
            measure_str += f"{i},"
        measure_str = measure_str[:-1] # to remove the last comma
        measure_str += "};"
        print(measure_str)
        print(label)
        with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
            f.writelines([f"{label}", " = ", f"{measure_str} \n"])

print('Finished getting the background-subtracted data bin counts.')

# Calculating the metrics

print("Calculating metrics:")
print()

all_mc = pd.concat([df for k, df in rundata.items() if k!='data'])

all_sig = all_mc['category_1e1p'] == 12
tot_all_sig = np.sum(all_mc.loc[all_sig, 'weights'])
print('Total candidate signal events:', tot_all_sig)

all_predict = all_mc.query(query, engine='python')

is_sig = all_predict['category_1e1p'] == 12
tot_sig = np.sum(all_predict.loc[is_sig, 'weights'])
tot_bkg = np.sum(all_predict.loc[~is_sig, 'weights'])
tot_evt = np.sum(all_predict['weights'])
print('After cuts:')
print('Total signal events:', tot_sig)
print('Total background events:', tot_bkg)
print('Total events:', tot_evt)
print('sig + bkg =', tot_sig+tot_bkg)
print()

efficiency = (tot_sig / tot_all_sig) * 100
purity = (tot_sig / tot_evt) * 100
print('Efficiency:', efficiency)
print('Purity:', purity)
print()

pred_res = all_predict['interaction'] == 1
tot_res = np.sum(all_predict.loc[pred_res, 'weights'])
print('Total resonance events:', tot_res)
pred_qe = all_predict['interaction'] == 0
tot_qe = np.sum(all_predict.loc[pred_qe, 'weights'])
print('Total quasielastic events:', tot_qe)

print('QE/RES =', tot_qe/tot_res)
print('RES/QE =', tot_res/tot_qe)
print('QE/Total =', tot_qe/tot_evt)

with open(f'unfolding_inputs_{run_combo}.txt', 'a') as f:
    f.write("\nMetrics:\n")
    f.writelines(['\nTotal candidate signal events = ', f"{tot_all_sig} \n"])
    f.writelines(["After cuts: \n"])
    f.writelines(['Total signal events = ', f"{tot_sig} \n"])
    f.writelines(['Total background events = ', f"{tot_bkg} \n"])
    f.writelines(['Total events = ', f"{tot_evt} \n"])
    f.writelines(['\nPurity = ', f"{purity} % \n"])
    f.writelines(['Efficiency = ', f"{efficiency} % \n"])
    f.writelines(['\nQE/RES = ', f"{tot_qe/tot_res} \n"])
    f.writelines(['RES/QE = ', f"{tot_res/tot_qe} \n"])
    f.writelines(['QE/Total = ', f"{tot_qe/tot_evt} \n"])

f.close()
print('Done.')