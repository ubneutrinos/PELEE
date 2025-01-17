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

keep_vars = [
    "Signal_1e1p", "mc_signal_1e1p", "nu_pdg", "TrueElecIdx", "TrueLeadProtonIdx", "InFV", "HasNoMesons",
    "TrueNElec", "TrueNProt", "TrueDeltaPT", "TrueDeltaAlphaT", "TruePN", "TrueAlpha3D",
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
    
selection = "OnePL_new"
preselection = "OneP_new"
    
# Making the nue events plots
nue_rundata = {}
for key in rundata:
    if key == 'data':
        nue_rundata[key] = rundata[key]
    else:
        nue_rundata[key] = rundata[key].query('abs(nu_pdg) == 12')
        
for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        nue_rundata,
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
    
    plt.savefig(f'plots/investigation/res_topo_nue_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_topo_nue_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )

    plt.savefig(f'plots/investigation/res_int_nue_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_int_nue_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

print("Done making nue plots.")
    
# Making the non-nue events plots
bkg_rundata = {}
for key in rundata:
    if key == 'data':
        bkg_rundata[key] = rundata[key]
    else:
        bkg_rundata[key] = rundata[key].query('abs(nu_pdg) != 12')
        
for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        bkg_rundata,
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
    
    plt.savefig(f'plots/investigation/res_topo_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_topo_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )

    plt.savefig(f'plots/investigation/res_int_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_int_bkg_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

print("Done making bkg plots.")
    
# Filtering just for RES events
filtered_rundata = {}
for key in rundata:
    if key == 'data':
        filtered_rundata[key] = rundata[key]
    else:
        filtered_rundata[key] = rundata[key].query('interaction == 1')
        
for binning_def in vdef.TKI_variables_1e1p:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    signal_generator = hist.RunHistGenerator(
        filtered_rundata,
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
    
    plt.savefig(f'plots/investigation/res_topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_topo_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()

    axes2 = plotter.plot(
        category_column="interaction",
        include_multisim_errors=True,
        add_ext_error_floor=False,
        show_data_mc_ratio=False,
        show_chi_square=False,
    )

    plt.savefig(f'plots/investigation/res_int_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
    plt.savefig(f'plots/investigation/res_int_{preselection}_{selection}_{binning_def[0]}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    
print('Done :)')