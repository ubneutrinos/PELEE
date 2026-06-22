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
from matplotlib.colors import LogNorm
import scipy.optimize as opt;
from scipy.optimize import curve_fit


def repeated_nom_values(hist):
    # repeat the last bin count
    y = hist.bin_counts
    y = np.append(y, y[-1])
    return y

#RUN = ["1","1A_OT","1B_OT","2","3_crt","3_nocrt","5"]
#RUN = ["1","2","3_crt","3_nocrt","5"]
#RUN = ["1","2","3_crt","3_nocrt","4a","4b","4c","4d","5"]
#RUN = ["1","2","3_crt","5"]
RUN = ["1","2","3_crt"]
#RUN = ["1"]
#RUN = ["1","1A_OT","1B_OT","2","3_crt","3_nocrt"]
blinded = True
#data="nuwro_fd"
data="bnb"
closure_test = False
use_detvar = False

# Choose the selection cuts


selection = "NUMUCRTNP0PI"
#selection = "NUMUCRT0P0PI"
preselection = "NUMU"
#preselection = "NUE"
#election = "NPXSBDT"

#selection = "None"
#preselection = "None"
#preselection = "NUMUCRT"
category_column = "category_fixed"
#category_column = "trk_pdg"
#sig_code = 11
sig_code = None

#signal_query = category_column + f" == {sig_code}"
#signal_query = "category_fixed == 11"
#signal_query = ""

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
    
QUERY_NUE = " and nslice == 1 and selected == 1 and shr_energy_tot_cali > 0.07 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and n_tracks_contained > 0 and CosmicIPAll3D > 10. and trkpid<(0.015*trk_len+0.02) and hits_ratio > 0.50 and shrmoliereavg < 9 and subcluster > 4 and trkfit < 0.65 and tksh_distance < 10.0 and tksh_angle > -0.9 and shr_trk_len < 300. and protonenergy_corr > 0.05 and pi0_score > 0.50 and nonpi0_score > 0.50 and n_showers_contained == 1"
QUERY_NUMU = " and nslice == 1 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and reco_nu_vtx_sce_x > 10.0 and reco_nu_vtx_sce_x < 246.4 and n_tracks_contained > 1 and reco_nu_vtx_sce_y > -101.5 and reco_nu_vtx_sce_y < 101.5 and reco_nu_vtx_sce_z > 10.0 and reco_nu_vtx_sce_z < 986.8 and topological_score > 0.06 and muon_Prot_Ang> -0.9 and n_muons_tot > 0 and n_muons_tot == 1 and n_showers_tot == 0 and n_protons_tot > 0 and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5. and protonenergy_corr > 0.05"

all_mc = pd.concat([df for k, df in rundata.items() if k not in ['data','ext']])
all_nue = pd.concat([df for k, df in rundata.items() if k in ['nue']])
all_sig = all_mc.query("category_fixed == 11"+QUERY_NUE, engine='python')

for binning_def in vdef.variables_ratio:
    # some binning definitions have more than 4 elements,
    # we ignore the last ones for now
    #binning = hist.Binning.from_config(*binning_def[:4])
    binning = hist.Binning.from_config(*binning_def)  # for variable bin sizes
    print(binning_def)
    print()
    label = binning_def[0].lstrip("Reco")
    true_var_name = "True" + label


#hist.binning = 'digitized_bin_cos_trk_theta', np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5]) #, 'digitized_bin_cos_trk_theta', r'Bin numbers for $cos\theta_{p}^{reco}$')

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
        binning,
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        )
    '''    
    signal_generator = hist.RunHistGenerator(
        rundata,
        binning.copy(),
        data_pot=data_pot,
        selection=selection,
        preselection=preselection,
        sideband_generator=None,
        uncertainty_defaults=None,
        detvar_data=None,
        extra_background_fractional_error = None,
        cache_total_covariance = False,
        # mc_hist_generator_cls = XsecCovarHistGenerator,
    # true_var_name=None, 
    #  signal_query=signal_query, 
    # # uncut_signal_df=rundata["nue"],
    # normalization_uncertainty=[0.01,0.02]
        )
    ''' 
    include_multisim_errors = False
    show_errorband = False
    plotter = rp.RunHistPlotter(signal_generator)
    axes = plotter.plot(
        category_column=category_column,
        #signal_category_num=sig_code,
        include_multisim_errors=include_multisim_errors,
        stat_variance_method="data",
        add_ext_error_floor=False,
        show_data_mc_ratio=show_data_mc_ratio,
        show_chi_square=show_chi_square,
        separate_signal = False,
    # show_total_unconstrained=False,
    # add_precomputed_detsys=add_precomputed_detsys,
    # show_errorband=show_errorband,
    # override_data_cov = override_data_cov
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
            facecolor="none",#
            edgecolor=(0.1, 0.1, 0.1),
            )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
        
   # plt.savefig(f'analysis_plots/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.pdf', bbox_inches='tight')
   # plt.savefig(f'analysis_plots/topo_{preselection}_{selection}_{binning_def[0]}_{data}_{run_combo}_{binning_def[1]}bins.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    #H, xedges, yedges = np.histogram2d(all_sig['proton_ke'], (all_sig['protonenergy']-all_sig['proton_ke'])/all_sig['proton_ke'], bins=20, range=[[0,0.5],[-0.5,0.1]])
   # X, Y = np.meshgrid(xedges,yedges)
   # plt.pcolormesh(X, Y, H.T, shading='flat', norm=LogNorm())
   # plt.xlabel(f'true proton_ke')
   # plt.ylabel(f'reco protonenergy')
   # plt.colorbar()
    #plt.savefig(f'analysis_plots/ProtonEnergyrecotrue.pdf', bbox_inches='tight')
  #  plt.show()

  

recov = 'protonenergy'
truev = 'proton_ke'
def func(x, *p):
    p1, p2 = p
    return p1/x + p2
'''
ACCEPTANCE_sophie = 'isVtxInFiducial == 1 and ccnc==0 and nu_pdg==12 and npi0==0 and npion==0 and elec_e>0.03051'
#ACCEPTANCE_sophie = 'isVtxInFiducial == 1 and ccnc==0 and nu_pdg==14 and npi0==0 and npion==0 and muon_e>0.13566'
QUERY_NUE = "nslice == 1 and selected == 1 and shr_energy_tot_cali > 0.07 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and n_tracks_contained > 0 and CosmicIPAll3D > 10. and trkpid<(0.015*trk_len+0.02) and hits_ratio > 0.50 and shrmoliereavg < 9 and subcluster > 4 and trkfit < 0.65 and tksh_distance < 10.0 and tksh_angle > -0.9 and shr_trk_len < 300. and protonenergy_corr > 0.05 and pi0_score > 0.50 and nonpi0_score > 0.50 and n_showers_contained == 1"
sel = ACCEPTANCE_sophie+' and '+QUERY_NUE
#sel = ACCEPTANCE_sophie+' and '+NUMUSELCRTNP0PI
# need to make sure cut on proton energy not included in query!
sel = sel.replace(" and protonenergy_corr > 0.05","")

goodreco = 'trk_pdg==2212 and trk_pur>0.5 and trk_cmp>0.5'
selgr = sel+' and '+goodreco


xval = []
yval = []
rng = np.linspace(0.01,0.20,20)
for i in range(0,len(rng)-1):
    low = rng[i]
    high = rng[i+1]
    mid = low+(high-low)/2
    extracut = (' and %s>%f and %s<%f'%(recov,low,recov,high))
    mysel = selgr+extracut
    ba = all_nue.query(mysel,engine='python')[truev]-all_nue.query(mysel,engine='python')[recov]
   # ba = samples['mc'].query(mysel,engine='python')[truev]-samples['mc'].query(mysel,engine='python')[recov]
    xval.append(mid)
    yval.append(np.median(ba))
    print(len(ba))


coeff, var_matrix = curve_fit(func, xval, yval, p0=(0.0005,-0.000))
curvebins = np.linspace(rng[0],rng[-1],200)
hist_fit = func(curvebins, *coeff)

fig = plt.figure(figsize=(6, 6))
plt.errorbar(x=xval,y=yval,fmt='r*',label='nue MC')
plt.plot(curvebins, hist_fit, label='Fitted data\n %.5f + %.5f/x'%(coeff[1],coeff[0]))
plt.xlim(rng[0],rng[-1])
plt.ylim(0,0.1)
plt.grid()
plt.xlabel('reco KE [GeV]')
plt.ylabel('true - reco KE [GeV]')
plt.legend()
plt.tight_layout()
plt.show()
print('p1=%.6f, p2=%.6f'%(coeff[0],coeff[1]))
print (xval)
print (yval)

'''
ACCEPTANCE_sophie = 'isVtxInFiducial == 1 and ccnc==0 and nu_pdg==14 and npi0==0 and npion==0 and muon_e>0.13566'
#QUERY_NUMU = "nslice == 1 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and reco_nu_vtx_sce_x > 10.0 and reco_nu_vtx_sce_x < 246.4 and n_tracks_contained > 1 and reco_nu_vtx_sce_y > -101.5 and reco_nu_vtx_sce_y < 101.5 and reco_nu_vtx_sce_z > 10.0 and reco_nu_vtx_sce_z < 986.8 and topological_score > 0.06 and muon_Prot_Ang> -0.9 and n_muons_tot > 0 and n_muons_tot == 1 and n_showers_tot == 0 and n_protons_tot > 0 and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5. and protonenergy_corr > 0.05"
#ACCEPTANCE_sophie = 'isVtxInFiducial == 1 '
#QUERY_NUMU = "nslice == 1 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and reco_nu_vtx_sce_x > 10.0 and reco_nu_vtx_sce_x < 246.4 and n_tracks_contained > 1 and reco_nu_vtx_sce_y > -101.5 and reco_nu_vtx_sce_y < 101.5 and reco_nu_vtx_sce_z > 10.0 and reco_nu_vtx_sce_z < 986.8 and topological_score > 0.06 and n_muons_tot > 0 and n_muons_tot == 1 and n_showers_tot == 0 and n_protons_tot > 0 and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5."
QUERY_NUMU = 'nslice == 1 and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1) and reco_nu_vtx_sce_x > 10 and reco_nu_vtx_sce_x < 246.4 and reco_nu_vtx_sce_y > -101.5 and reco_nu_vtx_sce_y < 101.5 and reco_nu_vtx_sce_z > 10 and reco_nu_vtx_sce_z < 986.8 and topological_score > 0.06  and n_muons_tot > 0 and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5. and n_muons_tot == 1 and n_showers_tot == 0 and n_protons_tot > 0' +'and n_tracks_contained > 1' +'and muon_Prot_Ang> -0.9'
sel = ACCEPTANCE_sophie+' and '+QUERY_NUMU
# need to make sure cut on proton energy not included in query!
#sel = sel.replace(" and protonenergy_corr > 0.05","")

goodreco = 'trk_pdg_prot==2212 and trk_pur_prot>0.5 and trk_cmp_prot>0.5'
selgr = sel +' and '+goodreco
print(selgr)


xval = []
yval = []
rng = np.linspace(0.01,0.20,20)
for i in range(0,len(rng)-1):
    low = rng[i]
    high = rng[i+1]
    mid = low+(high-low)/2
    extracut = (' and %s>%f and %s<%f'%(recov,low,recov,high))
    mysel = selgr+extracut
    ba = all_mc.query(mysel,engine='python')[truev]-all_mc.query(mysel,engine='python')[recov]
    print(len(ba))
    xval.append(mid)
    yval.append(np.median(ba))


coeff, var_matrix = curve_fit(func, xval, yval, p0=(0.0005,-0.000))
curvebins = np.linspace(rng[0],rng[-1],200)
hist_fit = func(curvebins, *coeff)

fig = plt.figure(figsize=(6, 6))
plt.errorbar(x=xval,y=yval,fmt='r*',label='MC')
plt.plot(curvebins, hist_fit, label='Fitted data\n %.5f + %.5f/x'%(coeff[1],coeff[0]))
##plt.xlim(rng[0],rng[-1])
#plt.ylim(0,0.1)
plt.grid()
plt.xlabel('reco KE [GeV]')
plt.ylabel('true - reco KE [GeV]')
plt.legend()
plt.tight_layout()
plt.show()
print('p1=%.6f, p2=%.6f'%(coeff[0],coeff[1]))
print (xval)
print (yval)

H, xedges, yedges = np.histogram2d(all_mc.query(selgr,engine='python')['proton_ke'], (all_mc.query(selgr,engine='python')['protonenergy']-all_mc.query(selgr,engine='python')['proton_ke'])/all_mc.query(selgr,engine='python')['proton_ke'], bins=20, range=[[0,0.5],[-0.5,0.1]])
X, Y = np.meshgrid(xedges,yedges)
plt.pcolormesh(X, Y, H.T, shading='flat', norm=LogNorm())
plt.xlabel(f'true proton_ke')
plt.ylabel(f'reco protonenergy')
plt.colorbar()
plt.savefig(f'analysis_plots/ProtonEnergyrecotrue.pdf', bbox_inches='tight')
plt.show()

print('Done :)')