import localSettings as ls
from load_data_run123 import *
from unblinding_far_sideband import *
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit

from sys_functions import *

INTERCEPT = 0.0
SLOPE = 0.83

CCNCSEL = "n_showers_contained >= 2 and nslice == 1 and contained_fraction > 0.4 and shr_energy_tot_cali > 0.07"\
          " and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)"\
          " and CosmicIPAll3D > 30. and shr_trk_sce_end_y < 105 and hits_y > 80 and shr_score < 0.25 and topological_score > 0.1"\
          " and slnunhits/slnhits>0.1"#the others are in Cosmic background category
CCSEL = CCNCSEL + ' and ((trkpid>0.6 and n_tracks_contained>0 and n_tracks_contained == n_tracks_tot) or (n_tracks_contained < n_tracks_tot))'
NCSEL = CCNCSEL + ' and (trkpid<0.6 or n_tracks_contained==0) and (n_tracks_contained == n_tracks_tot)'

var2label = {
    'pi0_e': 'True Pi0 Energy [GeV]',
    'leadPi0_uz': r'True Pi0 $\cos(\theta)$',
}

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

class ccncpi0_analysis(object):
    def __init__(self, ntuple_path, fold = "nuselection", tree = "NeutrinoSelectionFilter", fcc1 = "prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2", fcc3 = "prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2", fnc1 = "prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2", fnc3 = "prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2"):
        self.ucc1 = uproot.open(ntuple_path+fcc1+".root")[fold][tree]
        self.unc1 = uproot.open(ntuple_path+fnc1+".root")[fold][tree]
        self.ucc3 = uproot.open(ntuple_path+fcc3+".root")[fold][tree]
        self.unc3 = uproot.open(ntuple_path+fnc3+".root")[fold][tree]
        self.uproot_v = [self.ucc1, self.ucc3, self.unc1, self.unc3]
        
        self.vardict = get_variables()
        self.variables = self.vardict['VARIABLES']+\
                    self.vardict['NUEVARS']+\
                    self.vardict['PI0VARS']+\
                    self.vardict['MCFVARS']+\
                    ['weightSplineTimesTune']+\
                    ['pi0_mcgamma0_pz','pi0_mcgamma1_pz','pi0_mcgamma0_e','pi0_mcgamma1_e']+\
                    ["weightsGenie", "weightSpline"]+\
                    ["weightsFlux", "weightSpline"]
        self.variables = list(set(self.variables))

        self.ccr1 = self.ucc1.pandas.df(self.variables, flatten=False)
        self.ncr1 = self.unc1.pandas.df(self.variables, flatten=False)
        self.ccr3 = self.ucc3.pandas.df(self.variables, flatten=False)
        self.ncr3 = self.unc3.pandas.df(self.variables, flatten=False)
        self.df_v = [self.ccr1, self.ccr3, self.ncr1, self.ncr3]
        
        self.first_initialisation()
        self.set_POT()
        self.pi0s = pd.concat([self.ccr1, self.ccr3, self.ncr1, self.ncr3], ignore_index=True)
        self.set_selections()
        
    def first_initialisation(self):
        for i,df in enumerate(self.df_v):
            up = self.uproot_v[i]
            process_uproot(up,df)
            process_uproot_eta(up,df)
            process_uproot_ccncpi0vars(up,df)
            #process_uproot_recoveryvars(up,df)

            df['subcluster'] = df['shrsubclusters0'] + df['shrsubclusters1'] + df['shrsubclusters2']
            df['trkfit'] = df['shr_tkfit_npointsvalid'] / df['shr_tkfit_npoints']
            # and the 2d angle difference
            df['anglediff_Y'] = np.abs(df['secondshower_Y_dir']-df['shrclusdir2'])

            df['shr_tkfit_nhits_tot'] = (df['shr_tkfit_nhits_Y']+df['shr_tkfit_nhits_U']+df['shr_tkfit_nhits_V'])
            df.loc[:,'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_Y']
            df.loc[(df['shr_tkfit_nhits_U']>df['shr_tkfit_nhits_Y']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_U']
            df.loc[(df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_Y']) & (df['shr_tkfit_nhits_V']>df['shr_tkfit_nhits_U']),'shr_tkfit_dedx_max'] = df['shr_tkfit_dedx_V']

            df['asymm'] = np.abs(df['pi0_energy1_Y']-df['pi0_energy2_Y'])/(df['pi0_energy1_Y']+df['pi0_energy2_Y'])
            df['pi0energy'] = 134.98 * np.sqrt( 2. / ( (1-(df['asymm'])**2) * (1-df['pi0_gammadot']) ) )
            df['pi0momentum'] = np.sqrt(df['pi0energy']**2 - 134.98**2)
            df['pi0beta'] = df['pi0momentum']/df['pi0energy']
            df['pi0thetacm'] = df['asymm']/df['pi0beta']
            df['pi0momx'] = df['pi0_energy2_Y']*df['pi0_dir2_x'] + df['pi0_energy1_Y']*df['pi0_dir1_x']
            df['pi0momy'] = df['pi0_energy2_Y']*df['pi0_dir2_y'] + df['pi0_energy1_Y']*df['pi0_dir1_y']
            df['pi0momz'] = df['pi0_energy2_Y']*df['pi0_dir2_z'] + df['pi0_energy1_Y']*df['pi0_dir1_z']
            df['pi0energyraw'] = df['pi0_energy2_Y'] + df['pi0_energy1_Y']
            df['pi0energyraw_corr'] = df['pi0energyraw'] / SLOPE
            df['pi0momanglecos'] = df['pi0momz'] / df['pi0energyraw']
            df['epicospi'] = df['pi0energy'] * (1-df['pi0momanglecos'])
            df['boost'] = (np.abs(df['pi0_energy1_Y']-df['pi0_energy2_Y'])/SLOPE)/(np.sqrt((df['pi0energy'])**2-134.98**2))
            df['pi0_mass_Y_corr'] = df['pi0_mass_Y']/SLOPE
            df['pi0energymin'] = 134.98 * np.sqrt( 2. / (1-df['pi0_gammadot']) )
            df['pi0energyminratio'] = df['pi0energyraw_corr'] / df['pi0energymin']

            # define some energy-related variables
            df["reco_e"] = (df["shr_energy_tot_cali"] + INTERCEPT) / SLOPE + df["trk_energy_tot"]

            # and a way to filter out data
            df["bnbdata"] = np.zeros_like(df["shr_energy"])
            df["extdata"] = np.zeros_like(df["shr_energy"])

            df["pi0energygev"] = df["pi0energy"]*0.001


            df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
            df.loc[ df['weightSplineTimesTune'] > 100, 'weightSplineTimesTune' ] = 1.
            df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
            
    def set_POT(self):
        ccr1POT = 3.48E+21
        ccr3POT = 6.43E+21
        ncr1POT = 2.66E+21
        ncr3POT = 2.31E+21
        ccPOT = ccr1POT+ccr3POT
        ncPOT = ncr1POT+ncr3POT
        dataPOT = 6.86e20

        self.ccr1['POT'] = np.ones_like(self.ccr1['nslice'])*dataPOT/ccPOT
        self.ccr3['POT'] = np.ones_like(self.ccr3['nslice'])*dataPOT/ccPOT
        self.ncr1['POT'] = np.ones_like(self.ncr1['nslice'])*dataPOT/ncPOT
        self.ncr3['POT'] = np.ones_like(self.ncr3['nslice'])*dataPOT/ncPOT
        
    def set_selections(self):
        self.CCNCSEL = CCNCSEL
        self.CCSEL = CCSEL
        self.NCSEL = NCSEL
        self.ACCEPTANCE = 'isVtxInFiducial == 1 and npi0==1'

    def efficiency_plot(self, variable, bin_edges):
        '''variable will be pi0_e or leadPi0_uz'''
        if type(bin_edges) == list:
            bin_edges = np.array(bin_edges)
            
        plt.figure(figsize=(7,6))

        ax1=plt.subplot(1, 1, 1)

        pi0_types = ['ccnc==0','ccnc==1']
        colors = ['b','r']
        labels = ['cc','nc']

        effs = {}
        effs_err = {}

        for pi0_type, color, label in zip(pi0_types, colors, labels):
            efficiency, efficiency_err = Eff_vectorial(self.pi0s, 
                          weights=self.pi0s["weightSplineTimesTune"], 
                          var=variable, 
                          num_query=self.CCNCSEL, 
                          den_query=self.ACCEPTANCE+' and '+pi0_type, 
                          bin_edges=bin_edges,
                          with_uncertainties=True)
            
            ax1.errorbar((bin_edges[1:]+bin_edges[:-1])/2,
                         efficiency[0],
                         yerr=efficiency_err[0],
                         xerr=(bin_edges[1:]-bin_edges[:-1])/2,
                         fmt=color+'o-',
                         label=label)

            effs[pi0_type] = efficiency
            effs_err[pi0_type] = efficiency_err

        ax1.set_xlabel(var2label[variable])
        ax1.set_ylabel('CCNC Pi0 Selection Efficiency')
        ax1.set_ylim(0., 1)
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.legend(loc='upper left')
        ax1.grid(True)

        plt.show()

        return effs