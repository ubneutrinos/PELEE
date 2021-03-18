import localSettings as ls
from load_data_run123 import *
from unblinding_far_sideband import *
from plotting_tools import *
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
ACCEPTANCE = 'isVtxInFiducial == 1 and npi0==1'

dict_pi0enrg = {'varT':'pi0_e','labelT':'True Pi0 Energy [GeV]',\
                'varR':'pi0energygev','labelR':'Reco Pi0 Energy [GeV]',\
                'bin_edges': np.array([0.1, 0.25, 0.4, 0.7, 1.5])}

dict_pi0csth = {'varT':'leadPi0_uz','labelT':r'True Pi0 $\cos(\theta)$',\
                'varR':'pi0momanglecos','labelR':r'Reco Pi0 $\cos(\theta)$',\
                'bin_edges': np.array([-1, -0.4, 0.2, 0.7, 1.0])}

dict_pi0mass = {'varT':'','labelT':r'',\
                'varR':'pi0_mass_Y_corr','labelR':r'Reco Pi0 Mass [MeV]',\
                'bin_edges': np.linspace(0,400,11)}

var2dict = {'pi0enrg':dict_pi0enrg,'pi0csth':dict_pi0csth,'pi0mass':dict_pi0mass}

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

class ccncpi0_analysis(object):
    def __init__(self, ntuple_path, fold="nuselection", tree="NeutrinoSelectionFilter", fcc1="prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2", fcc3="prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2", fnc1="prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2", fnc3="prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2", addtlvars=["weightsGenieUp", "weightsGenieDn"]):
        self.vardict = get_variables()
        self.variables = self.vardict['VARIABLES']+\
                    self.vardict['NUEVARS']+\
                    self.vardict['PI0VARS']+\
                    self.vardict['MCFVARS']+\
                    ['weightSplineTimesTune']+\
                    ['pi0_mcgamma0_pz','pi0_mcgamma1_pz','pi0_mcgamma0_e','pi0_mcgamma1_e']+\
                    ["weightsGenie", "weightSpline"]+\
                    ["weightsFlux", "weightSpline"]+\
                    addtlvars
        self.variables = list(set(self.variables))
        
        self.uproot_v = []
        self.df_v = []
        if fcc1 is None:
            self.ucc1 = None
            self.ccr1 = None
        else:
            print('Loading cc1')
            self.ucc1 = uproot.open(ntuple_path+fcc1+".root")[fold][tree]
            self.uproot_v.append(self.ucc1)
            self.ccr1 = self.ucc1.pandas.df(self.variables, flatten=False)
            self.df_v.append(self.ccr1)
        if fnc1 is None:
            self.unc1 = None
            self.ncr1 = None
        else:
            print('Loading nc1')
            self.unc1 = uproot.open(ntuple_path+fnc1+".root")[fold][tree]
            self.uproot_v.append(self.unc1)
            self.ncr1 = self.unc1.pandas.df(self.variables, flatten=False)
            self.df_v.append(self.ncr1)
        if fcc3 is None:
            self.ucc3 = None
            self.ccr3 = None
        else:
            print('Loading cc3')
            self.ucc3 = uproot.open(ntuple_path+fcc3+".root")[fold][tree]
            self.uproot_v.append(self.ucc3)
            self.ccr3 = self.ucc3.pandas.df(self.variables, flatten=False)
            self.df_v.append(self.ccr3)
        if fnc3 is None:
            self.unc3 = None
            self.ncr3 = None
        else:
            print('Loading nc3')
            self.unc3 = uproot.open(ntuple_path+fnc3+".root")[fold][tree]
            self.uproot_v.append(self.unc3)
            self.ncr3 = self.unc3.pandas.df(self.variables, flatten=False)
            self.df_v.append(self.ncr3)
        
        self.first_initialisation()
        self.set_POT()
        self.pi0s = pd.concat(self.df_v, ignore_index=True)
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
            
    def get_dataPOT(self):
        return 6.86e20

    def set_POT(self):
        ccr1POT = 3.48E+21
        ccr3POT = 6.43E+21
        ncr1POT = 2.66E+21
        ncr3POT = 2.31E+21
        ccPOT = ccr1POT+ccr3POT
        ncPOT = ncr1POT+ncr3POT
        dataPOT = self.get_dataPOT()
        
        if self.ccr1 is not None:
            self.ccr1['POT'] = np.ones_like(self.ccr1['nslice'])*dataPOT/ccPOT
        if self.ccr3 is not None:
            self.ccr3['POT'] = np.ones_like(self.ccr3['nslice'])*dataPOT/ccPOT
        if self.ncr1 is not None:
            self.ncr1['POT'] = np.ones_like(self.ncr1['nslice'])*dataPOT/ncPOT
        if self.ncr3 is not None:
            self.ncr3['POT'] = np.ones_like(self.ncr3['nslice'])*dataPOT/ncPOT
        
    def set_selections(self):
        self.CCNCSEL = CCNCSEL
        self.CCSEL = CCSEL
        self.NCSEL = NCSEL
        self.ACCEPTANCE = ACCEPTANCE

    def efficiency_plot(self, variable, numqs="", denq="", pi0_types=['ccnc==0','ccnc==1'],\
                        colors=['b','r'], labels=['cc','nc'],\
                        ylim=(0.,1),ylab='CCNC Pi0 Selection Efficiency'):
        '''variable will be pi0enrg or pi0csth'''

        v2d = var2dict[variable]
        bin_edges = v2d['bin_edges']
            
        plt.figure(figsize=(7,6))

        ax1=plt.subplot(1, 1, 1)

        if numqs=="": numqs=self.CCNCSEL
        if denq=="": denq=self.ACCEPTANCE

        if type(numqs) is not type([]):
            numqs=[numqs for i in range(0,len(labels))]

        effs = {}
        effs_err = {}

        for pi0_type, numq, color, label in zip(pi0_types, numqs, colors, labels):
            efficiency, efficiency_err = Eff_vectorial(self.pi0s, 
                          weights=self.pi0s["weightSplineTimesTune"], 
                          var=v2d['varT'],
                          num_query=numq,
                          den_query=denq+' and '+pi0_type,
                          bin_edges=bin_edges,
                          with_uncertainties=True)
            
            ax1.errorbar((bin_edges[1:]+bin_edges[:-1])/2,
                         efficiency[0],
                         yerr=efficiency_err[0],
                         xerr=(bin_edges[1:]-bin_edges[:-1])/2,
                         fmt=color+'o-',
                         label=label)

            effs[label] = efficiency
            effs_err[label] = efficiency_err

        ax1.set_xlabel(v2d['labelT'])
        ax1.set_ylabel(ylab)
        ax1.set_ylim(ylim)
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.legend(loc='upper left')
        ax1.grid(True)

        plt.show()

        return effs, effs_err

    def ccnc_plot(self,variable,queries,acceptance="",\
                  colors=['b','r'],labels=['cc','nc'],title='truth',\
                  reco=False):

        v2d = var2dict[variable]
        bin_edges = v2d['bin_edges']
        if acceptance=="": acceptance=self.ACCEPTANCE

        myvar = v2d['varT']
        mylab = v2d['labelT']
        if reco:
            myvar = v2d['varR']
            mylab = v2d['labelR']

        plt.figure(figsize=(6, 6))

        vals = [self.pi0s.query(acceptance+' and '+Q)[myvar] for Q in queries]
        whgs = [self.pi0s.query(acceptance+' and '+Q)['weightSplineTimesTune']*self.pi0s.query(acceptance+' and '+Q)['POT'] for Q in queries]

        n, b, p = plt.hist(vals, bins=bin_edges, weights=whgs,
                           histtype='step',linestyle='solid',color=colors, linewidth=1.2,
                           stacked=False, label=labels)
        plt.title(title)
        plt.ylabel('selected events ('+str(self.get_dataPOT())+' POT)')
        plt.xlabel(mylab)
        plt.legend(loc=2)
        if variable == 'pi0enrg': plt.legend(loc=1)
        plt.show()

        return n, b, p

    def ccnc_plot_withratio(self,variable,queries,acceptance="",\
                            colors=['b','r'],labels=['cc','nc'],title='truth',rylab=""):

        v2d = var2dict[variable]
        bin_edges = v2d['bin_edges']
        if acceptance=="": acceptance=self.ACCEPTANCE
        if rylab=="": rylab=labels[1]+' / '+labels[0]

        plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])

        vals = [self.pi0s.query(acceptance+' and '+Q)[v2d['varT']] for Q in queries]
        whgs = [self.pi0s.query(acceptance+' and '+Q)['weightSplineTimesTune']*self.pi0s.query(acceptance+' and '+Q)['POT'] for Q in queries]

        n, b, p = plt.hist(vals, bins=bin_edges, weights=whgs,
                           histtype='step',linestyle='solid',color=colors, linewidth=1.2,
                           stacked=False, label=labels)
        plt.title(title)
        ax1.set_ylabel('selected events ('+str(self.get_dataPOT())+' POT)')
        ax1.legend(loc=2)
        if v2d['varT'] == 'pi0_e': plt.legend(loc=1)

        ratio = n[1] / n[0]

        ax2 = plt.subplot(gs[1])
        ax2.set_xlabel(v2d['labelT'])
        ax2.set_ylabel(rylab)
        bincenters = 0.5 * (b[1:] + b[:-1])
        bin_size = [(b[i + 1] - b[i]) / 2 for i in range(len(b) - 1)]
        ax2.errorbar(bincenters, ratio, xerr=bin_size, yerr=Ratio_err(n[1],n[0],np.sqrt(n[1]),np.sqrt(n[0])), fmt="rs")#ratio_error
        ax2.set_ylim(0.2, 0.8)
        ax2.grid(True)

        return ratio

    def ccnc_resolution_plot(self,variable,sel,tit,lbl,xlab,p0,nbins=16,rng=(-1,1)):
        v2d = var2dict[variable]

        plt.figure(figsize=(6, 6))

        var = self.pi0s.query(sel)[v2d['varR']]-self.pi0s.query(sel)[v2d['varT']]
        wgh = self.pi0s.query(sel)['weightSplineTimesTune']*self.pi0s.query(sel)['POT']
        n, b, p = plt.hist(var, bins=nbins, range=rng,weights=wgh,
                           histtype='step',linestyle='solid',color=('black'), linewidth=1.2,
                           stacked=False, label=lbl)
        plt.title(tit)
        plt.ylabel('selected events ('+str(self.get_dataPOT())+' POT)')
        plt.xlabel(xlab)
        plt.ylim(0., plt.gca().get_ylim()[1]*1.2)
        plt.legend(loc=2)

        #gaussian fit
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        bin_centres = (b[:-1] + b[1:])/2
        coeff, var_matrix = curve_fit(gauss, bin_centres, n, p0=p0)
        errs = np.sqrt(np.diag(var_matrix))
        curvebins = np.linspace(int(b[0]),int(b[-1]),200)
        # Get the fitted curve
        hist_fit = gauss(curvebins, *coeff)
        plt.text(0.25,plt.gca().get_ylim()[-1]*0.95,"sigma=%.3f+/-%.3f"%(coeff[2],errs[2]))
        plt.plot(curvebins, hist_fit, label='Fitted data')

    def smearing_matrix(self,variable,sel,title):
        v2d = var2dict[variable]

        fig = plt.figure(figsize=(4, 4))
        x = self.pi0s.query(sel)[v2d['varT']]
        y = self.pi0s.query(sel)[v2d['varR']]
        h2d = plt.hist2d(x, y, bins=v2d['bin_edges'])
        plt.xlabel(v2d['labelT'])
        plt.ylabel(v2d['labelR'])
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(6, 6))
        sm = h2d[0]
        sm = sm / sm.sum(axis=0)
        plt.imshow(sm,origin='lower')
        for i in range(len(sm[0])):
            for j in range(len(sm[0])):
                text = plt.text(j, i, "%.2f"%sm[i, j],ha="center", va="center", color="w")
        plt.title(title)
        plt.ylabel("true bin")
        plt.xlabel("reco bin")
        plt.colorbar()

        return sm

    def closure_test(self,variable,ccsm,recoCCsel,ncsm,recoNCsel,effCC,effNC,fCC,fNC,trueratio):
        v2d = var2dict[variable]
        bin_edges = v2d['bin_edges']
        trueCCsel = np.matmul(ccsm,recoCCsel)
        trueNCsel = np.matmul(ncsm,recoNCsel)
        print('recoCCsel =',recoCCsel)
        print('recoNCsel =',recoNCsel)
        print('trueCCsel =',trueCCsel)
        print('trueNCsel =',trueNCsel)
        print('effCC =',effCC)
        print('effNC =',effNC)
        print('fCC =',fCC)
        print('fNC =',fNC)
        print('trueratio =',trueratio)
        den = effNC*((1-fNC) * trueNCsel - fNC * trueCCsel)
        print("den =",den)
        num = effCC*((1-fCC) * trueCCsel - fCC * trueNCsel)
        print("num =",num)
        R = num / den
        print("R =",R)
        #dR2 = (dx*dR/dx)^2 + (dy*dR/dy)^2
        tmpx = trueNCsel
        tmpy = trueCCsel
        tmpa = (1-fCC)/fCC
        tmpb = fNC*effNC
        tmpc = (1-fNC)/fNC
        tmpd = fCC*effCC
        dR2 = trueNCsel * ( (tmpd*tmpy*(1-tmpa*tmpc))/(tmpb*(tmpy-tmpc*tmpx)*(tmpy-tmpc*tmpx)) )**2 +\
              trueCCsel * ( (tmpd*tmpx*(tmpa*tmpc-1))/(tmpb*(tmpy-tmpc*tmpx)*(tmpy-tmpc*tmpx)) )**2
        dR = np.sqrt(dR2)
        print("dR =",dR)
        plt.figure(figsize=(6, 6))
        print(bin_edges)
        print(bin_edges[1:])
        print(bin_edges[:-1])
        print(bin_edges[:-1]+0.5*(bin_edges[1:]-bin_edges[:-1]))
        bin_centers = bin_edges[:-1]+0.5*(bin_edges[1:]-bin_edges[:-1])
        plt.plot(bin_centers, trueratio, 'ro')
        plt.errorbar(bin_centers,trueratio,yerr=None,xerr=0.5*(bin_edges[1:]-bin_edges[:-1]),fmt='rd',label='truth')
        plt.errorbar(bin_centers,R,yerr=dR,xerr=0.5*(bin_edges[1:]-bin_edges[:-1]),fmt='bo',label='closure')
        plt.xlim(bin_edges[0],bin_edges[-1])
        plt.ylim(0,1)
        plt.ylabel('NC/CC ratio')
        plt.xlabel(v2d['labelT'])
        plt.legend()
        plt.show()
