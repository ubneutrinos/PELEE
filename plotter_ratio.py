#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package plotter
Plotter for searchingfornues TTree

This module produces plot from the TTree produced by the
searchingfornues framework (https://github.com/ubneutrinos/searchingfornues)

Example:
    my_plotter = plotter.Plotter(samples, weights)
    fig, ax1, ax2 = my_plotter.plot_variable(
        "reco_e",
        query="selected == 1"
        kind="event_category",
        title="$E_{deposited}$ [GeV]",
        bins=20,
        range=(0, 2)
    )

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
"""

import math
import warnings
import bisect

from collections import defaultdict
from collections.abc import Iterable
import scipy.stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


paper_labels_numu = {
    11: r"$\nu_e$ CC",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    22: r"$\nu_{\mu}$ CC 0p",
    23: r"$\nu_{\mu}$ CC 1p",
    24: r"$\nu_{\mu}$ CC 2p",
    25: r"$\nu_{\mu}$ CC 3+p",
    3: r"NC $\nu$",
    5: r"Dirt",# (Outside TPC)
}

paper_labels = {
    11: r"$\nu_e$ CC",
    111: r"MiniBooNE LEE",
    2: r"$\nu$ other",
    31: r"$\nu$ with $\pi^{0}$",
    5: r"Dirt",# (Outside TPC)
}

paper_labels_xsec = {
    1: r"$\nu_e$ CC with $\pi$",
    10: r"$\nu_e$ CC 0p0$\pi$",
    11: r"$\nu_e$ CC Np0$\pi$",
    111: r"MiniBooNE LEE",
    2: r"$\nu$ other",
    31: r"$\nu$ with $\pi^{0}$",
    5: r"Dirt",# (Outside TPC)
}


category_labels = {
    1: r"$\nu_e$ CC",
    10: r"$\nu_e$ CC0$\pi$0p",
    11: r"$\nu_e$ CC0$\pi$Np",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    222: r"$\nu_{\mu}$ CC w/ Michel",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    22: r"$\nu_{\mu}$ CC 0p",
    23: r"$\nu_{\mu}$ CC 1p",
    24: r"$\nu_{\mu}$ CC 2p",
    25: r"$\nu_{\mu}$ CC 3+p",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out. fid. vol.",
    # eta categories start with 80XX
    801: r"$\eta \rightarrow$ other",
    802: r"$\nu_{\mu} \eta \rightarrow \gamma\gamma$",
    803: r'1 $\pi^0$',
    804: r'2 $\pi^0$',
    807: r'3+ $\pi^0$',
    805: r'$\nu$ other',
    806: r'out of FV',
    # ccncpi0 categories start with 90X
    901: r"$\nu$, no $\pi^0$",
    902: r"$\nu$, 2+$\pi^0$",
    903: r"$\nu_{\mu}$ CC 1$\pi^{0} +$ 0$\pi^{\pm}$",
    904: r"$\nu$ NC 1$\pi^{0} +$ 0$\pi^{\pm}$",
    905: r"$\nu_{e}$ CC $\pi^{0}$",
    906: r"OOFV",
    907: r"Cosmic",
    908: r"$\nu_{\mu}$ CC 1$\pi^{0} +$ N$\pi^{\pm}$",
    909: r"$\bar{\nu_{\mu}}$ CC 1$\pi^{0} +$ N$\pi^{\pm}$",
    910: r"$\nu$ NC 1$\pi^{0} +$ N$\pi^{\pm}$",
    #
    6: r"other",
    0: r"No slice"
}


flux_labels = {
    1: r"$\pi$",
    10: r"K",
    111: r"MiniBooNE LEE",
    0: r"backgrounds"
}

sample_labels = {
    0: r"data",
    1: r"mc",
    2: r"nue",
    3: r"ext",
    4: r"lee",
    5: r"dirt",
    6: r"ccnopi",
    7: r"cccpi",
    8: r"ncnopi",
    9: r"nccpi",
    10: r"ncpi0",
    11: r"ccpi0",
    802: r"eta"
}

flux_colors = {
    0: "xkcd:cerulean",
    111: "xkcd:goldenrod",
    10: "xkcd:light red",
    1: "xkcd:purple",
}


pdg_labels = {
    2212: r"$p$",
    13: r"$\mu$",
    11: r"$e$",
    111: r"$\pi^0$",
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi^{\pm}$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic"
}

int_labels = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    3: "Coherent",
    4: "Coherent Elastic",
    5: "Electron scatt.",
    6: "IMDAnnihilation",
    7: r"Inverse $\beta$ decay",
    8: "Glashow resonance",
    9: "AMNuGamma",
    10: "MEC",
    11: "Diffractive",
    12: "EM",
    13: "Weak Mix"
}


int_colors = {
    0: "bisque",
    1: "darkorange",
    2: "goldenrod",
    3: "lightcoral",
    4: "forestgreen",
    5: "turquoise",
    6: "teal",
    7: "deepskyblue",
    8: "steelblue",
    80: "steelblue",
    81: "steelblue",
    82: "steelblue",
    9: "royalblue",
    10: "crimson",
    11: "mediumorchid",
    12: "magenta",
    13: "pink",
    111: "black"
}

category_colors = {
    4: "xkcd:light red",
    5: "xkcd:brick",
    8: "xkcd:cerulean",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    222: "pink",
    22: "xkcd:khaki",# "xkcd:lightblue",
    23: "xkcd:maroon",# "xkcd:cyan",
    24: "xkcd:teal",# "steelblue",
    25: "xkcd:coral",# "blue",
    3: "xkcd:cobalt",
    31: "xkcd:sky blue",
    1: "xkcd:green",
    10: "xkcd:mint green",
    11: "xkcd:lime green",
    111: "xkcd:goldenrod",
    6: "xkcd:grey",
    0: "xkcd:black",
    # eta categories
    803: "xkcd:cerulean",
    804: "xkcd:blue",
    807: "xkcd:blurple",
    801: "xkcd:purple",
    802: "xkcd:lavender",
    806: "xkcd:crimson",
    805: "xkcd:cyan",
    # ccncpi0 categories
    901: "xkcd:cyan",
    902: "xkcd:blurple",
    903: "xkcd:cerulean",
    904: "xkcd:sky blue",
    905: "xkcd:green",
    906: "xkcd:brick",
    907: "xkcd:light red",
    908: "xkcd:lime green",
    909: "xkcd:goldenrod",
    910: "xkcd:mint green"
}

pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "#e31a1c",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "#cab2d6",
}

class Plotter:
    """Main plotter class

    Args:
        samples (dict): Dictionary of pandas dataframes.
            mc`, `nue`, `data`, and `ext` are required. `lee` and `dirt` are optional.
        weights (dict): Dictionary of global dataframes weights.
            One for each entry in the samples dict.
        pot (int): Number of protons-on-target. Defaults is 4.5e19.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    def __init__(self, samples, weights, pot=4.5e19):
        self.weights = weights
        self.samples = samples
        self.pot = pot
        self.significance = 0
        self.significance_likelihood = 0
        self.chisqdatamc = 0
        self.sigma_shapeonly = 0
        self.detsys = {}
        self.stats = {}
        self.cov = None # covariance matrix from systematics
        self.cov_mc_stat = None
        self.cov_mc_detsys = None
        self.cov_stat = None
        self.cov_data_stat = None
        self.cov_full = None
        self._ratio_vals = None
        self._ratio_errs = None
        self.data = None # data binned events
        self.ext = None  # EXT binned events
        self.ext_err = None # EXT error (stat only)
        self.prediction = None # EXT + MC binned events

        #self.nu_pdg = nu_pdg = " nu_pdg != 100"
        self.nu_pdg = nu_pdg = "~(abs(nu_pdg) == 12 & ccnc == 0)" # query to avoid double-counting events in MC sample with other MC samples

        if ("ccpi0" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ccpi0==1)"
        if ("eta" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~( (neta>0) & (true_nu_vtx_x > 0) & (true_nu_vtx_x < 250) & (true_nu_vtx_y > -110) & (true_nu_vtx_y < 110) & (true_nu_vtx_z > 0) & (true_nu_vtx_y < 1030) )"
        if ("ncpi0" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_np0==1 & mcf_nmp==0 & mcf_nmm==0 & mcf_nem==0 & mcf_nep==0)" #note: mcf_pass_ccpi0 is wrong (includes 'mcf_actvol' while sample is in all cryostat)
        if ("ccnopi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ccnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("cccpi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_cccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("nccpi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_nccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("ncnopi" in self.samples):
            self.nu_pdg = self.nu_pdg+" & ~(mcf_pass_ncnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"


        if "dirt" not in samples:
            warnings.warn("Missing dirt sample")

        necessary = ["category"]#, "selected",  # "trk_pfp_id", "shr_pfp_id_v",
                     #"backtracked_pdg", "nu_pdg", "ccnc", "trk_bkt_pdg", "shr_bkt_pdg"]

        missing = np.setdiff1d(necessary, samples["mc"].columns)

        if missing.size > 0:
            raise ValueError(
                "Missing necessary columns in the DataFrame: %s" % missing)

    @staticmethod
    def _chisquare(data, mc, err_mc):
        num = (data - mc)**2
        den = data+err_mc**2
        if np.count_nonzero(data):
            return sum(num / den) / len(data)
        return np.inf


    @staticmethod
    def _chisq_pearson(data, mc):
        return (data-mc)**2 / mc

    @staticmethod
    def _chisq_neyman(data, mc):
        return (data-mc)**2 / data

    def _chisq_CNP(self,data, mc):
        return np.sum((1/3.) * (self._chisq_neyman(data,mc) + 2 * self._chisq_pearson(data,mc)))/len(data)

    @staticmethod
    def _sigma_calc_likelihood(sig, bkg, err_bkg, scale_factor=1):
        """It calculates the significance with the profile likelihood ratio
        assuming an uncertainity on the background entries.
        Taken from http://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
        """
        b = bkg * scale_factor
        if not isinstance(err_bkg, Iterable):
            e = np.array([err_bkg]) * scale_factor
        else:
            e = err_bkg * scale_factor

        s = sig * scale_factor

        p1 = (s+b)*np.log((s+b)*(b+e**2)/(b**2+(s+b)*e**2))

        p2 = -s
        if sum(e) > 0:
            p2 = -b**2/(e**2)*np.log(1+e**2*s/(b*(b+e**2)))
        z = 2*(p1+p2)

        return math.sqrt(sum(z))

    @staticmethod
    def _sigma_calc_matrix(signal, background, scale_factor=1, cov=None):
        """It calculates the significance as the square root of the Δχ2 score

        Args:
            signal (np.array): array of signal histogram
            background (np.array): array of background histogram
            scale_factor (float, optional): signal and background scaling factor.
                Default is 1

        Returns:
            Square root of S•B^(-1)•S^T
        """

        bkg_array = background * scale_factor
        empty_elements = np.where(bkg_array == 0)[0]
        sig_array = signal * scale_factor
        cov = cov * scale_factor * scale_factor
        sig_array = np.delete(sig_array, empty_elements)
        bkg_array = np.delete(bkg_array, empty_elements)
        cov[np.diag_indices_from(cov)] += bkg_array
        emtxinv = np.linalg.inv(cov)
        chisq = float(sig_array.dot(emtxinv).dot(sig_array.T))

        return np.sqrt(chisq)


    def deltachisqfakedata(self, BinMin, BinMax, LEE_v, SM_v, nsample):

        deltachisq_v = []
        deltachisq_SM_v  = []
        deltachisq_LEE_v = []

        #print ('deltachisqfakedata!!!!!!')
        
        for n in range(1000):

            SM_obs, LEE_obs = self.genfakedata(BinMin, BinMax, LEE_v, SM_v, nsample)

            #chisq = self._chisq_CNP(SM_obs,LEE_obs)           
            #print ('LEE obs : ',LEE_obs)
            #print ('SM obs : ',SM_obs)
            
            chisq_SM_SM  = self._chisq_CNP(SM_v,SM_obs)
            chisq_LEE_SM = self._chisq_CNP(LEE_v,SM_obs)
            
            chisq_SM_LEE  = self._chisq_CNP(SM_v,LEE_obs)
            chisq_LEE_LEE = self._chisq_CNP(LEE_v,LEE_obs)
            
            deltachisq_SM  = (chisq_SM_SM-chisq_LEE_SM)
            deltachisq_LEE = (chisq_SM_LEE-chisq_LEE_LEE)

            #if (np.isnan(chisq)):
            #    continue

            #deltachisq_v.append(chisq)
            
            if (np.isnan(deltachisq_SM ) or np.isnan(deltachisq_LEE) ):
                continue

            deltachisq_SM_v.append(deltachisq_SM)
            deltachisq_LEE_v.append(deltachisq_LEE)

        #median = np.median(deltachisq_v)
        #dof = len(LEE_v)

        #return median/float(dof)

        #print ('delta SM  : ',deltachisq_SM_v)
        #print ('delta LEE : ',deltachisq_LEE_v)

        deltachisq_SM_v  = np.array(deltachisq_SM_v)
        deltachisq_LEE_v = np.array(deltachisq_LEE_v)

        if (len(deltachisq_SM_v) == 0):
            return 999.
        
        # find median of LEE distribution
        med_LEE = np.median(deltachisq_LEE_v)
        #print ('median LEE is ',med_LEE)
        # how many values in SM are above this value?
        nabove = len( np.where(deltachisq_SM_v > med_LEE)[0] )
        #print ('n above is ',nabove)
        frac = float(nabove) / len(deltachisq_SM_v)

        #print ('deltachisqfakedata!!!!!!')
        
        return math.sqrt(2)*scipy.special.erfinv(1-frac*2)
        
        #return frac

            
    def genfakedata(self, BinMin, BinMax, LEE_v, SM_v, nsample):

        p_LEE = LEE_v / np.sum(LEE_v)
        p_SM  = SM_v / np.sum(SM_v)

        #print ('PDF for LEE : ',p_LEE)
        #print ('PDF for SM  : ',p_SM)

        obs_LEE = np.zeros(len(LEE_v))
        obs_SM  = np.zeros(len(SM_v))

        max_LEE = np.max(p_LEE)
        max_SM  = np.max(p_SM)

        #print ('max of LEE : ',max_LEE)
        #print ('max of SM  : ',max_SM)

        n_sampled_LEE = 0
        n_sampled_SM  = 0

        while (n_sampled_LEE < nsample):

            value = BinMin + (BinMax-BinMin) * np.random.random()

            BinNumber = int((value-BinMin)/(BinMax-BinMin) * len(LEE_v))
            
            prob = np.random.random() * max_LEE
            if (prob < p_LEE[BinNumber]):
                #print ('LEE simulation: prob of %.02f vs. bin prob of %.02f leads to selecting event at bin %i'%(prob,p_LEE[BinNumber],BinNumber))
                obs_LEE[BinNumber] += 1
                n_sampled_LEE += 1

        while (n_sampled_SM < nsample):

            value = BinMin + (BinMax-BinMin) * np.random.random()

            BinNumber = int((value-BinMin)/(BinMax-BinMin) * len(SM_v))
            
            prob = np.random.random() * max_SM
            if (prob < p_SM[BinNumber]):
                obs_SM[BinNumber] += 1
                n_sampled_SM += 1

        return obs_SM, obs_LEE
            
            

    def _chisq_full_covariance(self,data, mc,CNP=True,STATONLY=False,verbose=False):

        np.set_printoptions(precision=3)

        dof = len(data)
        
        COV = self.cov + self.cov_mc_stat + self.cov_mc_detsys

        # remove rows/columns with zero data and MC
        remove_indices_v = []
        for i,d in enumerate(data):
            idx = len(data)-i-1
            if ((data[idx]==0) and (mc[idx] == 0)):
                remove_indices_v.append(idx)

        for idx in remove_indices_v:
            COV = np.delete(COV,idx,0)
            COV = np.delete(COV,idx,1)
            data = np.delete(data,idx,0)
            mc   = np.delete(mc,idx,0)


        COV_STAT = np.zeros([len(data), len(data)])


        ERR_STAT = 3. / ( 1./data + 2./mc )
        
        for i,d in enumerate(data):
            
            if (d == 0):
                ERR_STAT[i] = mc[i]/2.
            if (mc[i] == 0):
                ERR_STAT[i] = d

        if (CNP == False):
            ERR_STAT = data + mc


        COV_STAT[np.diag_indices_from(COV_STAT)] = ERR_STAT

        self.cov_stat = COV_STAT

        COV += COV_STAT

        if (STATONLY == True):
            COV = COV_STAT

        frac_cov = np.empty([len(COV), len(COV)])
        corr     = np.empty([len(COV), len(COV)])

        for i in range(len(COV)):
            for j in range(len(COV)):
                frac_cov[i][j] =  COV[i][j] / (mc[i] * mc[j])
                corr[i][j] = COV[i][j] / np.sqrt(COV[i][i] * COV[j][j])

        #print("COV matrix : ",COV)
        #print("FRAC COV matrix :",frac_cov)                                                                                                                                                        
        #print("CORR matrix : ",corr)
                
        diff = (data-mc)
        emtxinv = np.linalg.inv(COV)
        chisq = float(diff.dot(emtxinv).dot(diff.T))
        
        covdiag = np.diag(COV)
        chisqsum = 0.
        for i,d in enumerate(diff):
            #print ('bin %i has COV value %.02f'%(i,covdiag[i]))
            chisqsum += ( (d**2) /covdiag[i])

        return chisq, chisqsum, dof

    @staticmethod
    def _data_err(data,doAsym=False):
        obs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        low = [0.00,0.17,0.71,1.37,2.09,2.84,3.62,4.42,5.23,6.06,6.89,7.73,8.58,9.44,10.30,11.17,12.04,12.92,13.80,14.68,15.56]
#        hig = [0.38,3.30,4.64,5.92,7.16,8.38,9.58,10.77,11.95,13.11,14.27,15.42,16.56,17.70,18.83,19.96,21.08,22.20,23.32,24.44,25.55]
        hig = [1.15,3.30,4.64,5.92,7.16,8.38,9.58,10.77,11.95,13.11,14.27,15.42,16.56,17.70,18.83,19.96,21.08,22.20,23.32,24.44,25.55]
        if doAsym:
            lb = [i-low[i] if i<=20 else (np.sqrt(i)) for i in data]
            hb = [hig[i]-i if i<=20 else (np.sqrt(i)) for i in data]
            return (lb,hb)
        else: return (np.sqrt(data),np.sqrt(data))


    @staticmethod
    def _ratio_err(num, den, num_err, den_err):
        n, d, n_e, d_e = num, den, num_err, den_err
        n[n == 0] = 0.00001
        #d[d == 0] = 0.00001
        return np.array([
            #n[i] / d[i] * math.sqrt((n_e[i] / n[i])**2 + (d_e[i] / d[i])**2) <= this does not work if n[i]==0
            math.sqrt( ( n_e[i] / d[i] )**2 + ( n[i] * d_e[i] / (d[i]*d[i]) )**2) if d[i]>0 else 0
            for i, k in enumerate(num)
        ])

    @staticmethod
    def _is_fiducial(x, y, z):
        try:
            x_1 = x[:, 0] > 10
            x_2 = x[:, 1] > 10
            y_1 = y[:, 0] > 15
            y_2 = y[:, 1] > 15
            z_1 = z[:, 0] > 10
            z_2 = z[:, 1] > 50

            return x_1 & x_2 & y_1 & y_2 & z_1 & z_2
        except IndexError:
            return True

    def print_stats(self):
        print ('print stats...')
        for key,val in self.stats.items():
            print ('%s : %.02f'%(key,val))


    def _select_showers(self, variable, variable_name, sample, query="selected==1", score=0.5, extra_cut=None):
        variable = variable.ravel()

        if variable.size > 0:
            if isinstance(variable[0], np.ndarray):
                variable = np.hstack(variable)
                if "shr" in variable_name and variable_name != "shr_score_v":
                    shr_score = np.hstack(self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut))
                    shr_score_id = shr_score < score
                    variable = variable[shr_score_id]
                elif "trk" in variable_name and variable_name != "trk_score_v":
                    trk_score = np.hstack(self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut))
                    trk_score_id = trk_score >= score
                    variable = variable[trk_score_id]

        return variable


    def _apply_track_cuts(self,df,variable,track_cuts,mask):
        '''
        df is dataframe of the sample of interest
        variable is what values will be in the output
        track_cuts are list of tuples defining track_cuts
        input mask to be built upon

        returns
            Series of values of variable that pass all track_cuts
            boolean mask that represents union of input mask and new cut mask
        '''
        #need to do this fancy business with the apply function to make masks
        #this is because unflattened DataFrames are used
        for (var,op,val) in track_cuts:
            if type(op) == list:
                #this means treat two conditions in an 'or' fashion
                or_mask1 = df[var].apply(lambda x: eval("x{}{}".format(op[0],val[0])))#or condition 1
                or_mask2 = df[var].apply(lambda x: eval("x{}{}".format(op[1],val[1])))#or condition 2
                mask *= (or_mask1 + or_mask2) #just add the booleans for "or"
            else:
                mask *= df[var].apply(lambda x: eval("x{}{}".format(op,val))) #layer on each cut mask
        vars = (df[variable]*mask).apply(lambda x: x[x != False]) #apply mask
        vars = vars[vars.apply(lambda x: len(x) > 0)] #clean up empty slices
        #fix list comprehension issue for non '_v' variables
        if variable[-2:] != "_v":
            vars = vars.apply(lambda x: x[0])
        elif "_v" not in variable:
            print("_v not found in variable, assuming event-level")
            print("not fixing list comprehension bug for this variable")

        return vars, mask

    def _select_longest(self,df, variable, mask):
        '''
        df: dataframe for sample
        variable: Series of values that pass cuts defined by mask
        mask: mask used to find variable

        returns
            list of values of variable corresponding to longest track in each slices
            boolean mask for longest tracks in df
        '''

        #print("selecting longest...")
        #print("mask", mask)
        trk_lens = (df['trk_len_v']*mask).apply(lambda x: x[x != False])#apply mask to track lengths
        trk_lens = trk_lens[trk_lens.apply(lambda x: len(x) > 0)]#clean up slices
        variable = variable.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
        variable = variable[variable.apply(lambda x: len(x) > 0)] #clean up empty slices
        nan_mask = variable.apply(lambda x: np.nan in x or "nan" in x)
        longest_mask = trk_lens.apply(lambda x: x == x[list(x).index(max(x))])#identify longest
        variable = (variable*longest_mask).apply(lambda x: x[x!=False])#apply mask
        if len(variable.iloc[0]) == 1:
            variable = variable.apply(lambda x: x[0] if len(x)>0 else -999)#expect values, not lists, for each event
        else:
            if len(variable.iloc[0]) == 0:
                raise ValueError(
                    "There is no longest track per slice")
            elif len(variable.iloc[0]) > 1:
                #this happens with the reco_nu_e_range_v with unreconstructed values
                print("there are more than one longest slice")
                print(variable.iloc[0])
                try:
                    variable = variable.apply(lambda x: x[0])
                except:
                    raise ValueError(
                        "There is more than one longest track per slice in \n var {} lens {}".format(variable,trk_lens))

        return variable, longest_mask

    def _selection(self, variable, sample, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        '''
        variable,  must be specified
        select_longest, True by default, keeps from multiple tracks of same event making it through
        query must be a string defining event-level cuts
        track_cuts is a list of cuts of which each entry looks like
            (variable_tobe_cut_on, '>'or'<'or'=='etc, cut value )
            or
            (variable, [operator1, operator2], [cutval1, cutval2]) to do an 'or' cut
        track_
        returns an Series of values that pass all track_cuts
        '''
        sel_query = query
        if extra_cut is not None:
            sel_query += "& %s" % extra_cut
        '''
        if ( (track_cuts == None) or (select_longest == False) ):
            return sample.query(sel_query).eval(variable).ravel()
        '''


        '''
        df = sample.query(sel_query)
        #print (df.isna().sum())
        dfna = df.isna()
        for (colname,colvals) in dfna.iteritems():
            if (colvals.sum() != 0):
                print ('name : ',colname)
                print ('nan entries : ',colvals.sum())
        '''
        df = sample.query(sel_query)
        #if (track_cuts != None):
        #    df = sample.query(sel_query).dropna().copy() #don't want to eliminate anything from memory

        #df = sample.query(sel_query).dropna().copy() #don't want to eliminate anything from memory

        track_cuts_mask = None #df['trk_score_v'].apply(lambda x: x == x) #all-True mask, assuming trk_score_v is available
        if track_cuts is not None:
            vars, track_cuts_mask = self._apply_track_cuts(df,variable,track_cuts,track_cuts_mask)
        else:
            vars = df[variable]
        #vars is now a Series object that passes all the cuts

        #select longest of the cut passing tracks
        #assuming all track-level variables end in _v
        if variable[-2:] == "_v" and select_longest:
            vars, longest_mask = self._select_longest(df, vars, track_cuts_mask)
        elif "_v_" in variable:
            print("Variable is being interpretted as event-level, not track_level, despite having _v in name")
            print("the longest track is NOT being selected")
        return vars.ravel()

    def _categorize_entries_pdg(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):

        if "trk" in variable:
            pfp_id_variable = "trk_pfp_id"
            score_v = self._selection("trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        else:
            pfp_id_variable = "shr_pfp_id_v"
            score_v = self._selection("shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)


        pfp_id = self._selection(
            pfp_id_variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        pfp_id = np.subtract(pfp_id, 1)
        backtracked_pdg = np.abs(self._selection(
            "backtracked_pdg", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest))

        plotted_variable = self._select_showers(
            plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        if "trk" in variable:
            pfp_id = np.array([pf_id[score > 0.5] for pf_id, score in zip(pfp_id, score_v)])
        else:
            pfp_id = np.array([pf_id[score <= 0.5] for pf_id, score in zip(pfp_id, score_v)])

        pfp_pdg = np.array([pdg[pf_id]
                            for pdg, pf_id in zip(backtracked_pdg, pfp_id)])
        pfp_pdg = np.hstack(pfp_pdg)
        pfp_pdg = abs(pfp_pdg)

        return pfp_pdg, plotted_variable

    def _categorize_entries_single_pdg(self, sample, variable, query="selection==1", extra_cut=None, track_cuts=None, select_longest=True):
        if "trk" in variable:
            bkt_variable = "trk_bkt_pdg"
        else:
            bkt_variable = "shr_bkt_pdg"

        backtracked_pdg = np.abs(self._selection(
            bkt_variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest))
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        return backtracked_pdg, plotted_variable

    def _categorize_entries(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "category", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable

    def _categorize_entries_paper(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "paper_category", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable

    def _categorize_entries_paper_xsec(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "paper_category_xsec", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)

        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable

    
    def _categorize_entries_ccncpi0(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "ccncpi0_category", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable

    
    def _categorize_entries_int(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "interaction", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable

    def _categorize_entries_flux(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "flux", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable

    def _categorize_entries_trk1(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "trk1_backtracked_pdg", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable

    def _categorize_entries_paper_numu(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "paper_category_numu", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable
    
    def _categorize_entries_sample(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None, select_longest=True):
        category = self._selection(
            "sample", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        return category, plotted_variable


        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s > 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                    category = np.array([
                        np.array([c] * len(v[s < 0.5])) for c, v, s in zip(category, plotted_variable, score)
                    ])
                category = np.hstack(category)

            plotted_variable = self._select_showers(
                plotted_variable, variable, sample, query=query, extra_cut=extra_cut)

        return category, plotted_variable



    @staticmethod
    def _variable_bin_scaling(bins, bin_width, variable):
        idx = bisect.bisect_left(bins, variable)
        if len(bins) > idx:
            return bin_width/(bins[idx]-bins[idx-1])
        return 0

    def _get_genie_weight(self, sample, variable, query="selected==1", extra_cut=None, track_cuts=None,\
                          select_longest=True, weightvar="weightSplineTimesTune",weightsignal=None):

        plotted_variable = self._selection(
            variable, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        genie_weights = self._selection(
            weightvar, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        if (weightsignal != None):
            genie_weights *= self._selection(
            weightsignal, sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
        if plotted_variable.size > 0:
            if isinstance(plotted_variable[0], np.ndarray):
                if "trk" in variable or select_longest:
                    score = self._selection(
                        "trk_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                else:
                    score = self._selection(
                        "shr_score_v", sample, query=query, extra_cut=extra_cut, track_cuts=track_cuts, select_longest=select_longest)
                genie_weights = np.array([
                    np.array([c] * len(v[s > 0.5])) for c, v, s in zip(genie_weights, plotted_variable, score)
                ])
                genie_weights = np.hstack(genie_weights)
        return genie_weights

    def _get_variable(self, variable, query, track_cuts=None):

        '''
        nu_pdg = "~(abs(nu_pdg) == 12 & ccnc == 0)"
        if ("ccpi0" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ccpi0==1)"
        if ("ncpi0" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_np0==1 & mcf_nmp==0 & mcf_nmm==0 & mcf_nem==0 & mcf_nep==0)" #note: mcf_pass_ccpi0 is wrong (includes 'mcf_actvol' while sample is in all cryostat)
        if ("ccnopi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ccnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("cccpi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_cccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("nccpi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_nccpi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        if ("ncnopi" in self.samples):
            nu_pdg = nu_pdg+" & ~(mcf_pass_ncnopi==1 & (nslice==0 | (slnunhits/slnhits)>0.1))"
        '''

        # if plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
        #     query += "& %s <= %g & %s >= %g" % (
        #         variable, plot_options["range"][1], variable, plot_options["range"][0])

        mc_plotted_variable = self._selection(
            variable, self.samples["mc"], query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts)
        mc_plotted_variable = self._select_showers(
            mc_plotted_variable, variable, self.samples["mc"], query=query, extra_cut=self.nu_pdg)
        mc_weight = [self.weights["mc"]] * len(mc_plotted_variable)

        nue_plotted_variable = self._selection(
            variable, self.samples["nue"], query=query, track_cuts=track_cuts)
        nue_plotted_variable = self._select_showers(
            nue_plotted_variable, variable, self.samples["nue"], query=query)
        nue_weight = [self.weights["nue"]] * len(nue_plotted_variable)

        ext_plotted_variable = self._selection(
            variable, self.samples["ext"], query=query, track_cuts=track_cuts)
        ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)
        ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)

        dirt_weight = []
        dirt_plotted_variable = []
        if "dirt" in self.samples:
            dirt_plotted_variable = self._selection(
                variable, self.samples["dirt"], query=query, track_cuts=track_cuts)
            dirt_plotted_variable = self._select_showers(
                dirt_plotted_variable, variable, self.samples["dirt"], query=query)
            dirt_weight = [self.weights["dirt"]] * len(dirt_plotted_variable)

        ncpi0_weight = []
        ncpi0_plotted_variable = []
        if "ncpi0" in self.samples:
            ncpi0_plotted_variable = self._selection(
                variable, self.samples["ncpi0"], query=query, track_cuts=track_cuts)
            ncpi0_plotted_variable = self._select_showers(
                ncpi0_plotted_variable, variable, self.samples["ncpi0"], query=query)
            ncpi0_weight = [self.weights["ncpi0"]] * len(ncpi0_plotted_variable)

        ccpi0_weight = []
        ccpi0_plotted_variable = []
        if "ccpi0" in self.samples:
            ccpi0_plotted_variable = self._selection(
                variable, self.samples["ccpi0"], query=query, track_cuts=track_cuts)
            ccpi0_plotted_variable = self._select_showers(
                ccpi0_plotted_variable, variable, self.samples["ccpi0"], query=query)
            ccpi0_weight = [self.weights["ccpi0"]] * len(ccpi0_plotted_variable)

        eta_weight = []
        eta_plotted_variable = []
        if "eta" in self.samples:
            eta_plotted_variable = self._selection(
                variable, self.samples["eta"], query=query, track_cuts=track_cuts)
            eta_plotted_variable = self._select_showers(
                eta_plotted_variable, variable, self.samples["eta"], query=query)
            eta_weight = [self.weights["eta"]] * len(eta_plotted_variable)

        ccnopi_weight = []
        ccnopi_plotted_variable = []
        if "ccnopi" in self.samples:
            ccnopi_plotted_variable = self._selection(
                variable, self.samples["ccnopi"], query=query, track_cuts=track_cuts)
            ccnopi_plotted_variable = self._select_showers(
                ccnopi_plotted_variable, variable, self.samples["ccnopi"], query=query)
            ccnopi_weight = [self.weights["ccnopi"]] * len(ccnopi_plotted_variable)

        cccpi_weight = []
        cccpi_plotted_variable = []
        if "cccpi" in self.samples:
            cccpi_plotted_variable = self._selection(
                variable, self.samples["cccpi"], query=query, track_cuts=track_cuts)
            cccpi_plotted_variable = self._select_showers(
                cccpi_plotted_variable, variable, self.samples["cccpi"], query=query)
            cccpi_weight = [self.weights["cccpi"]] * len(cccpi_plotted_variable)

        nccpi_weight = []
        nccpi_plotted_variable = []
        if "nccpi" in self.samples:
            nccpi_plotted_variable = self._selection(
                variable, self.samples["nccpi"], query=query, track_cuts=track_cuts)
            nccpi_plotted_variable = self._select_showers(
                nccpi_plotted_variable, variable, self.samples["nccpi"], query=query)
            nccpi_weight = [self.weights["nccpi"]] * len(nccpi_plotted_variable)

        ncnopi_weight = []
        ncnopi_plotted_variable = []
        if "ncnopi" in self.samples:
            ncnopi_plotted_variable = self._selection(
                variable, self.samples["ncnopi"], query=query, track_cuts=track_cuts)
            ncnopi_plotted_variable = self._select_showers(
                ncnopi_plotted_variable, variable, self.samples["ncnopi"], query=query)
            ncnopi_weight = [self.weights["ncnopi"]] * len(ncnopi_plotted_variable)

        lee_weight = []
        lee_plotted_variable = []
        if "lee" in self.samples:
            lee_plotted_variable = self._selection(
                variable, self.samples["lee"], query=query, track_cuts=track_cuts)
            lee_plotted_variable = self._select_showers(
                lee_plotted_variable, variable, self.samples["lee"], query=query)
            lee_weight = self.samples["lee"].query(
                query)["leeweight"] * self.weights["lee"]

        total_weight = np.concatenate((mc_weight, nue_weight, ext_weight, dirt_weight, ncpi0_weight, ccpi0_weight, eta_weight, ccnopi_weight, cccpi_weight, nccpi_weight, ncnopi_weight, lee_weight))
        total_variable = np.concatenate((mc_plotted_variable, nue_plotted_variable, ext_plotted_variable, dirt_plotted_variable, ncpi0_plotted_variable, ccpi0_plotted_variable, eta_plotted_variable, ccnopi_plotted_variable, cccpi_plotted_variable, nccpi_plotted_variable, ncnopi_plotted_variable, lee_plotted_variable))
        return total_variable, total_weight


    def plot_2d(self, variable1_name, variable2_name, query="selected==1", track_cuts=None, **plot_options):
        variable1, weight1 = self._get_variable(variable1_name, query, track_cuts=track_cuts)
        variable2, weight2 = self._get_variable(variable2_name, query, track_cuts=track_cuts)

        heatmap, xedges, yedges = np.histogram2d(variable1, variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=weight1)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, axes  = plt.subplots(1,3, figsize=(15,5))

        axes[0].imshow(heatmap.T, extent=extent, origin='lower', aspect="auto")

        data_variable1 = self._selection(variable1_name, self.samples["data"], query=query, track_cuts=track_cuts)
        data_variable1 = self._select_showers(data_variable1, variable1_name, self.samples["data"], query=query)

        data_variable2 = self._selection(
            variable2_name, self.samples["data"], query=query, track_cuts=track_cuts)
        data_variable2 = self._select_showers(
            data_variable2, variable2_name, self.samples["data"], query=query)

        heatmap_data, xedges, yedges = np.histogram2d(data_variable1, data_variable2, range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [
                                                      plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                      bins=[plot_options["bins_x"],
                                                      plot_options["bins_y"]])

        axes[1].imshow(heatmap_data.T, extent=extent, origin='lower', aspect="auto")

        ratio = heatmap_data/heatmap
        im_ratio = axes[2].imshow(ratio.T, extent=extent, origin='lower', aspect='auto', vmin=0, vmax=2, cmap="coolwarm")
        fig.colorbar(im_ratio)

        axes[0].title.set_text('MC+EXT')
        axes[1].title.set_text('Data')
        axes[2].title.set_text('Data/(MC+EXT)')
        if "title" in plot_options:
            axes[0].set_xlabel(plot_options["title"].split(";")[0])
            axes[0].set_ylabel(plot_options["title"].split(";")[1])
            axes[1].set_xlabel(plot_options["title"].split(";")[0])
            axes[2].set_xlabel(plot_options["title"].split(";")[0])
        else:
            axes[0].set_xlabel(variable1_name)
            axes[0].set_ylabel(variable2_name)
            axes[1].set_xlabel(variable1_name)
            axes[2].set_xlabel(variable1_name)

        return fig, axes

    def plot_2d_oneplot(self, variable1_name, variable2_name, query="selected==1", track_cuts=None, **plot_options):
        variable1, weight1 = self._get_variable(variable1_name, query, track_cuts=track_cuts)
        variable2, weight2 = self._get_variable(variable2_name, query, track_cuts=track_cuts)

        heatmap, xedges, yedges = np.histogram2d(variable1, variable2,
                                                 range=[[plot_options["range_x"][0], plot_options["range_x"][1]], [plot_options["range_y"][0], plot_options["range_y"][1]]],
                                                 bins=[plot_options["bins_x"], plot_options["bins_y"]],
                                                 weights=weight1)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        #if figure is passed, use that to build plot
        if "figure" in plot_options:
            fig = plot_options["figure"]
        else:
            fig = plt.figure(figsize=(6,6))
        if "axis" in plot_options:
            axis = plot_options["axis"]
        else:
            axis = plt.gca()

        if 'range_z' in plot_options:
            image = axis.imshow(heatmap.T, extent=extent, origin='lower', aspect="auto",
                vmin=plot_options['range_z'][0], vmax=plot_options['range_z'][1])
        else:
            image = axis.imshow(heatmap.T, extent=extent, origin='lower', aspect="auto")

        return fig, axis, image


    def load_detsys_errors(self,sample,var,path,binedges,fullcov=False):

        detsys_frac_cov = np.zeros((len(binedges)-1,len(binedges)-1))
        detsys_frac_cov = detsys_frac_cov.astype(float)
        detsys_frac = np.zeros(len(binedges)-1)

        DETSAMPLES = ["X", "YZ", 'aYZ', "aXZ","R2","SCE","LYD","LYR","LYA"]
        #DETSAMPLES = ["X", "YZ", 'aYZ', "aXZ","dEdX","SCE","LYD","LYR","LYA"]

        #if (os.path.isdir(path) == False):
        #    #print ('DETSYS. path %s is not valid'%path)
        #    return detsys_frac

        anyfilefound = False
        for varsample in DETSAMPLES:

            # using only diagonal terms
            if (fullcov == False):

                filename = var + "_" + sample + "_" + varsample + ".txt"

                if (os.path.isfile(path+filename) == False):
                    #print ('file-name %s @ path %s is not valid'%(filename,path))
                    continue

                anyfilefound = True
                f = open(path+filename,'r')

                for binnumber in range(len(detsys_frac)):
                    
                    binmin = binedges[binnumber]
                    binmax = binedges[binnumber+1]
                    bincenter = 0.5*(binmin+binmax)
                    
                    # find uncertainty associated to this bin in the text-file

                    f.seek(0,0)
                
                    for line in f:

                        words = line.split(",")
                        binrange_v = words[0].split(" - ")
                        binrangemin = float(binrange_v[0])
                        binrangemax = float(binrange_v[1])

                        if ( (bincenter > binrangemin) and (bincenter <= binrangemax) ):
                    
                            fracerror = float(words[1].split()[0])
                            detsys_frac[binnumber] += fracerror * fracerror

                            break

            # using full covariance
            if (fullcov == True):

                filename = var + "_" + sample + "_" + varsample + "_COV.txt"

                if (os.path.isfile(path+filename) == False):
                    continue

                f = open(path+filename,'r')

                # import the fractional covariance matrix
                f.seek(0,0)
                matrix_binedge_v = []
                frac_cov_matrix = []
                linectr = 0
                for line in f:

                    # get bin edges
                    if (linectr == 0):
                        words = line.split(',')
                        wordctr = 0
                        words = words[:-1]
                        for word in words:
                            binrange_v = word.split("-")
                            #print ('binrange_v : ',binrange_v)
                            binrangemin = float(binrange_v[0])
                            binrangemax = float(binrange_v[1])
                            if (wordctr == 0):
                                matrix_binedge_v.append(binrangemin)
                                matrix_binedge_v.append(binrangemax)
                            else:
                                matrix_binedge_v.append(binrangemax)
                            wordctr += 1

                    # get cov matrix entries
                    if (linectr > 0):
                        words = line.split(',')
                        words = np.array(words[1:-1])
                        words = words.astype(float)
                        frac_cov_matrix.append(words)
                            
                    linectr += 1

                frac_cov_matrix = np.array(frac_cov_matrix)
                #print ('the full COV matrix for sample %s is : \n'%(varsample),frac_cov_matrix)

                # use of covariance matrix input required identical binning...let's check
                if (len(binedges) != len(matrix_binedge_v)):
                    print ('COV matrix and binning for plotter not identical [1]...exit')
                    continue

                binmatch = True
                for i,e in enumerate(binedges):
                    if ( np.abs(matrix_binedge_v[i] - e) > 0.001):
                        #print ('binedges[i] == %.05f and matrix_binedge_v[i] == %.05f'%(e,matrix_binedge_v[i]))
                        binmatch = False
                        break

                if (binmatch == False):
                    print ('COV matrix and binning for plotter not identical [2]...exit')
                    continue

                anyfilefound = True

                detsys_frac_cov += frac_cov_matrix

        '''
        if anyfilefound:
            detsys_frac = np.sqrt(np.array(detsys_frac))
            detsys_frac_cov = np.array(detsys_frac_cov)
            np.nan_to_num(detsys_frac, copy=False, nan=1)
            np.nan_to_num(detsys_frac, copy=False, nan=1)
        else:
            detsys_frac = 0.2*np.ones(len(binedges)-1)
            detsys_frac_cov = np.array(detsys_frac_cov)
            detsys_frac_cov[np.diag_indices_from(detsys_frac_cov)] = (0.2**2) * np.ones(len(binedges)-1)
        '''

        if (fullcov == False):
            detsys_frac_cov[np.diag_indices_from(detsys_frac_cov)] = np.array(detsys_frac)
            
        detsys_frac_cov = np.array(detsys_frac_cov)
            
        if (anyfilefound == True):
            np.nan_to_num(detsys_frac_cov, copy=False, nan=1)
            for i,j in zip(np.diag_indices_from(detsys_frac_cov)[0],np.diag_indices_from(detsys_frac_cov)[1]):
                if detsys_frac_cov[i][j] == 0:
                    detsys_frac_cov[i][j] = (0.2**2)
        else:
            detsys_frac_cov[np.diag_indices_from(detsys_frac_cov)] = (0.2**2) * np.ones(len(binedges)-1)
        
            
        #print (sample,': frac. detsys matrix is : \n', detsys_frac_cov)
        print (sample,': frac. detsys matrix is : \n', np.sqrt(np.diag(detsys_frac_cov)))
        return detsys_frac_cov

    def add_detsys_error(self,sample,mc_entries_v,weight):

        detsys_v  = np.zeros( (len(mc_entries_v),len(mc_entries_v)) )

        if (self.detsys == None): return detsys_v
        
        if sample in self.detsys:
            if (len(self.detsys[sample]) == len(mc_entries_v)):
                for i,n in enumerate(mc_entries_v):
                    for j,m in enumerate(mc_entries_v):
                        #print ('indices i, j : %i, %i'%(i,j))
                        #print ('entries n, m : %.02f, %.02f'%(n,m))
                        #print ('weight : ',weight) 
                        detsys_v[i][j] = (self.detsys[sample][i][j] * n * m * weight * weight)
            else:
                print ('NO MATCH! len detsys : %i. Len plotting : %i'%(len(self.detsys[sample]),len(mc_entries_v) ))

        #print ('mc entries for sample %s : \n'%(sample),mc_entries_v)
        #print ('weight for sample %s : '%sample,weight)
        #print ('detys frac. cov. matrix for sample %s : \n '%(sample),self.detsys[sample])
        #print ('detsys covariance matrix for sample %s : \n'%(sample), detsys_v)
                
        return detsys_v
    
    '''
    def add_detsys_error(self,sample,mc_entries_v,weight):
        detsys_v  = np.zeros(len(mc_entries_v))
        entries_v = np.zeros(len(mc_entries_v))
        if (self.detsys == None): return detsys_v
        if sample in self.detsys:
            if (len(self.detsys[sample]) == len(mc_entries_v)):
                for i,n in enumerate(mc_entries_v):
                    detsys_v[i] = (self.detsys[sample][i] * n * weight)#**2
                    entries_v[i] = n * weight
            else:
                print ('NO MATCH! len detsys : %i. Len plotting : %i'%(len(self.detsys[sample]),len(mc_entries_v) ))

        return detsys_v
    '''



    def plot_variable(self, variable, query="selected==1", title="", kind="event_category",
                      draw_sys=False, stacksort=0, track_cuts=None, select_longest=False,
                      detsysdict=None,ratio=True,chisq=False,draw_data=True,asymErrs=False,genieweight="weightSplineTimesTune",
                      fullcov=False,
                      predictedevents=True, # add number of predicted events for MC contributions to legend labels.
                      legendloc="best",
                      figtitle="MicroBooNE Preliminary", # title for legend
                      drawsystematics=True, # draw systematics error bar around MC
                      labeldecimals=2, # number of decimal places for y-axis label
                      ncol=2,
                      COVMATRIX='', # path to covariance matrix file
                      DETSYSPATH="", # path where to find detector systematics files
                      ACCEPTANCE="", # acceptance query to calculate xsec
                      **plot_options):
        """It plots the variable from the TTree, after applying an eventual query

        Args:
            variable (str): name of the variable.
            query (str): pandas query. Default is ``selected``.
            title (str, optional): title of the plot. Default is ``variable``.
            kind (str, optional): Categorization of the plot.
                Accepted values are ``event_category``, ``particle_pdg``, and ``sample``
                Default is ``event_category``.
            track_cuts (list of tuples (var, operation, cut val), optional):
                List of cuts ot be made on track-level variables ("_v" in variable name)
                These get applied one at a time in self._selection
            select_longest (bool): if variable is a track-level variable
                setting to True will take the longest track of each slice
                    after QUERY and track_cuts have been applied
                select_longest = False might have some bugs...
            **plot_options: Additional options for matplotlib plot (e.g. range and bins).

        Returns:
            Figure, top subplot, and bottom subplot (ratio)

        """

        self.detsys = {}

        if not title:
            title = variable
        if not query:
            query = "nslice==1"

        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if plot_options["range"]!=None and plot_options["range"][0] >= 0 and plot_options["range"][1] >= 0 and variable[-2:] != "_v":
            query += "& %s <= %g & %s >= %g" % (
                variable, plot_options["range"][1], variable, plot_options["range"][0])

        #eventually used to subdivide monte-carlo sample
        if kind == "event_category":
            categorization = self._categorize_entries
            cat_labels = category_labels
        elif kind == "trk1_backtracked_pdg":
            categorization = self._categorize_entries_trk1
            cat_labels = pdg_labels
        elif kind == "paper_category":
            categorization = self._categorize_entries_paper
            cat_labels = paper_labels
        elif kind == "paper_category_xsec":
            categorization = self._categorize_entries_paper_xsec
            cat_labels = paper_labels_xsec
        elif kind == "paper_category_numu":
            categorization = self._categorize_entries_paper_numu
            cat_labels = paper_labels_numu
        elif kind == "ccncpi0_category":
            categorization = self._categorize_entries_ccncpi0
            cat_labels = category_labels
        elif kind == "particle_pdg":
            var = self.samples["mc"].query(query).eval(variable)
            if var.dtype == np.float32:
                categorization = self._categorize_entries_single_pdg
            else:
                categorization = self._categorize_entries_pdg
            cat_labels = pdg_labels
        elif kind == "interaction":
            categorization = self._categorize_entries_int
            cat_labels = int_labels
        elif kind == "flux":
            categorization = self._categorize_entries_flux
            cat_labels = flux_labels
        elif kind == "sample":
            categorization = self._categorize_entries_sample
            cat_labels = sample_labels
            #return self._plot_variable_samples(variable, query, title, asymErrs, **plot_options)
        else:
            raise ValueError(
                "Unrecognized categorization, valid options are 'sample', 'event_category', and 'particle_pdg'")

        category, mc_plotted_variable = categorization(
            self.samples["mc"], variable, query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts, select_longest=select_longest)


        var_dict = defaultdict(list)
        weight_dict = defaultdict(list)
        mc_genie_weights = self._get_genie_weight(
            self.samples["mc"], variable, query=query, extra_cut=self.nu_pdg, track_cuts=track_cuts,select_longest=select_longest, weightvar=genieweight)

        for c, v, w in zip(category, mc_plotted_variable, mc_genie_weights):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["mc"] * w)

        nue_genie_weights = self._get_genie_weight(
            self.samples["nue"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)

        category, nue_plotted_variable = categorization(
            self.samples["nue"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

        for c, v, w in zip(category, nue_plotted_variable, nue_genie_weights):
            var_dict[c].append(v)
            weight_dict[c].append(self.weights["nue"] * w)

        if "ncpi0" in self.samples:
            ncpi0_genie_weights = self._get_genie_weight(
                    self.samples["ncpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ncpi0_plotted_variable = categorization(
                self.samples["ncpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ncpi0_plotted_variable, ncpi0_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ncpi0"] * w)

        if "ccpi0" in self.samples:
            ccpi0_genie_weights = self._get_genie_weight(
                    self.samples["ccpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ccpi0_plotted_variable = categorization(
                self.samples["ccpi0"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ccpi0_plotted_variable, ccpi0_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ccpi0"] * w)

        if "eta" in self.samples:
            eta_genie_weights = self._get_genie_weight(
                self.samples["eta"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, eta_plotted_variable = categorization(
                self.samples["eta"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, eta_plotted_variable, eta_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["eta"] * w)

        if "ccnopi" in self.samples:
            ccnopi_genie_weights = self._get_genie_weight(
                    self.samples["ccnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ccnopi_plotted_variable = categorization(
                self.samples["ccnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ccnopi_plotted_variable, ccnopi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ccnopi"] * w)

        if "cccpi" in self.samples:
            cccpi_genie_weights = self._get_genie_weight(
                    self.samples["cccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, cccpi_plotted_variable = categorization(
                self.samples["cccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, cccpi_plotted_variable, cccpi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["cccpi"] * w)

        if "nccpi" in self.samples:
            nccpi_genie_weights = self._get_genie_weight(
                    self.samples["nccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, nccpi_plotted_variable = categorization(
                self.samples["nccpi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, nccpi_plotted_variable, nccpi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["nccpi"] * w)

        if "ncnopi" in self.samples:
            ncnopi_genie_weights = self._get_genie_weight(
                    self.samples["ncnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, ncnopi_plotted_variable = categorization(
                self.samples["ncnopi"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, ncnopi_plotted_variable, ncnopi_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["ncnopi"] * w)

        if "dirt" in self.samples:
            dirt_genie_weights = self._get_genie_weight(
                self.samples["dirt"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest, weightvar=genieweight)
            category, dirt_plotted_variable = categorization(
                self.samples["dirt"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)

            for c, v, w in zip(category, dirt_plotted_variable, dirt_genie_weights):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["dirt"] * w)

        if "lee" in self.samples:
            category, lee_plotted_variable = categorization(
                self.samples["lee"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest)
            #print ('weight 1 : ',len(self.samples["lee"].query(query)["leeweight"]))
            #print ('weight 2 : ',len(self._selection("weightSplineTimesTune", self.samples["lee"], query=query, track_cuts=track_cuts, select_longest=select_longest)))
            #print ('track cuts : ',track_cuts)
            #print ('select_longest : ',select_longest)
            leeweight = self._get_genie_weight(
                self.samples["lee"], variable, query=query, track_cuts=track_cuts, select_longest=select_longest,weightsignal="leeweight", weightvar=genieweight)
            #self.samples["lee"].query(query)["leeweight"] * self._selection("weightSplineTimesTune", self.samples["lee"], query=query, track_cuts=track_cuts, select_longest=select_longest)
            
            for c, v, w in zip(category, lee_plotted_variable, leeweight):
                var_dict[c].append(v)
                weight_dict[c].append(self.weights["lee"] * w)

            lee_hist, lee_bins = np.histogram(
                var_dict[111],
                bins=plot_options["bins"],
                range=plot_options["range"],
                weights=weight_dict[111])

        if draw_data:
            ext_plotted_variable = self._selection(
                variable, self.samples["ext"], query=query, track_cuts=track_cuts, select_longest=select_longest)
            ext_plotted_variable = self._select_showers(
            ext_plotted_variable, variable, self.samples["ext"], query=query)
            data_plotted_variable = self._selection(
            variable, self.samples["data"], query=query, track_cuts=track_cuts, select_longest=select_longest)
            data_plotted_variable = self._select_showers(data_plotted_variable, variable,
                                                     self.samples["data"], query=query)
            #### for paper add EXT to the stacked plot
            if (kind == "paper_category" or kind == "paper_category_xsec" or kind == "paper_category_numu"):
                var_dict[100] = ext_plotted_variable
                ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)
                weight_dict[100] = ext_weight
                cat_labels[100] = "Cosmics"
                #category_colors[100] = "xkcd:cerulean"
                category_colors[100] = "xkcd:greyish blue"
                n_ext, dummy = np.histogram(ext_plotted_variable,bins=plot_options["bins"],
                                   range=plot_options["range"],weights=ext_weight)
                self.ext = n_ext


        if ratio:
            fig = plt.figure(figsize=(8, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
        else:
            fig = plt.figure(figsize=(7, 5))
            gs = gridspec.GridSpec(1, 1)#, height_ratios=[2, 1])
            ax1 = plt.subplot(gs[0])



        # order stacked distributions
        order_dict = {}
        order_var_dict    = {}
        order_weight_dict = {}
        if (stacksort >= 1 and stacksort <= 3):
            # figure out ordering based on total yield.
            # Options are to have no exceptions (stacksort=1),
            # put eLEE on top (stacksort=2), or put nue+eLEE on top (stacksort=3)
            # put numu on top (stacksort >= 4)
            has1 = False
            has10 = False
            has11 = False
            has111 = False
            for c in var_dict.keys():
                if stacksort >= 2:
                    if int(c)==111:
                        has111 = True
                        continue
                if stacksort == 3:
                    if int(c)==1:
                        has1 = True
                        continue
                    if int(c)==10:
                        has10 = True
                        continue
                    if int(c)==11:
                        has11 = True
                        continue
                order_dict[c] = sum(weight_dict[c])
                order_dict = {k: v for k, v in sorted(order_dict.items(), key=lambda item: item[1])}
            if has1:
                order_dict[1] = sum(weight_dict[1])
            if has10:
                order_dict[10] = sum(weight_dict[10])
            if has11:
                order_dict[11] = sum(weight_dict[11])
            #### for paper do not add lee to stack plot
            if has111 and kind != "paper_category" and kind != "paper_category_xsec" and kind != "paper_category_numu":
                order_dict[111] = sum(weight_dict[111])
            # now that the order has been sorted out, fill the actual dicts
            for c in order_dict.keys():
                order_var_dict[c] = var_dict[c]
            for c in order_dict.keys():
                order_weight_dict[c] = weight_dict[c]
        elif stacksort == 4:
            #put the numu stuff on top
            hasprotons = 23 in var_dict.keys()
            keys = list(var_dict.keys())
            if hasprotons:
                keys.remove(22)#take them out
                keys.remove(23)
                keys.remove(24)
                keys.remove(25)
                keys.append(22)#and put at end
                keys.append(23)
                keys.append(24)
                keys.append(25)

            for c in keys:
                order_var_dict[c] = var_dict[c]
                order_weight_dict[c] = weight_dict[c]
        else:
            for c in var_dict.keys():
                order_var_dict[c] = var_dict[c]
            for c in weight_dict.keys():
                order_weight_dict[c] = weight_dict[c]

        total = sum(sum(order_weight_dict[c]) for c in order_var_dict)
        if draw_data:
            total += sum([self.weights["ext"]] * len(ext_plotted_variable))
        if (predictedevents == True):
            labels = [
                "%s: %.1f" % (cat_labels[c], sum(order_weight_dict[c])) \
                if sum(order_weight_dict[c]) else ""
                for c in order_var_dict.keys()
            ]
        else:
            labels = [
                "%s" % (cat_labels[c]) \
                if sum(order_weight_dict[c]) else ""
                for c in order_var_dict.keys()
            ]

        if kind == "event_category" or kind == "paper_category" or kind == "paper_category_xsec" or kind == "paper_category_numu" or kind == "ccncpi0_category":
            plot_options["color"] = [category_colors[c]
                                     for c in order_var_dict.keys()]
        elif kind == "particle_pdg":
            plot_options["color"] = [pdg_colors[c]
                                     for c in order_var_dict.keys()]
        elif kind == "trk1_backtracked_pdg":
            plot_options["color"] = [pdg_colors[c]
                                     for c in order_var_dict.keys()]
        elif kind == "flux":
            plot_options["color"] = [flux_colors[c]
                                     for c in order_var_dict.keys()]
        else:
            plot_options["color"] = [int_colors[c]
                                     for c in order_var_dict.keys()]

        #for key in order_var_dict:
        #    print ('key ',key)
        #    print ('val ',order_var_dict[key])
        #for key in order_weight_dict:
        #    print ('key ',key)
        #    print ('val ',order_weight_dict[key])

        stacked = ax1.hist(
            order_var_dict.values(),
            weights=list(order_weight_dict.values()),
            stacked=True,
            label=labels,
            **plot_options)

        total_array = np.concatenate(list(order_var_dict.values()))
        total_weight = np.concatenate(list(order_weight_dict.values()))

        #print(stacked)
        #print(labels)

        plot_options.pop('color', None)

        total_hist, total_bins = np.histogram(
            total_array, weights=total_weight,  **plot_options)

        #### for paper do not draw EXT on top of stack plot
        if draw_data and kind != "paper_category" and kind != "paper_category_xsec" and kind!= "paper_category_numu":
            ext_weight = [self.weights["ext"]] * len(ext_plotted_variable)
            extlabel = "Cosmic" if sum(ext_weight) else ""
            if (predictedevents == True): extlabel="EXT: %.1f" % sum(ext_weight) if sum(ext_weight) else ""
            n_ext, ext_bins, patches = ax1.hist(
            ext_plotted_variable,
            weights=ext_weight,
            bottom=total_hist,
            label=extlabel,
            hatch="//",
            color="white",
            **plot_options)
            total_array = np.concatenate([total_array, ext_plotted_variable])
            total_weight = np.concatenate([total_weight, ext_weight])

        #### for paper draw lee as dashed line on top of stack plot
        '''
        if kind == "paper_category":
            lee_tot_array = np.concatenate([total_array,var_dict[111]])
            lee_tot_weight = np.concatenate([total_weight,weight_dict[111]])
            n_tot_lee, bin_edges_lee, patches_lee = ax1.hist(
                lee_tot_array,
                weights=lee_tot_weight,
                histtype="step",
                edgecolor="red",#category_colors[111],
                linestyle="--",
                linewidth=2,
                label="eLEE: %.1f" % sum(weight_dict[111]) if sum(weight_dict[111]) else "",
                **plot_options)
        '''

        n_tot, bin_edges, patches = ax1.hist(
        total_array,
        weights=total_weight,
        histtype="step",
        edgecolor="black",
        **plot_options)

        #print(n_tot)
        #print(np.sum(n_tot))

        summarydict = {}
        if 0:#kind == "paper_category":
            for c in order_var_dict.keys():
                summarydict[c] = {'cat' : cat_labels[c], 'val' : sum(order_weight_dict[c])}
            summarydict[100] = {'cat' : 'ext', 'val' : sum(n_ext)}
            summarydict[111] = {'cat' : 'lee', 'val' : sum(weight_dict[111])}
            #print(summarydict)

        bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mcw2 = mc_genie_weights * mc_genie_weights * self.weights["mc"] * self.weights["mc"]
        err_mc, bins = np.histogram(mc_plotted_variable, weights=mcw2, **plot_options)
        if ("mc" in detsysdict.keys() and detsysdict["mc"]==True):
            self.detsys["mc"] = self.load_detsys_errors("mc",variable,DETSYSPATH,bin_edges,fullcov)
        mc_pred_nopotw, bins = np.histogram(mc_plotted_variable, weights=mc_genie_weights, **plot_options)
        sys_mc = self.add_detsys_error("mc",mc_pred_nopotw,self.weights["mc"])

        nuew2 = nue_genie_weights * nue_genie_weights * self.weights["nue"] * self.weights["nue"]
        err_nue, bins = np.histogram(nue_plotted_variable, weights=nuew2, **plot_options)
        if ("nue" in detsysdict.keys() and detsysdict["nue"]==True):
            self.detsys["nue"] = self.load_detsys_errors("nue",variable,DETSYSPATH,bin_edges,fullcov)
        nue_pred_nopotw, bins = np.histogram(nue_plotted_variable, weights=nue_genie_weights, **plot_options)
        sys_nue = self.add_detsys_error("nue",nue_pred_nopotw,self.weights["nue"])

        err_dirt = np.array([0 for n in err_mc])
        sys_dirt = np.array([0 for n in err_mc])
        if "dirt" in self.samples:
            dirtw2 = dirt_genie_weights * dirt_genie_weights * self.weights["dirt"] * self.weights["dirt"]
            err_dirt, bins = np.histogram(dirt_plotted_variable, weights=dirtw2, **plot_options)
            if ("dirt" in detsysdict.keys() and detsysdict["dirt"]==True):
                self.detsys["dirt"] = self.load_detsys_errors("dirt",variable,DETSYSPATH,bin_edges,fullcov)
            dirt_pred_nopotw, bins = np.histogram(dirt_plotted_variable, weights=dirt_genie_weights, **plot_options)
            sys_dirt = self.add_detsys_error("dirt",dirt_pred_nopotw,self.weights["dirt"])

        err_lee = np.array([0 for n in err_mc])
        sys_lee = np.array([0 for n in err_mc])
        if "lee" in self.samples:
            #fixme?
            lee_uncertainties, bins = np.histogram(lee_plotted_variable, weights=leeweight, **plot_options)
            if isinstance(plot_options["bins"], Iterable):
                lee_bins = plot_options["bins"]
            else:
                bin_size = (plot_options["range"][1] - plot_options["range"][0])/plot_options["bins"]
                lee_bins = [plot_options["range"][0]+n*bin_size for n in range(plot_options["bins"]+1)]

            if variable[-2:] != "_v":
                binned_lee = pd.cut(self.samples["lee"].query(query).eval(variable), lee_bins)
                err_lee = self.samples["lee"].query(query).groupby(binned_lee)['leeweight'].agg(
                    "sum").values * self.weights["lee"] * self.weights["lee"]
            if ("lee" in detsysdict.keys() and detsysdict["lee"]==True):
                self.detsys["lee"] = self.load_detsys_errors("lee",variable,DETSYSPATH,bin_edges,fullcov)
            sys_lee = self.add_detsys_error("lee",lee_uncertainties,self.weights["lee"])

        err_ncpi0 = np.array([0 for n in err_mc])
        sys_ncpi0 = np.array([0 for n in err_mc])
        if "ncpi0" in self.samples:
            ncpi0w2 = ncpi0_genie_weights * ncpi0_genie_weights * self.weights["ncpi0"] * self.weights["ncpi0"]
            err_ncpi0, bins = np.histogram(ncpi0_plotted_variable, weights=ncpi0w2, **plot_options)
            if ("ncpi0" in detsysdict.keys() and detsysdict["ncpi0"]==True):
                self.detsys["ncpi0"] = self.load_detsys_errors("ncpi0",variable,DETSYSPATH,bin_edges,fullcov)
            ncpi0_pred_nopotw, bins = np.histogram(ncpi0_plotted_variable, weights=ncpi0_genie_weights, **plot_options)
            sys_ncpi0 = self.add_detsys_error("ncpi0",ncpi0_pred_nopotw,self.weights["ncpi0"])

        err_ccpi0 = np.array([0 for n in err_mc])
        sys_ccpi0 = np.array([0 for n in err_mc])
        if "ccpi0" in self.samples:
            ccpi0w2 = ccpi0_genie_weights * ccpi0_genie_weights * self.weights["ccpi0"] * self.weights["ccpi0"]
            err_ccpi0, bins = np.histogram(ccpi0_plotted_variable, weights=ccpi0w2, **plot_options)
            if ("ccpi0" in detsysdict.keys() and detsysdict["ccpi0"]==True):
                self.detsys["ccpi0"] = self.load_detsys_errors("ccpi0",variable,DETSYSPATH,bin_edges,fullcov)
            ccpi0_pred_nopotw, bins = np.histogram(ccpi0_plotted_variable, weights=ccpi0_genie_weights, **plot_options)
            sys_ccpi0 = self.add_detsys_error("ccpi0",ccpi0_pred_nopotw,self.weights["ccpi0"])

        err_eta = np.array([0 for n in err_mc])
        sys_eta = np.array([0 for n in err_mc])
        if "eta" in self.samples:
            eta0w2 = eta_genie_weights * eta_genie_weights * self.weights["eta"] * self.weights["eta"]
            err_eta, bins = np.histogram(eta_plotted_variable, weights=etaw2, **plot_options)
            if ("eta" in detsysdict.keys() and detsysdict["eta"]==True):
                self.detsys["eta"] = self.load_detsys_errors("eta",variable,DETSYSPATH,bin_edges,fullcov)
            eta_pred_nopotw, bins = np.histogram(eta_plotted_variable, weights=eta_genie_weights, **plot_options)
            sys_eta = self.add_detsys_error("eta",eta_uncertainties,self.weights["eta"])

        err_ccnopi = np.array([0 for n in err_mc])
        sys_ccnopi = np.array([0 for n in err_mc])
        if "ccnopi" in self.samples:
            ccnopiw2 = ccnopi_genie_weights * ccnopi_genie_weights * self.weights["ccnopi"] * self.weights["ccnopi"]
            err_ccnopi, bins = np.histogram(ccnopi_plotted_variable, weights=ccnopiw2, **plot_options)
            if ("ccnopi" in detsysdict.keys() and detsysdict["ccnopi"]==True):
                self.detsys["ccnopi"] = self.load_detsys_errors("ccnopi",variable,DETSYSPATH,bin_edges,fullcov)
            ccnopi_pred_nopotw, bins = np.histogram(ccnopi_plotted_variable, weights=ccnopi_genie_weights, **plot_options)
            sys_ccnopi = self.add_detsys_error("ccnopi",ccnopi_pred_nopotw,self.weights["ccnopi"])

        err_cccpi = np.array([0 for n in err_mc])
        sys_cccpi = np.array([0 for n in err_mc])
        if "cccpi" in self.samples:
            cccpiw2 = cccpi_genie_weights * cccpi_genie_weights * self.weights["cccpi"] * self.weights["cccpi"]
            err_cccpi, bins = np.histogram(cccpi_plotted_variable, weights=cccpiw2, **plot_options)
            if ("cccpi" in detsysdict.keys() and detsysdict["cccpi"]==True):
                self.detsys["cccpi"] = self.load_detsys_errors("cccpi",variable,DETSYSPATH,bin_edges,fullcov)
            cccpi_pred_nopotw, bins = np.histogram(cccpi_plotted_variable, weights=cccpi_genie_weights, **plot_options)
            sys_cccpi = self.add_detsys_error("cccpi",cccpi_pred_nopotw,self.weights["cccpi"])

        err_nccpi = np.array([0 for n in err_mc])
        sys_nccpi = np.array([0 for n in err_mc])
        if "nccpi" in self.samples:
            nccpiw2 = nccpi_genie_weights * nccpi_genie_weights * self.weights["nccpi"] * self.weights["nccpi"]
            err_nccpi, bins = np.histogram(nccpi_plotted_variable, weights=nccpiw2, **plot_options)
            if ("nccpi" in detsysdict.keys() and detsysdict["nccpi"]==True):
                self.detsys["nccpi"] = self.load_detsys_errors("nccpi",variable,DETSYSPATH,bin_edges,fullcov)
            nccpi_pred_nopotw, bins = np.histogram(nccpi_plotted_variable, weights=nccpi_genie_weights, **plot_options)
            sys_nccpi = self.add_detsys_error("nccpi",nccpi_pred_nopotw,self.weights["nccpi"])

        err_ncnopi = np.array([0 for n in err_mc])
        sys_ncnopi = np.array([0 for n in err_mc])
        if "ncnopi" in self.samples:
            ncnopiw2 = ncnopi_genie_weights * ncnopi_genie_weights * self.weights["ncnopi"] * self.weights["ncnopi"]
            err_ncnopi, bins = np.histogram(ncnopi_plotted_variable, weights=ncnopiw2, **plot_options)
            if ("ncnopi" in detsysdict.keys() and detsysdict["ncnopi"]==True):
                self.detsys["ncnopi"] = self.load_detsys_errors("ncnopi",variable,DETSYSPATH,bin_edges,fullcov)
            ncnopi_pred_nopotw, bins = np.histogram(ncnopi_plotted_variable, weights=ncnopi_genie_weights, **plot_options)
            sys_ncnopi = self.add_detsys_error("ncnopi",ncnopi_pred_nopotw,self.weights["ncnopi"])

        if draw_data:
            err_ext = np.array(
                [n * self.weights["ext"] for n in n_ext]) #here n is already weighted
            err_ext[err_ext==0] = (1.4*self.weights["ext"])**2
            self.ext_err = np.sqrt(err_ext)
        else:
            err_ext = np.zeros(len(err_mc))

        exp_err    = np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_eta + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #print("counting_err: {}".format(exp_err))
        detsys_err = sys_mc + sys_nue + sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_eta + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi
        #print ('detsys covariance matrix is : \n',detsys_err)
        #print("detsys_err: {}".format(detsys_err))
        exp_err = np.sqrt(exp_err**2 + np.diag(detsys_err))#**2)
        #print ('total exp_err : ', exp_err)

        bin_size = [(bin_edges[i + 1] - bin_edges[i]) / 2
                    for i in range(len(bin_edges) - 1)]

        self.cov           = np.zeros([len(exp_err), len(exp_err)])
        self.cov_mc_stat   = np.zeros([len(exp_err), len(exp_err)])
        self.cov_mc_detsys = np.zeros([len(exp_err), len(exp_err)])
        self.cov_data_stat = np.zeros([len(exp_err), len(exp_err)])

        self.cov_mc_stat[np.diag_indices_from(self.cov_mc_stat)]     = (err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_eta + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi)
        #self.cov_mc_detsys[np.diag_indices_from(self.cov_mc_detsys)] = (sys_mc + sys_nue + sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_eta + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)**2
        self.cov_mc_detsys = (sys_mc + sys_nue + sys_dirt + sys_ncpi0 + sys_ccpi0 + sys_eta + sys_ccnopi + sys_cccpi + sys_nccpi + sys_ncnopi)
                        

        if draw_sys:
            #cov = self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], "weightSplineTimesTune")

            if (COVMATRIX == ""):
                '''
                self.cov = ( self.sys_err("weightsReint", variable, query, plot_options["range"], plot_options["bins"], genieweight) )
                '''
                self.cov = ( self.sys_err("weightsGenie", variable, query, plot_options["range"], plot_options["bins"], genieweight) + \
                             self.sys_err("weightsFlux", variable, query, plot_options["range"], plot_options["bins"], genieweight) + \
                             self.sys_err("weightsReint", variable, query, plot_options["range"], plot_options["bins"], genieweight) + \
                             self.sys_err_unisim(variable, query, plot_options["range"], plot_options["bins"], genieweight)
                            )

                # for calculating the cross-section uncertainty [not needed for PeLEE]
                #self.xsec_err("weightsFlux",variable, query, plot_options["range"],1,genieweight,ACCEPTANCE)
                #self.xsec_err("weightsGenie",variable, query, plot_options["range"],1,genieweight,ACCEPTANCE)
                #self.xsec_err("weightsReint",variable, query, plot_options["range"],1,genieweight,ACCEPTANCE)

            else:
                self.cov = self.get_SBNFit_cov_matrix(COVMATRIX,len(bin_edges)-1)
            exp_err = np.sqrt( np.diag((self.cov + self.cov_mc_stat + self.cov_mc_detsys))) # + exp_err*exp_err)
        else:
            exp_err = np.sqrt( np.diag(self.cov_mc_stat) )
                                    
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


        if 0:#kind == "paper_category":
            '''
            11: r"$\nu_e$ CC",
            111: r"MiniBooNE LEE",
            2: r"$\nu$ other",
            31: r"$\nu$ with $\pi^{0}$",
            5: r"dirt",
            '''
            if 11 in summarydict.keys():
                nue_covF = self.sys_err("weightsFlux", variable, query+" and paper_category==11",plot_options["range"], 1,genieweight)
                nue_covG = self.sys_err("weightsGenie", variable, query+" and paper_category==11",plot_options["range"],1,genieweight)
                nue_covR = self.sys_err("weightsReint", variable, query+" and paper_category==11",plot_options["range"],1,genieweight)
                nue_covU = self.sys_err_unisim(variable, query+" and paper_category==11", plot_options["range"], 1, genieweight)
                summarydict[11]['err2'] = np.diag(nue_covF).sum() + np.diag(nue_covG).sum() + \
                                          np.diag(nue_covR).sum() + np.diag(nue_covU).sum() + \
                                          np.diag(err_nue).sum() + np.diag(sys_nue**2).sum()
            if 5 in summarydict.keys():
                drt_covF = self.sys_err("weightsFlux", variable, query+" and paper_category==5",plot_options["range"], 1,genieweight)
                drt_covG = self.sys_err("weightsGenie", variable, query+" and paper_category==5",plot_options["range"],1,genieweight)
                drt_covR = self.sys_err("weightsReint", variable, query+" and paper_category==5",plot_options["range"],1,genieweight)
                drt_covU = self.sys_err_unisim(variable, query+" and paper_category==5", plot_options["range"], 1, genieweight)
                summarydict[5]['err2'] = np.diag(drt_covF).sum() + np.diag(drt_covG).sum() + \
                                          np.diag(drt_covR).sum() + np.diag(drt_covU).sum() + \
                                          np.diag(err_dirt).sum() + np.diag(sys_dirt**2).sum()
            if 111 in summarydict.keys():
                lee_covF = self.sys_err("weightsFlux", variable, query+" and paper_category==111",plot_options["range"], 1,genieweight,islee=True)
                lee_covG = self.sys_err("weightsGenie", variable, query+" and paper_category==111",plot_options["range"],1,genieweight,islee=True)
                lee_covR = self.sys_err("weightsReint", variable, query+" and paper_category==111",plot_options["range"],1,genieweight,islee=True)
                lee_covU = self.sys_err_unisim(variable, query+" and category==111", plot_options["range"], 1, genieweight,islee=True)
                summarydict[111]['err2'] = np.diag(lee_covF).sum() + np.diag(lee_covG).sum() + \
                                           np.diag(lee_covR).sum() + np.diag(lee_covU).sum() + \
                                           np.diag(err_lee).sum() + np.diag(sys_lee**2).sum()
            if 31 in summarydict.keys():
                pi0_covF = self.sys_err("weightsFlux", variable, query+" and paper_category==31",plot_options["range"], 1,genieweight)
                pi0_covG = self.sys_err("weightsGenie", variable, query+" and paper_category==31",plot_options["range"],1,genieweight)
                pi0_covR = self.sys_err("weightsReint", variable, query+" and paper_category==31",plot_options["range"],1,genieweight)
                pi0_covU = self.sys_err_unisim(variable, query+" and paper_category==31", plot_options["range"], 1, genieweight)
                summarydict[31]['err2'] = np.diag(pi0_covF).sum() + np.diag(pi0_covG).sum() +\
                                          np.diag(pi0_covR).sum() + np.diag(pi0_covU).sum() + \
                                          np.diag(err_ncpi0).sum() + np.diag(err_ccpi0).sum() + \
                                          np.diag(sys_ncpi0**2).sum() + np.diag(sys_ccpi0**2).sum()
            if 2 in summarydict.keys():
                oth_covF = self.sys_err("weightsFlux", variable, query+" and paper_category==2",plot_options["range"], 1,genieweight)
                oth_covG = self.sys_err("weightsGenie", variable, query+" and paper_category==2",plot_options["range"],1,genieweight)
                oth_covR = self.sys_err("weightsReint", variable, query+" and paper_category==2",plot_options["range"],1,genieweight)
                oth_covU = self.sys_err_unisim(variable, query+" and paper_category==2", plot_options["range"], 1, genieweight)
                summarydict[2]['err2'] = np.diag(oth_covF).sum() + np.diag(oth_covG).sum() + \
                                         np.diag(oth_covR).sum() + np.diag(oth_covU).sum() + \
                                         np.diag(err_mc).sum() + np.diag(err_ccnopi).sum() + np.diag(err_cccpi).sum() + \
                                         np.diag(err_nccpi).sum() + np.diag(err_ncnopi).sum() + \
                                         np.diag(sys_mc**2).sum() + np.diag(sys_ccnopi**2).sum() + np.diag(sys_cccpi**2).sum() + \
                                         np.diag(sys_nccpi**2).sum() + np.diag(sys_ncnopi**2).sum()
            if 100 in summarydict.keys():
                summarydict[100]['err2'] = err_ext.sum()
            for v in summarydict.values():
                if (v['val']>0 and 'err2' in v): print(v['cat'],v['val'],np.sqrt(v['err2']))

        if "lee" in self.samples:
            if kind == "event_category":
                try:
                    #SCALE = 1.01e21/self.pot
                    SCALE = 1.0
                    self.significance = self._sigma_calc_matrix(
                        lee_hist, n_tot-lee_hist, scale_factor=SCALE, cov=(self.cov+self.cov_mc_stat))
                    self.significance_likelihood = self._sigma_calc_likelihood(
                        lee_hist, n_tot-lee_hist, np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_eta + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi), scale_factor=SCALE)
                    # area normalized version
                    #normLEE = 68. / np.sum(n_tot)
                    #normSM  = 68. / np.sum(n_tot-lee_hist)
                    #self.significance_likelihood = self._sigma_calc_likelihood(
                    #    lee_hist * normLEE, (n_tot-lee_hist) * normSM, np.sqrt(normSM) * np.sqrt(err_mc + err_ext + err_nue + err_dirt + err_ncpi0 + err_ccpi0 + err_eta + err_ccnopi + err_cccpi + err_nccpi + err_ncnopi), scale_factor=1.0)
                except (np.linalg.LinAlgError, ValueError) as err:
                    print("Error calculating the significance", err)
                    self.significance = -1
                    self.significance_likelihood = -1
        # old error-bar plotting
        #ax1.bar(bincenters, n_tot, facecolor='none',
        #       edgecolor='none', width=0, yerr=exp_err)
        # DC0721
        if (drawsystematics == True):
            ax1.bar(bincenters, exp_err*2,width=[n*2 for n in bin_size],facecolor='gray',alpha=0.35,bottom=(n_tot-exp_err),label='Uncertainty')
        #ax1.errorbar(bincenters,n_tot,yerr=exp_err,fmt='k.',lw=35,alpha=0.2)
        '''
        ax1.fill_between(
            bincenters+(bincenters[1]-bincenters[0])/2.,
            n_tot-exp_err,
            n_tot+exp_err,
            step="pre",
            color="grey",
            alpha=0.5)
        '''

        self.prediction = n_tot
        if draw_data:
            n_data, bins = np.histogram(data_plotted_variable, **plot_options)
            self.data = n_data
            data_err = self._data_err(n_data,asymErrs)

            self.cov_data_stat[np.diag_indices_from(self.cov_data_stat)] = n_data

        self.cov_data_stat[np.diag_indices_from(self.cov_data_stat)] = n_data

        if (predictedevents==True):
            datalabel = "BNB Data: %i" % len(data_plotted_variable)
        else:
            datalabel = "BNB Data"
            

        if sum(n_data) > 0:
            ax1.errorbar(
                bincenters,
                n_data,
                xerr=bin_size,
                yerr=data_err,
                fmt='ko',
                label=datalabel if len(data_plotted_variable) else "")

        #frac = self.deltachisqfakedata(plot_options["range"][0], plot_options["range"][-1], np.array([1,1,1,5,5,5]), np.array([1,1,1,5,5,5]), 70)
        #self.sigma_shapeonly = self.deltachisqfakedata(plot_options["range"][0], plot_options["range"][-1], n_tot, (n_tot-lee_hist), 220)

        if (draw_sys):

            #chisq = self._chisquare(n_data, n_tot, exp_err)
            #self.stats['chisq'] = chisq
            chisqCNP = self._chisq_CNP(n_data,n_tot)
            #self.stats['chisqCNP'] = chisqCNP
            #print ('chisq for data/mc agreement with diagonal terms only : %.02f'%(chisq))
            #print ('chisq for data/mc agreement with diagonal terms only : %.02f'%(self._chisquare(n_data, n_tot, np.sqrt(np.diag(cov)))))
            chistatonly, aab, aac = self._chisq_full_covariance(n_data,n_tot,CNP=True,STATONLY=True)
            #chiarea, aab, aac = self._chisq_full_covariance(n_tot-lee_hist,n_tot,CNP=True,STATONLY=True,AREANORMED=True)
            chicov, chinocov,dof = self._chisq_full_covariance(n_data,n_tot,CNP=True)#,USEFULLCOV=True)
            if "lee" in self.samples: chilee, chileenocov,dof = self._chisq_full_covariance(n_tot-lee_hist,n_tot,CNP=True)
            #self.stats['chisq full covariance'] = chicov
            #self.stats['chisq full covariance (diagonal only)'] = chinocov
            self.stats['dof']            = dof
            self.stats['chisqstatonly']  = chistatonly
            #self.stats['chiarea']  = chiarea
            self.stats['pvaluestatonly'] = (1 - scipy.stats.chi2.cdf(chistatonly,dof))
            self.stats['chisqdiag']     = chinocov
            self.stats['pvaluediag']     = (1 - scipy.stats.chi2.cdf(chinocov,dof))
            #self.stats['parea']          = (1 - scipy.stats.chi2.cdf(chiarea,dof))
            self.stats['chisq']          = chicov
            #self.stats['chilee']          = chilee
            self.stats['pvalue']         = (1 - scipy.stats.chi2.cdf(chicov,dof))
            if "lee" in self.samples: self.stats['pvaluelee']         = (1 - scipy.stats.chi2.cdf(chilee,dof))
            #print ('chisq for data/mc agreement with full covariance is : %.02f. without cov : %.02f'%(chicov,chinocov))

            #self.print_stats()
        #'''
        if (ncol > 3):
            leg = ax1.legend(
                frameon=False, ncol=4, title=r'%s %.2f $\times 10^{20}$ POT' % (figtitle,(self.pot/1e20)),
                fontsize=12,loc=legendloc,
                prop={'size': fig.get_figwidth()})
        else:
            leg = ax1.legend(
                frameon=False, ncol=2, title=r'%s %.2f $\times 10^{20}$ POT' % (figtitle,(self.pot/1e20)),fontsize=12,loc=legendloc)
        leg._legend_box.align = "center"
        plt.setp(leg.get_title(), fontweight='normal',fontsize=12)#14
        #'''

        unit = title[title.find("[") +
                     1:title.find("]")] if "[" and "]" in title else ""
        if plot_options["range"] != None:
            x_range = plot_options["range"][1] - plot_options["range"][0]
            NNN = round(x_range / plot_options["bins"],labeldecimals)
        if (labeldecimals == 0):
            NNN = int(NNN)
        if isinstance(plot_options["bins"], Iterable):
            ax1.set_ylabel("Entries",fontsize=18)
        else:
            ax1.set_ylabel(
                "Entries / %s %s" % (str(NNN), unit),fontsize=18)

        if (ratio==True):
            ax1.set_xticks([])

        ax1.set_xlim(bin_edges[0], bin_edges[-1])

        '''
        ax1.fill_between(
            bincenters+(bincenters[1]-bincenters[0])/2.,
            n_tot - exp_err,
            n_tot + exp_err,
            step="pre",
            color="grey",
            alpha=0.5)
        '''

        if (chisq==True):
            if sum(n_data) > 0:

                nd = sum(n_data)
                nm = sum(n_tot)
                cov1b = self.sys_err("weightsFlux", variable, query, (bin_edges[0], bin_edges[-1]), 1, genieweight) + \
                        self.sys_err("weightsGenie", variable, query, (bin_edges[0], bin_edges[-1]), 1, genieweight) + \
                        self.sys_err("weightsReint", variable, query, (bin_edges[0], bin_edges[-1]), 1, genieweight) + \
                        self.sys_err_unisim(variable, query, (bin_edges[0], bin_edges[-1]), 1, genieweight)
                mcnormerr2 = np.diag(cov1b).sum() + np.diag(self.cov_mc_stat).sum() + np.diag(self.cov_mc_detsys).sum()
                datanormerr2 = (0.5*(self._data_err([nd],asymErrs)[0][0]+\
                                     self._data_err([nd],asymErrs)[1][0]))**2
                ax1.text(
                    0.19,
                    0.89,
                    r'$\chi^2 /$n.d.f. = %.2f' % (self.stats['chisq']/self.stats['dof']), #+
                             #'K.S. prob. = %.2f' % scipy.stats.ks_2samp(n_data, n_tot)[1],
                             #', p = %.2f' % (1 - scipy.stats.chi2.cdf(self.stats['chisq'],self.stats['dof'])) +
                             #'\n'+'Obs/Pred = %.2f' % (nd/nm) +
                             #' $\pm$ %.2f' % ((nd/nm)*np.sqrt(datanormerr2/(nd**2) + mcnormerr2/(nm**2))),
                    va='center',
                    ha='center',
                    ma='left',
                    fontsize=12,#12
                    transform=ax1.transAxes)

        if (ratio==True):
            if draw_data == False:
                n_data = np.zeros(len(n_tot))
                data_err = (np.zeros(len(n_tot)),np.zeros(len(n_tot)))
            else:
                self.chisqdatamc = self._chisquare(n_data, n_tot, exp_err)
            self._draw_ratio(ax2, bins, n_tot, n_data, exp_err, data_err)

        if (ratio==True):
            ax2.set_xlabel(title,fontsize=18)
            ax2.set_xlim(bin_edges[0], bin_edges[-1])
        else:
            ax1.set_xlabel(title,fontsize=18)

        fig.tight_layout()
        if title == variable:
            ax1.set_title(query)
        #     fig.suptitle(query)
        # fig.savefig("plots/%s_cat.pdf" % variable.replace("/", "_"))
        #print(n_data)

        if ratio and draw_data:
            return fig, ax1, ax2, stacked, labels, n_ext
        elif ratio:
            return fig, ax1, ax2, stacked, labels
        elif draw_data:
            return fig, ax1, stacked, labels, n_ext
        else:
            return fig, ax1, stacked, labels

    def _draw_ratio(self, ax, bins, n_tot, n_data, tot_err, data_err, draw_data=True):
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        bin_size = [(bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)]
        if draw_data:
            data_err_low = self._ratio_err(n_data, n_tot, data_err[0], np.zeros(len(data_err[0])))
            data_err_high = self._ratio_err(n_data, n_tot, data_err[1], np.zeros(len(data_err[1])))
            ratio_error = (data_err_low,data_err_high)
            ax.errorbar(bincenters, n_data / n_tot,
                    xerr=bin_size, yerr=ratio_error, fmt="ko")

            ratio_error_mc = self._ratio_err(n_tot, n_tot, tot_err, np.zeros(len(n_tot)))
            ratio_error_mc = np.insert(ratio_error_mc, 0, ratio_error_mc[0])
            bins = np.array(bins)
            ratio_error_mc = np.array(ratio_error_mc)
        self._ratio_vals = n_data / n_tot
        self._ratio_errs = ratio_error
        np.set_printoptions(precision=3)
        #print ('ratio : ',np.array(self._ratio_vals))
        #print ('ratio : ',np.array(self._ratio_errs))
        ax.fill_between(
            bins,
            1.0 - ratio_error_mc,
            ratio_error_mc + 1,
            step="pre",
            color="grey",
            alpha=0.5)

        #ax.set_ylim(0, 2)
        ax.set_ylim(0.5, 1.5)
        #ax.set_ylim(0.8, 1.2)
        ax.set_ylabel("BNB / (MC+EXT)")
        ax.axhline(1, linestyle="--", color="k")

    def sys_err_unisim(self, var_name, query, x_range, n_bins, weightVar, islee=False):

        if x_range == None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0],x_range[1],n_bins+1)

        n_bins = len(bins)-1

        # assume list of knobs is fixed. If not we will get errors
        # knobRPA [up,dn]
        # knobCCMEC [up,dn]
        # knobAxFFCCQE [up,dn]
        # knobVecFFCCQE [up,dn]
        # knobDecayAngMEC [up,dn]
        # knobThetaDelta2Npi [up,dn]
        knob_v = ['knobRPA','knobCCMEC','knobAxFFCCQE','knobVecFFCCQE','knobDecayAngMEC','knobThetaDelta2Npi']
        knob_n = [2,1,1,1,1,1]

        n_tot_v = []
        for u,knob in enumerate(knob_v):
            n_tot_v.append( np.empty([ knob_n[u] ,n_bins]) )
            n_tot_v[-1].fill(0)
        
        n_cv_tot = np.empty(n_bins)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["ext", "data", "data_7e18", "data_1e20"]:#, "lee"
                continue
            if islee and t not in ["lee"]:
                continue
            if islee==False and t in ["lee"]:
                continue

            tree = self.samples[t]

            extra_query = ""
            if t == "mc":
                extra_query = "& " + self.nu_pdg

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            if islee:
                spline_fix_cv  = queried_tree["weightSplineTimesTune"] * queried_tree['leeweight'] * self.weights[t]
                spline_fix_var = queried_tree["weightSpline"] * queried_tree['leeweight'] * self.weights[t]
            else:
                spline_fix_cv  = queried_tree["weightSplineTimesTune"] * self.weights[t]
                spline_fix_var = queried_tree["weightSpline"] * self.weights[t]

            n_cv, bins = np.histogram(variable,bins=bins,weights=spline_fix_cv)
            n_cv_tot += n_cv

            for n,knob in enumerate(knob_v):
                
                weight_up = queried_tree['%sup'%knob].values
                weight_dn = queried_tree['%sdn'%knob].values
                
                weight_up[np.isnan(weight_up)] = 1
                weight_up[weight_up > 100] = 1
                weight_up[weight_up < 0] = 1
                weight_up[weight_up == np.inf] = 1
                
                weight_dn[np.isnan(weight_dn)] = 1
                weight_dn[weight_dn > 100] = 1
                weight_dn[weight_dn < 0] = 1
                weight_dn[weight_dn == np.inf] = 1
                
                n_up, bins = np.histogram(variable, weights=weight_up * spline_fix_var,bins=bins)
                n_dn, bins = np.histogram(variable, weights=weight_dn * spline_fix_var,bins=bins)

                if (knob_n[n] == 2):
                    n_tot_v[n][0] += n_up
                    n_tot_v[n][1] += n_dn
                if (knob_n[n] == 1):
                    n_tot_v[n][0] += n_up
                
        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n,knob in enumerate(knob_v):

            this_cov = np.empty([len(n_cv), len(n_cv)])
            this_cov.fill(0)

            #print ('knob : %s has :'%knob)
            #print ('n_cv : ',n_cv_tot)
            #print ('n_up : ',n_tot_v[n][0])
            #print ('n_dn : ',n_tot_v[n][1])

            if (knob_n[n] == 2):
                for i in range(len(n_cv)):
                    #print ('knob %s has CV: %.0f, VAR UP: %.0f, VAR DN: %.0f entries'%(knob,n_cv_tot[i],n_tot_v[n][0][i],n_tot_v[n][1][i]))
                    for j in range(len(n_cv)):
                        this_cov[i][j] += (n_tot_v[n][0][i] - n_cv_tot[i]) * (n_tot_v[n][0][j] - n_cv_tot[j])
                        this_cov[i][j] += (n_tot_v[n][1][i] - n_cv_tot[i]) * (n_tot_v[n][1][j] - n_cv_tot[j])
                this_cov /= 2.

            if (knob_n[n] == 1):
                for i in range(len(n_cv)):
                    #print ('knob %s has CV: %.0f, VAR: %.0f entries'%(knob,n_cv_tot[i],n_tot_v[n][0][i]))
                    for j in range(len(n_cv)):
                        this_cov[i][j] += (n_tot_v[n][0][i] - n_cv_tot[i]) * (n_tot_v[n][0][j] - n_cv_tot[j])

            cov += this_cov            
            
        return cov

    def xsec_err(self, name, var_name, query, x_range, n_bins, weightVar, acceptance):

        flux_int_v = [117419938, 119294119, 124732356, 142130956, 111027304, 123491329, 128116799, 119006981, 131109155, 119678403, 115105442, 116061418,\
                      124081783, 132845701, 136341953, 112413472, 124370678, 129001686, 128984743, 131330634, 126136084, 124793127, 126580334, 122370971,\
                      129251361, 117928527, 116317847, 127552850, 118683377, 111158087, 116422331, 119799623, 111256181, 124337340, 116303272, 132888883,\
                      107775386, 127174400, 128222532, 110909929, 114969863, 112519702, 117680981, 111173550, 117807553, 125006077, 131216159, 102213961,\
                      127538762, 117277841, 112407550, 118280092, 125378074, 126686937, 124364840, 121954198, 109580696, 131131874, 122790177, 118944024,\
                      121452812, 112182828, 137376653, 140085747, 131693385, 113896621, 116996638, 122522167, 118366681, 123866765, 118902362, 119977912,\
                      134135063, 110308966, 116189312, 120799209, 124508819, 127231203, 120551006, 110201348, 113905095, 115963885, 118575673, 109243818,\
                      107209111, 122380543, 117143023, 122797219, 127540049, 120723447, 120762010, 120025734, 114271425, 120016862, 137782068, 120018537,\
                      124626938, 121113854, 117043736, 123396738]

        flux_int_v = np.array(flux_int_v)
        flux_int_v = flux_int_v.astype(float)
        flux_int_v /= 1e8

        print('CALLING xsec_err with uncertainty for %s'%name)

        Nuniverse = 100

        # remove signal events for background prediction estimation                                                                                                                                                 
        query_bkgd = query + ' and ~(' + acceptance + ")"
        query_sgnl = query + ' and ' + acceptance

        #print ('QUERY : ',query)                                                                                                                      
        #print ('QUERY_SGNL : ',query_sgnl)                           
        #print ('QUERY_BKGD : ',query_bkgd)
                                      
        # observed events in data                                                                                                                                                     
        nBNB = 0
        nEXT = 0

        # total signal events in sample with query                                                                                                              
        nsignaltot = 0.
        # total selected signal events in sample with query                                                  
        nsignalsel = 0.

        for t in self.samples:

            if t in ['lee']:
                continue

            if t in ["data","ext"]:
                tree = self.samples[t]
                data_tree = tree.query(query)
                variable_data = data_tree[var_name]
                weights_data = np.ones(len(variable_data))
                if (t == "ext"):
                    weights_data  = np.ones(len(variable_data)) * self.weights[t]

                n_data, bins = np.histogram(variable_data,range=x_range,bins=1,weights=weights_data)
                if (t == "data"): nBNB = n_data[0]
                if (t == "ext"):  nEXT = n_data[0]

                print ('key is %s and there are %.0f entries'%(t,n_data[0]))

            # if not in data samples 
            if t not in ['data','ext','lee']:

                extra_query = ""
                if (t == "mc"):
                    extra_query = " & " + self.nu_pdg

                tree = self.samples[t]
                signal_tot_tree = tree.query(acceptance + extra_query)
                signal_sel_tree = tree.query(query_sgnl + extra_query)
                variable_tot = signal_tot_tree[var_name]
                variable_sel = signal_sel_tree[var_name]
                syst_weights_tot = signal_tot_tree[name]
                syst_weights_sel = signal_sel_tree[name]
                spline_fix_cv_tot  = signal_tot_tree[weightVar] * self.weights[t]
                spline_fix_cv_sel  = signal_sel_tree[weightVar] * self.weights[t]
                if (name == "weightsGenie"):
                    spline_fix_cv_tot = signal_tot_tree["weightSpline"] * self.weights[t]
                    spline_fix_cv_sel = signal_sel_tree["weightSpline"] * self.weights[t]

                n_cv_tot, bins = np.histogram(variable_tot,range=x_range,bins=1,weights=spline_fix_cv_tot)
                nsignaltot += n_cv_tot[0]
                n_cv_sel, bins = np.histogram(variable_sel,range=x_range,bins=1,weights=spline_fix_cv_sel)
                nsignalsel += n_cv_sel[0]

        eff_cv = float(nsignalsel)/float(nsignaltot)

        print ('%.0f signal events, %.0f selected. efficiency is %.02f'%(nsignalsel,nsignaltot,eff_cv))

        Bkgd_v = np.zeros(Nuniverse)
        Sgnl_v = np.zeros(Nuniverse)
        Totl_v = np.zeros(Nuniverse)

        n_cv_tot_bkgd = 0
        n_cv_tot_sgnl = 0
        n_cv_tot_totl = 0

        for t in self.samples:

            if t in ["data","ext","lee"]:
                continue

            extra_query = ""
            if t == "mc":
                extra_query = "& " + self.nu_pdg

            #print ('key is %s and there are %.0f entries'%(t,n_data[0])) 

            tree = self.samples[t]

            queried_tree_bkgd = tree.query(query_bkgd + extra_query)
            queried_tree_sgnl = tree.query(query_sgnl + extra_query)
            queried_tree_totl = tree.query(acceptance + extra_query)

            print ('key is %s and there are %.0f signal and %.0f background entries'%(t,queried_tree_sgnl.shape[0],queried_tree_bkgd.shape[0]))

            variable_bkgd = queried_tree_bkgd[var_name]
            syst_weights_bkgd = queried_tree_bkgd[name]
            npi0_bkgd = queried_tree_bkgd['npi0']
            variable_sgnl = queried_tree_sgnl[var_name]
            syst_weights_sgnl = queried_tree_sgnl[name]
            variable_totl = queried_tree_totl[var_name]
            syst_weights_totl = queried_tree_totl[name]

            spline_fix_cv_bkgd  = queried_tree_bkgd[weightVar] * self.weights[t]
            spline_fix_var_bkgd = queried_tree_bkgd[weightVar] * self.weights[t]
            spline_fix_cv_sgnl  = queried_tree_sgnl[weightVar] * self.weights[t]
            spline_fix_var_sgnl = queried_tree_sgnl[weightVar] * self.weights[t]
            spline_fix_cv_totl  = queried_tree_totl[weightVar] * self.weights[t]
            spline_fix_var_totl = queried_tree_totl[weightVar] * self.weights[t]

            if (name == "weightsGenie"):
                spline_fix_var_bkgd = queried_tree_bkgd["weightSpline"] * self.weights[t]
                spline_fix_var_sgnl = queried_tree_sgnl["weightSpline"] * self.weights[t]
                spline_fix_var_totl = queried_tree_totl["weightSpline"] * self.weights[t]

            arr_bkgd = np.array(syst_weights_bkgd.values)

            # set weights to 1 for events with pi0s
            '''
            npi0_bkgd = np.array(npi0_bkgd.values)
            print ('there are %i background events'%len(npi0_bkgd))
            for h,npi0 in enumerate(npi0_bkgd):
                if (h < 0):
                    print ('entry %i has %i npi0s'%(h,npi0))
                    print ('weights pre are ',arr_bkgd[h])
                    print ('weights has %i entries'%len(arr_bkgd[h]))
                this_arr_bkgd = arr_bkgd[h]
                nvals = len(this_arr_bkgd)
                if (npi0 > 0):
                    arr_bkgd[h] = np.ones(nvals) * 1000
                if (h < 0):
                    print ('weights post are',arr_bkgd[h])
            '''

            df_bkgd = pd.DataFrame(arr_bkgd.tolist())
            arr_sgnl = np.array(syst_weights_sgnl.values)
            df_sgnl = pd.DataFrame(arr_sgnl.tolist())
            arr_totl = np.array(syst_weights_totl.values)
            df_totl = pd.DataFrame(arr_totl.tolist())

            n_cv_bkgd, bins = np.histogram(variable_bkgd,range=x_range,bins=1,weights=spline_fix_cv_bkgd)
            n_cv_tot_bkgd += n_cv_bkgd
            n_cv_sgnl, bins = np.histogram(variable_sgnl,range=x_range,bins=1,weights=spline_fix_cv_sgnl)
            n_cv_tot_sgnl += n_cv_sgnl
            n_cv_totl, bins = np.histogram(variable_totl,range=x_range,bins=1,weights=spline_fix_cv_totl)
            n_cv_tot_totl += n_cv_totl

            if not df_bkgd.empty:
                for i in range(Nuniverse):
                    # background events
                    weight_bkgd = df_bkgd[i].values / 1000.
                    weight_bkgd[np.isnan(weight_bkgd)] = 1
                    weight_bkgd[weight_bkgd > 100] = 1
                    weight_bkgd[weight_bkgd < 0] = 1
                    weight_bkgd[weight_bkgd == np.inf] = 1
                    n_var_bkgd, bins = np.histogram(variable_bkgd, weights=weight_bkgd*spline_fix_var_bkgd, range=x_range, bins=1)
                    Bkgd_v[i] += n_var_bkgd
                    #if (i < 2):
                    #    print ('variation %i has N bkgd = %.0f for sample %s'%(i,n_var_bkgd,t))
                    #    print ('Bkgd_v[%i] has %.0f entries'%(i,Bkgd_v[i]))

            if not df_sgnl.empty:
                for i in range(Nuniverse):
                    #signal events
                    weight_sgnl = df_sgnl[i].values / 1000.
                    weight_sgnl[np.isnan(weight_sgnl)] = 1
                    weight_sgnl[weight_sgnl > 100] = 1
                    weight_sgnl[weight_sgnl < 0] = 1
                    weight_sgnl[weight_sgnl == np.inf] = 1
                    n_var_sgnl, bins = np.histogram(variable_sgnl, weights=weight_sgnl * spline_fix_var_sgnl, range=x_range, bins=1)
                    Sgnl_v[i] += n_var_sgnl
                    #if (i < 2):
                    #    print ('variation %i has N sgnl = %.0f for sample %s'%(i,n_var_sgnl,t))
                    #    print ('Sgnl_v[%i] has %.0f entries'%(i,Sgnl_v[i]))     

            if not df_totl.empty:
                for i in range(Nuniverse):
                    #total events
                    weight_totl = df_totl[i].values / 1000.
                    weight_totl[np.isnan(weight_totl)] = 1
                    weight_totl[weight_totl > 100] = 1
                    weight_totl[weight_totl < 0] = 1
                    weight_totl[weight_totl == np.inf] = 1
                    n_var_totl, bins = np.histogram(variable_totl, weights=weight_totl * spline_fix_var_totl, range=x_range, bins=1)
                    Totl_v[i] += n_var_totl
                    #if (i < 2):
                    #    print ('variation %i has N totl = %.0f for sample %s'%(i,n_var_totl,t))
                    #    print ('Totl_v[%i] has %.0f entries'%(i,Totl_v[i]))

        # calculate Obs - Bkgd = Signal in each universe                                                                                                                                                            
        Nsignal_v = nBNB - Bkgd_v - nEXT
        Eff_v = Sgnl_v / Totl_v

        for n in range(Nuniverse):
            if (Eff_v[n] < 0.01): Eff_v[n] = eff_cv

        xsec_v = Nsignal_v / eff_cv#Eff_v                                                                                                                                                                           

        if (name == "weightsFlux"):
            xsec_v /= flux_int_v

        print("universe %i has xsec %.02f"%(n,xsec_v[n]))
        print("xsec mean : %.02f. xsec var : %.02f"%(np.mean(xsec_v),np.std(xsec_v)))
        print("bkgd mean : %.02f. bkgd var : %.02f"%(np.mean(Bkgd_v),np.std(Bkgd_v)))
        print("sgnl mean : %.02f. sgnl var : %.02f"%(np.mean(Sgnl_v),np.std(Sgnl_v)))
        print("Eff mean  : %.02f. eff  var : %.02f"%(np.mean(Eff_v),np.std(Eff_v)))
        print("Flux mean : %.02f. Flux var : %.02f"%(np.mean(flux_int_v),np.std(flux_int_v)))

        return            
        
    def sys_err(self, name, var_name, query, x_range, n_bins, weightVar, islee=False, maxUniv = False):

        if x_range == None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0],x_range[1],n_bins+1)

        # how many universes?
        Nuniverse = 100 #len(df)
        if maxUniv:
            if (name == "weightsGenie"): Nuniverse = 500
            if (name == "weightsFlux"):  Nuniverse = 1000
            if (name == "weightsReint"): Nuniverse = 1000
        n_bins = len(bins)-1

        n_tot = np.empty([Nuniverse, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["ext", "data", "data_7e18", "data_1e20"]: #, "lee"#,"dirt","ccnopi","cccpi","nccpi","ncnopi","ncpi0","mc","ccpi0"]:
                continue
            if islee and t not in ["lee"]:
                continue
            if islee==False and t in ["lee"]:
                continue

            # for pi0 fit only
            #if ((t in ["ncpi0","ccpi0"]) and (name == "weightsGenie") ):
            #    continue

            tree = self.samples[t]


            extra_query = ""
            if t == "mc":
                extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            syst_weights = queried_tree[name]
            #print ('N universes is :',len(syst_weights))
            if islee:
                spline_fix_cv  = queried_tree[weightVar] * queried_tree['leeweight'] * self.weights[t]
                spline_fix_var = queried_tree[weightVar] * queried_tree['leeweight'] * self.weights[t]
                if (name == "weightsGenie"):
                    spline_fix_var = queried_tree["weightSpline"] * queried_tree['leeweight'] * self.weights[t]
            else:
                spline_fix_cv  = queried_tree[weightVar] * self.weights[t]
                spline_fix_var = queried_tree[weightVar] * self.weights[t]
                if (name == "weightsGenie"):
                    spline_fix_var = queried_tree["weightSpline"] * self.weights[t]

            s = syst_weights

            '''
            # set weights to 1 for category == 802 events
            category_v = queried_tree["category"].values
            arr = np.array(syst_weights.values)
            for r,category in enumerate(category_v):
                if (category == 802 and name == 'weightsGenie'):
                    arr[r] = np.ones(100) * 1000
            '''
            
            df = pd.DataFrame(s.values.tolist())
            #print (df)
            #print(t,name,np.shape(df))
            #continue

            if var_name[-2:] == "_v":
                #this will break for vector, "_v", entries
                variable = variable.apply(lambda x: x[0])

            n_cv, bins = np.histogram(
                variable,
                bins=bins,
                weights=spline_fix_cv)
            n_cv_tot += n_cv

            if not df.empty:
                for i in range(Nuniverse):
                    weight = df[i].values / 1000.
                    weight[np.isnan(weight)] = 1
                    weight[weight > 100] = 1
                    weight[weight < 0] = 1
                    weight[weight == np.inf] = 1

                    n, bins = np.histogram(
                        variable, weights=weight*spline_fix_var,bins=bins)
                    n_tot[i] += n

        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n in n_tot:
            for i in range(len(n_cv)):
                for j in range(len(n_cv)):
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse

        return cov
    
    #function for calculating cross covariance 

    def sys_err_cross_cov(self, name, var_name, queries, x_range, n_bins, weightVar, islee=False, maxUniv=False):
        if x_range is None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0], x_range[1], n_bins + 1)

        # how many universes?
        Nuniverse = 100  # len(df)
        if maxUniv:
            if name == "weightsGenie": 
                Nuniverse = 500
            if name == "weightsFlux":
                Nuniverse = 1000
            if name == "weightsReint":
                Nuniverse = 1000
        n_bins = len(bins) - 1

        n_tot = [np.empty([Nuniverse, n_bins]) for _ in queries]
        n_cv_tot = [np.empty(n_bins) for _ in queries]
        for n in n_tot:
            n.fill(0)
        for n in n_cv_tot:
            n.fill(0)

        # Iterate over each query
        for query_idx, query in enumerate(queries):
            for t in self.samples:
                if t in ["ext", "data", "data_7e18", "data_1e20"]: #, "lee"#,"dirt","ccnopi","cccpi","nccpi","ncnopi","ncpi0","mc","ccpi0"]:
                    continue
                if islee and t not in ["lee"]:
                    continue
                if islee==False and t in ["lee"]:
                    continue

                tree = self.samples[t]


                extra_query = ""
                if t == "mc":
                    extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"

                # Process each query separately
                queried_tree = tree.query(query)  
                # Rest of logic here
                variable = queried_tree[var_name]
                syst_weights = queried_tree[name]
                #print ('N universes is :',len(syst_weights))
                #print ('N universes is :',len(syst_weights))
                if islee:
                    spline_fix_cv  = queried_tree[weightVar] * queried_tree['leeweight'] * self.weights[t]
                    spline_fix_var = queried_tree[weightVar] * queried_tree['leeweight'] * self.weights[t]
                    if (name == "weightsGenie"):
                        spline_fix_var = queried_tree["weightSpline"] * queried_tree['leeweight'] * self.weights[t]
                else:
                    spline_fix_cv  = queried_tree[weightVar] * self.weights[t]
                    spline_fix_var = queried_tree[weightVar] * self.weights[t]
                    if (name == "weightsGenie"):
                        spline_fix_var = queried_tree["weightSpline"] * self.weights[t]

                s = syst_weights
                df = pd.DataFrame(s.values.tolist())
                #print (df)
                #print(t,name,np.shape(df))
                #continue

                if var_name[-2:] == "_v":
                    #this will break for vector, "_v", entries
                    variable = variable.apply(lambda x: x[0])
                # Calculate histograms for each query
                n_cv, _ = np.histogram(
                    variable,
                    bins=bins,
                    weights=spline_fix_cv)
                n_cv_tot[query_idx] += n_cv

                # Loop for calculating n_tot for each query
                if not df.empty:
                    for i in range(Nuniverse):
                        #existing logic for weights and histogram calculation
                        weight = df[i].values / 1000.
                        weight[np.isnan(weight)] = 1
                        weight[weight > 100] = 1
                        weight[weight < 0] = 1
                        weight[weight == np.inf] = 1

                        n, _ = np.histogram(
                            variable, weights=weight * spline_fix_var, bins=bins)
                        n_tot[query_idx][i] += n


        # Calculate covariance for each query and cross-covariance
        cov = np.zeros((len(queries), len(queries), n_bins, n_bins))

        for i, n_tot_i in enumerate(n_tot):
            for j, n_tot_j in enumerate(n_tot):
                for uni in range(Nuniverse):
                    diff_i = n_tot_i[uni] - n_cv_tot[i]
                    diff_j = n_tot_j[uni] - n_cv_tot[j] if i != j else diff_i
                    cov[i, j] += np.outer(diff_i, diff_j)
                cov[i, j] /= Nuniverse

        return cov

    
    

    def ResponseMatrix(self,sample,acceptance,fullsel,vart,varr,bin_edges,wlab,potw,univ=-1,wvar=''):
        #get number of signal events at true level before selection
        truevals = sample.query(acceptance)[vart]
        tweights = sample.query(acceptance)[wlab]*potw
        if univ>=0:
            vweights = sample.query(acceptance)[wvar]
            if (np.stack(vweights).ndim>1):
                #multisim, pick specific universe
                vweights = np.stack(vweights)[:,univ]/1000.
            vweights[np.isnan(vweights)] = 1
            vweights[vweights > 100] = 1
            vweights[vweights < 0] = 1
            vweights[vweights == np.inf] = 1
            tweights = tweights*vweights
        n, bins = np.histogram(truevals,weights=tweights,bins=bin_edges)
        #get number of signal events at reco and true level after selection
        x = sample.query(acceptance+' and '+fullsel)[vart]
        y = sample.query(acceptance+' and '+fullsel)[varr]
        w = sample.query(acceptance+' and '+fullsel)[wlab]*potw
        if univ>=0:
            vw = sample.query(acceptance+' and '+fullsel)[wvar]
            if (np.stack(vw).ndim>1):
                #multisim, pick specific universe
                vw = np.stack(vw)[:,univ]/1000.
            vw[np.isnan(vw)] = 1
            vw[vw > 100] = 1
            vw[vw < 0] = 1
            vw[vw == np.inf] = 1
            w = w*vw
        H, xb, yb = np.histogram2d(x,y,weights=w,bins=[bin_edges,bin_edges])
        #get response matrix
        rm = np.transpose(H)/n
        return rm, xb, yb

    def sys_err_unisim_with_resp_func(self, signame, var_name, true_var_name, query, acceptance, x_range, n_bins):
        #this is not done for lee

        if x_range == None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0],x_range[1],n_bins+1)

        n_bins = len(bins)-1

        weightVarCV = "weightSplineTimesTune"
        weightVarVar = "weightSpline"

        # assume list of knobs is fixed. If not we will get errors
        # knobRPA [up,dn]
        # knobCCMEC [up,dn]
        # knobAxFFCCQE [up,dn]
        # knobVecFFCCQE [up,dn]
        # knobDecayAngMEC [up,dn]
        # knobThetaDelta2Npi [up,dn]
        knob_v = ['knobRPA','knobCCMEC','knobAxFFCCQE','knobVecFFCCQE','knobDecayAngMEC','knobThetaDelta2Npi']
        knob_n = [2,1,1,1,1,1]

        n_tot_v = []
        for u,knob in enumerate(knob_v):
            n_tot_v.append( np.empty([ knob_n[u] ,n_bins]) )
            n_tot_v[-1].fill(0)

        n_cv_tot = np.empty(n_bins)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["ext", "data", "data_7e18", "data_1e20","lee"]:
                continue

            tree = self.samples[t]

            extra_query = ""
            if t == "mc":
                extra_query = "& " + self.nu_pdg
            if t == signame:
                extra_query = "& ~(" + acceptance +")"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            spline_fix_cv  = queried_tree[weightVarCV] * self.weights[t]
            spline_fix_var = queried_tree[weightVarVar] * self.weights[t]

            n_cv, bins = np.histogram(variable,bins=bins,weights=spline_fix_cv)
            n_cv_tot += n_cv

            for n,knob in enumerate(knob_v):

                weight_up = queried_tree['%sup'%knob].values
                weight_up[np.isnan(weight_up)] = 1
                weight_up[weight_up > 100] = 1
                weight_up[weight_up < 0] = 1
                weight_up[weight_up == np.inf] = 1
                n_up, bins = np.histogram(variable, weights=weight_up * spline_fix_var,bins=bins)
                n_tot_v[n][0] += n_up

                if (knob_n[n] == 2):
                    weight_dn = queried_tree['%sdn'%knob].values
                    weight_dn[np.isnan(weight_dn)] = 1
                    weight_dn[weight_dn > 100] = 1
                    weight_dn[weight_dn < 0] = 1
                    weight_dn[weight_dn == np.inf] = 1
                    n_dn, bins = np.histogram(variable, weights=weight_dn * spline_fix_var,bins=bins)
                    n_tot_v[n][1] += n_dn

        # do stuff for signal sample
        tree = self.samples[signame]

        extra_query = ""
        if signame == "mc":
            extra_query += "& " + self.nu_pdg

        queried_tree = tree.query(query+"& (" + acceptance +")"+extra_query)
        variable = queried_tree[var_name]
        #print ('N universes is :',len(syst_weights))
        spline_fix_cv  = queried_tree[weightVarCV] * self.weights[signame]

        n_cv, bins = np.histogram(variable,bins=bins,weights=spline_fix_cv)
        n_cv_tot += n_cv

        true_variable = tree.query(acceptance)[true_var_name]
        spline_fix_cv  = tree.query(acceptance)[weightVarCV] * self.weights[signame]
        t_cv, bins = np.histogram(true_variable,bins=bins,weights=spline_fix_cv)

        for n,knob in enumerate(knob_v):

                rmv_up, xb, yb = self.ResponseMatrix(tree,acceptance,query,true_var_name,var_name,\
                                                     bins,weightVarVar,self.weights[signame],0,'%sup'%knob)
                rp_up = rmv_up.dot(t_cv)
                n_tot_v[n][0] += rp_up
                if (knob_n[n] == 2):
                    rmv_dn, xb, yb = self.ResponseMatrix(tree,acceptance,query,true_var_name,var_name,\
                                                         bins,weightVarVar,self.weights[signame],0,'%sdn'%knob)
                    rp_dn = rmv_dn.dot(t_cv)
                    n_tot_v[n][1] += rp_dn

        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n,knob in enumerate(knob_v):

            this_cov = np.empty([len(n_cv), len(n_cv)])
            this_cov.fill(0)

            #print ('knob : %s has :'%knob)
            #print ('n_cv : ',n_cv_tot)
            #print ('n_up : ',n_tot_v[n][0])
            #print ('n_dn : ',n_tot_v[n][1])

            if (knob_n[n] == 2):
                for i in range(len(n_cv)):
                    #print ('knob %s has CV: %.0f, VAR UP: %.0f, VAR DN: %.0f entries'%(knob,n_cv_tot[i],n_tot_v[n][0][i],n_tot_v[n][1][i]))
                    for j in range(len(n_cv)):
                        this_cov[i][j] += (n_tot_v[n][0][i] - n_cv_tot[i]) * (n_tot_v[n][0][j] - n_cv_tot[j])
                        this_cov[i][j] += (n_tot_v[n][1][i] - n_cv_tot[i]) * (n_tot_v[n][1][j] - n_cv_tot[j])
                this_cov /= 2.

            if (knob_n[n] == 1):
                for i in range(len(n_cv)):
                    #print ('knob %s has CV: %.0f, VAR: %.0f entries'%(knob,n_cv_tot[i],n_tot_v[n][0][i]))
                    for j in range(len(n_cv)):
                        this_cov[i][j] += (n_tot_v[n][0][i] - n_cv_tot[i]) * (n_tot_v[n][0][j] - n_cv_tot[j])

            cov += this_cov

        return cov
    
    def sys_err_unisim_with_resp_func_x_cov(self, concat_sample, var_name, true_var_name, queries, acceptance_, x_range, n_bins):
        if x_range is None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0], x_range[1], n_bins + 1)

        n_bins = len(bins) - 1
        weightVarCV = "weightSplineTimesTune"
        weightVarVar = "weightSpline"

        # Knob configuration
        knob_v = ['knobRPA', 'knobCCMEC', 'knobAxFFCCQE', 'knobVecFFCCQE', 'knobDecayAngMEC', 'knobThetaDelta2Npi']
        knob_n = [2, 1, 1, 1, 1, 1]

        # Initialize storage for histograms with knobs as the outer dimension
        n_cv_tot_knobs = [[np.empty(n_bins) for _ in queries] for _ in knob_v]
        #n_tot_v_knobs = [[[np.empty(n_bins) for _ in queries] for _ in range(knob_n[u])] for u in knob_v]
        n_tot_v_knobs = [[[np.empty(n_bins) for _ in queries] for _ in range(knob_n[knob_idx])] for knob_idx in range(len(knob_v))]

        for n_cv_tot_queries in n_cv_tot_knobs:
            for n_cv_tot in n_cv_tot_queries:
                n_cv_tot.fill(0)
        
       
        for n_tot_v_knob in n_tot_v_knobs:
            for n_tot_v_queries in n_tot_v_knob:
                for n_tot_v in n_tot_v_queries:
                    n_tot_v.fill(0)

        # Loop over knobs
        for n, knob in enumerate(knob_v):
            # Process each query within the knob loop
            for query_idx, query in enumerate(queries):
                if (query_idx == 0) :
                    acceptance = acceptance_ + " and ccnc == 0 "
                else: 
                    acceptance = acceptance_ + " and ccnc == 1"
                # Initialize a counter for the number of processed samples
                processed_samples = 0
                for t in self.samples:
                    # Sample-specific logic remains unchanged...
                    if t in ["ext", "data", "data_7e18", "data_1e20","lee"]:
                        continue
                    # Increment the counter
                    processed_samples += 1
                    extra_query = ""
                    if t == "mc":
                        extra_query = "& " + self.nu_pdg
                        
                    if processed_samples <= 7:
                        tree = self.samples[t]
                        # Process knob-specific histograms
                        queried_tree = tree.query(query + extra_query)
                        variable = queried_tree[var_name]
                        spline_fix_cv  = queried_tree[weightVarCV] * self.weights[t]
                        spline_fix_var = queried_tree[weightVarVar] * self.weights[t]

                        n_cv, bins = np.histogram(variable, bins=bins, weights=spline_fix_cv)
                        n_cv_tot_knobs[n][query_idx] += n_cv

                        # Process each knob variation
                        for variation in range(knob_n[n]):
                            weight_var = queried_tree['%s%s' % (knob, 'up' if variation == 0 else 'dn')].values
                            # Apply necessary weight corrections...
                            n_var, bins = np.histogram(variable, weights=weight_var * spline_fix_var, bins=bins)
                            n_tot_v_knobs[n][variation][query_idx] += n_var
                    
                    else:
                        extra_query = "& ~(" + acceptance +")"
                        #print("here")
                        cc_name = "ccpi0"
                        nc_name = "ncpi0"
                        cc = self.samples[cc_name]
                        nc = self.samples[nc_name]
                        potw_cc = self.weights[cc_name]
                        #print(potw_cc)
                        potw_nc = self.weights[nc_name]
                        #print(potw_nc)
                        cc["pot"] = potw_cc
                        nc["pot"] = potw_nc
                        nue = pd.concat([cc, nc])
                        tree = nue 
                        # Process knob-specific histograms
                        queried_tree = tree.query(query + extra_query)
                        variable = queried_tree[var_name]
                        spline_fix_cv  = queried_tree[weightVarCV] * queried_tree['pot'] #self.weights[t]
                        spline_fix_var = queried_tree[weightVarVar] * queried_tree['pot'] #self.weights[t]

                        n_cv, bins = np.histogram(variable, bins=bins, weights=spline_fix_cv)
                        n_cv_tot_knobs[n][query_idx] += n_cv

                        # Process each knob variation
                        for variation in range(knob_n[n]):
                            weight_var = queried_tree['%s%s' % (knob, 'up' if variation == 0 else 'dn')].values
                            # Apply necessary weight corrections...
                            n_var, bins = np.histogram(variable, weights=weight_var * spline_fix_var, bins=bins)
                            n_tot_v_knobs[n][variation][query_idx] += n_var
                    
           
                # do stuff for signal sample
                tree = concat_sample #self.samples[signame]

                extra_query = ""
                queried_tree = tree.query(query+"& (" + acceptance +")"+extra_query)
                #queried_tree = tree.query(query + extra_query)
                variable = queried_tree[var_name]
                spline_fix_cv  = queried_tree[weightVarCV] * queried_tree["pot"] #self.weights[signame]

                n_cv, bins = np.histogram(variable, bins=bins, weights=spline_fix_cv)
                n_cv_tot_knobs[n][query_idx] += n_cv

                true_variable = tree.query(acceptance)[true_var_name]
                spline_fix_cv  = tree.query(acceptance)[weightVarCV] * tree.query(acceptance)["pot"] #self.weights[signame]
                t_cv, bins = np.histogram(true_variable, bins=bins, weights=spline_fix_cv)
                weights = tree.query(acceptance)["pot"][0]
                # Process the response matrix
                for variation in range(knob_n[n]):
                    variation_suffix = 'up' if variation == 0 else 'dn'
                    rmv, xb, yb = self.ResponseMatrix(tree, acceptance, query, true_var_name, var_name,
                                                      bins, weightVarVar, weights, 0, '%s%s' % (knob, variation_suffix))
                    rp = rmv.dot(t_cv)
                    n_tot_v_knobs[n][variation][query_idx] += rp


        # Calculate covariance and cross-covariance
        cov_matrices = [[[np.zeros((n_bins, n_bins)) for _ in queries] for _ in queries] for _ in knob_v]

        for n, knob in enumerate(knob_v):
            for i in range(len(queries)):
                for j in range(len(queries)):
                    for bin_i in range(n_bins):
                        for bin_j in range(n_bins):
                            for variation in range(knob_n[n]):
                                diff_i = n_tot_v_knobs[n][variation][i][bin_i] - n_cv_tot_knobs[n][i][bin_i]
                                diff_j = n_tot_v_knobs[n][variation][j][bin_j] - n_cv_tot_knobs[n][j][bin_j]
                                cov_matrices[n][i][j][bin_i][bin_j] += diff_i * diff_j
                            if knob_n[n] == 2:
                                cov_matrices[n][i][j][bin_i][bin_j] /= 2.  # Averaging for 'up' and 'down' variations

        return cov_matrices
    
    

    def sys_err_with_resp_func(self, wname, signame, var_name, true_var_name, query, acceptance, x_range, n_bins, weightVar, maxUniv = False):
        #this is not done for lee

        if x_range == None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0],x_range[1],n_bins+1)

        weightVarCV = weightVar
        #need special case since genie tune changed at some point
        weightVarVar = weightVar
        if (wname == "weightsGenie"): weightVarVar = "weightSpline"

        # how many universes?
        Nuniverse = 100 #len(df)
        if maxUniv:
            if (wname == "weightsGenie"): Nuniverse = 500
            if (wname == "weightsFlux"):  Nuniverse = 1000
            if (wname == "weightsReint"): Nuniverse = 1000
        n_bins = len(bins)-1

        n_tot = np.empty([Nuniverse, n_bins])
        n_cv_tot = np.empty(n_bins)
        n_tot.fill(0)
        n_cv_tot.fill(0)

        for t in self.samples:
            if t in ["ext", "data", "data_7e18", "data_1e20","lee"]:
                continue

            tree = self.samples[t]

            extra_query = ""
            if t == "mc":
                extra_query = "& " + self.nu_pdg # "& ~(abs(nu_pdg) == 12 & ccnc == 0) & ~(npi0 == 1 & category != 5)"
            if t == signame:
                extra_query = "& ~(" + acceptance +")"

            queried_tree = tree.query(query+extra_query)
            variable = queried_tree[var_name]
            syst_weights = queried_tree[wname]
            #print ('N universes is :',len(syst_weights))
            spline_fix_cv  = queried_tree[weightVarCV] * self.weights[t]
            spline_fix_var = queried_tree[weightVarVar] * self.weights[t]

            s = syst_weights

            df = pd.DataFrame(s.values.tolist())
            #print (df)
            #print(t,wname,np.shape(df))
            #continue

            n_cv, bins = np.histogram(
                variable,
                bins=bins,
                weights=spline_fix_cv)
            n_cv_tot += n_cv

            if not df.empty:
                for i in range(Nuniverse):
                    weight = df[i].values / 1000.
                    weight[np.isnan(weight)] = 1
                    weight[weight > 100] = 1
                    weight[weight < 0] = 1
                    weight[weight == np.inf] = 1

                    n, bins = np.histogram(
                        variable, weights=weight*spline_fix_var,bins=bins)
                    n_tot[i] += n

        # do stuff for signal sample
        tree = self.samples[signame]

        extra_query = ""
        if signame == "mc":
            extra_query += "& " + self.nu_pdg

        queried_tree = tree.query(query+"& (" + acceptance +")"+extra_query)
        variable = queried_tree[var_name]
        syst_weights = queried_tree[wname]
        #print ('N universes is :',len(syst_weights))
        spline_fix_cv  = queried_tree[weightVarCV] * self.weights[signame]

        s = syst_weights

        df = pd.DataFrame(s.values.tolist())
        #print (df)
        #print(t,wname,np.shape(df))
        #continue

        #print(bins)
        #print(variable)
        #print(spline_fix_cv)
        n_cv, bins = np.histogram(
            variable,
            bins=bins,
            weights=spline_fix_cv)
        n_cv_tot += n_cv
        #print('n_cv',n_cv)

        true_variable = tree.query(acceptance)[true_var_name]
        spline_fix_cv  = tree.query(acceptance)[weightVarCV] * self.weights[signame]
        t_cv, bins = np.histogram(
            true_variable,
            bins=bins,
            weights=spline_fix_cv)
        #print('t_cv',t_cv)

        if not df.empty:
            for i in range(Nuniverse):
                rmv, xb, yb = self.ResponseMatrix(tree,acceptance,query,true_var_name,var_name,\
                                                  bins,weightVarVar,self.weights[signame],i,wname)
                #print(rmv)
                rp = rmv.dot(t_cv)
                #print("variation: ",rp)
                n_tot[i] += rp

        # now compute the covariance
        cov = np.empty([len(n_cv), len(n_cv)])
        cov.fill(0)

        for n in n_tot:
            for i in range(len(n_cv)):
                for j in range(len(n_cv)):
                    cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])

        cov /= Nuniverse

        return cov

    def get_SBNFit_cov_matrix(self,COVMATRIX,NBINS):

        covmatrix = np.zeros([NBINS,NBINS])
        
        if (os.path.isfile("COV/"+COVMATRIX) == False):
            print ('ERROR : file-path for covariance matrix not valid!')
            return covmatrix

        covmatrixfile = open("COV/"+COVMATRIX,"r")

        NLINES = len(covmatrixfile.readlines())

        print ('file has %i lines and histo has %i bins'%(NLINES,NBINS))

        if NLINES != NBINS:
            print ('ERROR : number of lines in text-file does not match number of bins!')
            return covmatrix

        LINECTR = 0

        covmatrixfile.seek(0,0)
        for line in covmatrixfile:

            words = line.split(",")

            WORDCTR = 0

            if len(words) != NBINS:
                print ('ERROR : number of words in line does not match number of bins!')
                break
                
            for word in words:

                val = float(word)

                covmatrix[LINECTR][WORDCTR] = val

                WORDCTR += 1

            LINECTR += 1

        return covmatrix
    
