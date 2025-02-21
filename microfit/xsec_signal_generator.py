import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from microfit.histogram import HistogramGenerator
from microfit.histogram import Histogram, MultiChannelHistogram
from microfit.statistics import covariance

class XsecCovarHistGenerator(HistogramGenerator):

    def __init__(self, *args, true_var_name=None, signal_query=None, uncut_signal_df=None, **kwargs):
        self.true_var_name = true_var_name
        assert signal_query is not None
        self.signal_query = signal_query
        assert uncut_signal_df is not None
        self.uncut_signal_df = uncut_signal_df
        super().__init__(*args, **kwargs)

    def calculate_unisim_uncertainties(
        self,
        central_value_hist: Optional[Histogram] = None,
        extra_query: Optional[str] = None,
        return_histograms: bool = False,
        skip_covariance: bool = False,
    ):
        # First, apply the usual calculation to the background only
        background_query = f"~({self.signal_query})"
        background_hist_dict = None
        background_covariance = None
        background_extra_query = background_query if extra_query is None else background_query + " & " + extra_query
        background_covariance, background_vars_dict = super().calculate_unisim_uncertainties(
            central_value_hist = central_value_hist,
            extra_query = background_extra_query,
            return_histograms = True,
        )

         # Calculate the signal variation hists
        
        knob_v = ['knobRPA','knobCCMEC','knobAxFFCCQE','knobVecFFCCQE','knobDecayAngMEC','knobThetaDelta2Npi']
        knob_n = [2,1,1,1,1,1]
        n_tot_v = [] # this will store the variation hists
        for u, knob in enumerate(knob_v):
            n_tot_v.append(np.zeros([ knob_n[u] , self.binning.n_bins]))

        if self.true_var_name is None:
            label = self.binning.variable.lstrip("Reco")
            self.true_var_name = "True" + label
        
        if extra_query is not None:
            sel_df = self.dataframe.query(extra_query, engine="python")
        else:
            sel_df = self.dataframe

        raw_df = self.uncut_signal_df
        queried_df = sel_df.query(f"{self.signal_query}", engine="python")
        base_weight_column = "weights_no_tune"

        # Calculate the true signal CV that stays constant in all the universes
        true_sig = raw_df.query(self.signal_query, engine="python")
        true_variable = true_sig[self.true_var_name]
        true_var_weightsCV  = true_sig["weights"]
        t_cv, bins = np.histogram(true_variable, bins=self.binning.bin_edges, weights=true_var_weightsCV)
        
        sig_vars_dict = dict()
        for n,knob in enumerate(knob_v):
            rmv_up, xb, yb = self.ResponseMatrix(raw_df, sel_df, self.signal_query, self.true_var_name, self.binning.variable, self.binning.bin_edges, base_weight_column, 0, f"{knob}up")
            rp_up = rmv_up.dot(t_cv)
            n_tot_v[n][0] += rp_up

            if (knob_n[n] == 2):
                rmv_dn, xb, yb = self.ResponseMatrix(raw_df, sel_df, self.signal_query, self.true_var_name, self.binning.variable, self.binning.bin_edges, base_weight_column, 0, f"{knob}dn")
                rp_dn = rmv_dn.dot(t_cv)
                n_tot_v[n][1] += rp_dn
                
            sig_vars_dict[knob] = n_tot_v[n]

        # Add the signal and background variation hists and calculate cov matrix for each knob
        if central_value_hist is None:
            central_value_hist = self._histogram_multi_channel(sel_df)

        total_cov = np.zeros((self.binning.n_bins, self.binning.n_bins))
        total_vars_dict = dict()
        for n,knob in enumerate(knob_v):
            total_vars_dict[knob] = sig_vars_dict[knob] + background_vars_dict[knob]
            # If we get to this point without having either calculated a central value hist
            # or taken one from the cache, something is wrong
            assert central_value_hist is not None
            if skip_covariance:
                continue
            # calculate the covariance matrix from the histograms
            cov = covariance(
                total_vars_dict[knob],
                central_value_hist.bin_counts,
                allow_approximation=True,
                debug_name=knob,
                tolerance=1e-8,
            )
            self.logger.debug(
                f"Bin-wise error contribution for knob {knob}: {np.sqrt(np.diag(cov))}"
            )
            # add it to the total covariance matrix
            total_cov += cov
        
        if return_histograms:
            return total_cov, total_vars_dict
        return total_cov


    def calculate_multisim_uncertainties(
        self,
        multisim_weight_column: str,
        weight_rescale: float = 1 / 1000,
        weight_column: Optional[str] = None,
        central_value_hist: Optional[Union[Histogram, MultiChannelHistogram]] = None,
        extra_query: Optional[str] = None,
        return_histograms: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        if multisim_weight_column in ["weightsFlux", "weightsReint"]:
            return super().calculate_multisim_uncertainties(
                multisim_weight_column = multisim_weight_column,
                weight_rescale = weight_rescale,
                weight_column = weight_column,
                central_value_hist = central_value_hist,
                extra_query = extra_query,
                return_histograms = return_histograms, # type: ignore
            )
        assert multisim_weight_column == "weightsGenie"
        if weight_column is None:
            weight_column = (
                "weights_no_tune" if multisim_weight_column == "weightsGenie" else "weights"
            )

        # First, apply the usual calculation to the background only
        background_query = f"~({self.signal_query})"
        background_histograms = None
        background_covariance = None
        background_extra_query = background_query if extra_query is None else background_query + " & " + extra_query
        background_covariance, background_histograms = super().calculate_multisim_uncertainties(
            multisim_weight_column=multisim_weight_column,
            weight_rescale=weight_rescale,
            weight_column=weight_column,
            central_value_hist=central_value_hist,
            extra_query=background_extra_query,
            return_histograms=True
        )

         # Calculate the signal variation hists

        if self.true_var_name is None:
            label = self.binning.variable.lstrip("Reco")
            self.true_var_name = "True" + label
        
        if extra_query is not None:
            sel_df = self.dataframe.query(extra_query, engine="python")
        else:
            sel_df = self.dataframe

        raw_df = self.uncut_signal_df
        Nuniverse = 100 #len(raw_df[multisim_weight_column][0]) # should be 100 in the 2024 PELEE n-tuples
        n_tot = np.zeros([Nuniverse, self.binning.n_bins]) # this will store the variation hists (rows)
        queried_df = sel_df.query(f"{self.signal_query}", engine="python")
        syst_weights = queried_df[multisim_weight_column]

        # Calculate the signal variation hists and store them

        weights_df = pd.DataFrame(syst_weights.values.tolist()) # df of multisim weights that have been flattened horizontally - i.e. Nuniv (100) columns per row

        # Calculate the true signal CV that stays constant in all the universes
        true_sig = raw_df.query(self.signal_query, engine="python")
        true_variable = true_sig[self.true_var_name]
        true_var_weightsCV  = true_sig["weights"]
        t_cv, bins = np.histogram(true_variable, bins=self.binning.bin_edges, weights=true_var_weightsCV)

        if not weights_df.empty:
            for i in range(Nuniverse):
                rmv, xb, yb = self.ResponseMatrix(raw_df, sel_df, self.signal_query, self.true_var_name, self.binning.variable, self.binning.bin_edges, weight_column, i, multisim_weight_column)
                #print(rmv)
                rp = rmv.dot(t_cv)
                #print("variation: ",rp)
                n_tot[i] += rp

        # Add the signal and background variation hists
        universe_hists = background_histograms + n_tot

        # Now finally compute the covariance
        if central_value_hist is None:
            central_value_hist = self._histogram_multi_channel(sel_df)
        # calculate the covariance matrix from the histograms
        final_covariance = covariance(
            universe_hists,
            central_value_hist.bin_counts,
            allow_approximation=True,
            tolerance=1e-8,
        )

        if return_histograms:
             return final_covariance, universe_hists
        else:
             return final_covariance
        
    def ResponseMatrix(self, raw_df, sel_df, truth_def, true_var_name, var_name, bin_edges, base_weight_Var, univ=-1, wname=""):

        """Calculate the response matrix.

            Parameters
            ----------
            raw_df : pandas dataframe
                The dataframe containing the signal events with no selection (truth and/or reco) applie to it.
            sel_df : pandas dataframe
                The dataframe with the full selection applied.
            truth_def : str
                A set of cuts specifying the signal definition using truth-level variables.
            var_name : str
                The name of the reconstructed variable you are calculating the covariance for.
            true_var_name : str
                The name of the truth-level variable corresponding to var_name. 
            bin_edges : np.ndarray
                Array of bin edges.
            base_weight_Var : str
                Name of the column containing the baseline weights of the events. 
            univ : int
                The universe index.
            wname : str
                Name of multisim weight column. 

            Returns
            -------
            rm : array_like
                Response matrix of the bin counts.
            xb : array_like
                Horizontal bin edges.
            yb : array_like
                Vertical bin edges. 
        """ # type: ignore
    
        # Get number of signal events at true level before selection
        true_sig = raw_df.query(truth_def, engine="python")
        truevals = true_sig[true_var_name]
        tweights = true_sig[base_weight_Var]
    
        if univ>=0:
            vweights = true_sig[wname]
            if (np.stack(vweights).ndim>1):
                #multisim, pick specific universe
                vweights = np.stack(vweights)[:,univ]/1000.
            vweights[np.isnan(vweights)] = 1
            vweights[vweights > 100] = 1
            vweights[vweights < 0] = 1
            vweights[vweights == np.inf] = 1
            tweights = tweights * vweights
        
        n, bins = np.histogram(truevals, weights=tweights, bins=bin_edges)
        
        # Get number of signal events at reco and true level after selection
        sel_true_sig = sel_df.query(truth_def, engine="python")
        x = sel_true_sig[true_var_name]
        y = sel_true_sig[var_name]
        w = sel_true_sig[base_weight_Var]
        if univ>=0:
            vw = sel_true_sig[wname]
            if (np.stack(vw).ndim>1):
                #multisim, pick specific universe
                vw = np.stack(vw)[:,univ]/1000.
            vw[np.isnan(vw)] = 1
            vw[vw > 100] = 1
            vw[vw < 0] = 1
            vw[vw == np.inf] = 1
            w = w * vw
        H, xb, yb = np.histogram2d(x, y, weights=w, bins=[bin_edges, bin_edges])
        # Get response matrix
        rm = np.transpose(H)/n
        return rm, xb, yb
