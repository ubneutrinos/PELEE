import numpy as np
import pandas as pd

# Functions for calculating the GENIE multisim and unisim covariance matrices 
# using the response matrix method for xsec analyses to be used with Microfit
# (based on the functions in the original PELEE framework by G Cerati and S Berkman)
# Author: M Moudgalya

################################################################################

def ResponseMatrix(df, truth_def, fullsel, true_var_name, var_name, bin_edges, base_weight_Var, univ=-1, wname=""):

    """Calculate the response matrix.

        Parameters
        ----------
        df : pandas dataframe
            The dataframe of the n-tuple being used to calculate the response matrix.
        truth_def : str
            A set of cuts specifying the signal definition using truth-level variables.
        fullsel : str
            The full set of preselection and selection cuts to be applied.
        var_name : str
            The name of the reconstructed variable you are calculating the covariance for.
        true_var_name : str
            The name of the truth-level variable corresponding to var_name. 
        bin_edges : np.ndarray
            Array of bin edges.
        base_weight_Var : str
            Name of the column containing the baseline weights of the events. 

        Returns
        -------
        rm : array_like
            Response matrix of the bin counts.
        xb : array_like
            Horizontal bin edges.
        yb : array_like
            Vertical bin edges.
    """
    
    # Get number of signal events at true level before selection
    true_sig = df.query(truth_def, engine="python")
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
    sel_true_sig = true_sig.query(fullsel, engine="python")
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

################################################################################

def multisim_err_with_resp_func(rundata, signame, var_name, true_var_name, preselection, selection, truth_def, n_bins=None, x_range=None, bin_edges=None, weightCV="weights", base_weight_Var="weights_no_tune", wname="weightsGenie"):

        """Calculate multisim uncertainties using the response matrix method.

        Each of the given multisim weight columns is expected to contain a list of weights
        for every row that correspond to the weights of the fluctuated "universes". The
        histogram is regenerated for every universe and the covariance matrix is calculated
        from the resulting histograms.

        This particular method, whereby the response matrix is used, is required for xsec analyses.
        Here, the signal and background are treated differently. The covariance for the background
        is calculated as described above. But the covariance for the signal is calculated using the
        response matrix; the variation for a particular universe is calculated by multiplying the
        response matrix for that universe (S^{univ}_{sel} / S^{univ}_{total}) by the total true
        signal CV (central value) prediction.
        
        This function has written to be used with microfit.

        Parameters
        ----------
        rundata : dict
            Dictionary of dataframes for each sample. This is the output from the load_runs()
            function in data_loading.py.
        signame : str
            The name of the MC sample that contains your signal events. This must refer to one of the keys in rundata.
            For my analysis, this is "nue".
        var_name : str
            The name of the reconstructed variable you are calculating the covariance for.
        true_var_name : str
            The name of the truth-level variable corresponding to var_name. This is required, unless var_name and true_var_name
            share the same stem but with either "Reco" or "True" at the front of the string 
            (e.g. var_name = "RecoDeltaPT" and true_var_name = "TrueDeltaPT"), in which case true_var_name is optional.
        preselection : str
            The name of one of the sets of preselection cuts in selections.py in microfit.
        selection : str
            The name of one of the sets of selection cuts in selections.py in microfit.
        truth_def : str
            A set of cuts specifying the signal definition using truth-level variables.
        n_bins : int, optional
            Number of bins used in the distribution. If not provided, then bin_edges must be provided.
        x_range : tuple, optional
            Tuple of lower and upper limits. If not provided, then bin_edges must be provided.
        bin_edges : np.ndarray, optional
            Array of bin edges. If this is provided, the n_bins and limits are ignored.
        weightCV : str
            The central value weights for the CV histogram. In the PELEE framework, the column "weights" = "weightSplineTimesTune" * data_pot / mc_pot
            so additional scaling is not needed.
        base_weight_Var : str
            Name of the column containing the baseline weights of the events to be used for the variation histograms
            (as this function is to calculate the GENIE multisim, we need to use the weights without the GENIE tune). 
            In the PELEE framework, the column "weights_no_tune" = "weightSpline" * data_pot / mc_pot, so additional scaling is not needed.
        wname : str
            The name of the column containing the multisim weights of the events.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """
        
        # For my own 1e1p TKI branch, I have named my variables in a certain way - not applicable in general
        if true_var_name is None:
                label = var_name.lstrip("Reco")
                true_var_name = "True" + label

        # Sometimes n_bins is none in the Binning object if using variable bin sizes
        if bin_edges is not None:
            n_bins = len(bin_edges) - 1
            bins = bin_edges
        elif x_range is None:
            bins = n_bins
        else:
            bins = np.linspace(x_range[0],x_range[1],n_bins+1)

        Nuniverse = len(rundata["mc"][wname][0]) # should be 100 in the 2024 PELEE n-tuples

        n_tot = np.zeros([Nuniverse, n_bins]) # this will store the variation hists (rows)
        n_cv_tot = np.zeros(n_bins) # this will store the sum of the CV hists

        # First calculate the covariance for the background
        for key, df in rundata.items():
            if key not in ["data", "ext"]: # the calculation should exclude all data and only be performed on MC
                #
                extra_query = ""
                if key == signame:
                    extra_query = f"& ~({truth_def})" # removing the signal events
                        
                from microfit import selections as sel
                query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

                queried_df = df.query(query+extra_query, engine="python")
                variable = queried_df[var_name]
                CV_base_weight = queried_df[weightCV]
                Var_base_weight = queried_df[base_weight_Var]
                syst_weights = queried_df[wname]

                # Calculate the background CV hist sum
                n_cv, bins = np.histogram(variable, bins=bins, weights=CV_base_weight)
                n_cv_tot += n_cv

                # Calculate the background variation hists and store them

                weights_df = pd.DataFrame(syst_weights.values.tolist()) # df of multisim weights that have been flattened horizontally - i.e. Nuniv (100) columns per row

                if not weights_df.empty:
                    for i in range(Nuniverse):
                        weight = weights_df[i].values / 1000.
                        weight[np.isnan(weight)] = 1
                        weight[weight > 100] = 1
                        weight[weight < 0] = 1
                        weight[weight == np.inf] = 1

                        n, bins = np.histogram(variable, weights=weight * Var_base_weight,bins=bins)
                        n_tot[i] += n

        # Now calculate the covariance for the signal using the response matrix method

        df = rundata[signame]
        queried_df = df.query(f"{query} & ({truth_def})", engine="python")
        variable = queried_df[var_name]
        CV_base_weight = queried_df[weightCV]
        Var_base_weight = queried_df[base_weight_Var]
        syst_weights = queried_df[wname]

        # Calculate the signal CV hist sum
        n_cv, bins = np.histogram(variable, bins=bins, weights=CV_base_weight)
        n_cv_tot += n_cv

        # Calculate the signal variation hists and store them

        weights_df = pd.DataFrame(syst_weights.values.tolist()) # df of multisim weights that have been flattened horizontally - i.e. Nuniv (100) columns per row

        # Calculate the true signal CV that stays constant in all the universes
        true_sig = df.query(truth_def, engine="python")
        true_variable = true_sig[true_var_name]
        true_var_weightsCV  = true_sig[weightCV]
        t_cv, bins = np.histogram(true_variable, bins=bins, weights=true_var_weightsCV)

        if not weights_df.empty:
            for i in range(Nuniverse):
                rmv, xb, yb = ResponseMatrix(df, truth_def, query, true_var_name, var_name, bins, base_weight_Var, i, wname)
                #print(rmv)
                rp = rmv.dot(t_cv)
                #print("variation: ",rp)
                n_tot[i] += rp

        # Now finally compute the covariance

        cov = np.zeros([len(n_cv_tot), len(n_cv_tot)])
        for n in n_tot:
                for i in range(len(n_cv_tot)):
                        for j in range(len(n_cv_tot)):
                                cov[i][j] += (n[i] - n_cv_tot[i]) * (n[j] - n_cv_tot[j])
        
        cov /= Nuniverse

        return cov
    
################################################################################

def unisim_err_with_resp_func(rundata, signame, var_name, true_var_name, preselection, selection, truth_def, n_bins=None, x_range=None, bin_edges=None, weightCV="weights", base_weight_Var="weights_no_tune"):

    """Calculate unisim uncertainties using the response matrix method.

        Unisim means that a single variation of a given analysis input parameter is performed according to its uncertainty.
        The difference in the number of selected events between this variation and the central value is taken as the
        uncertainty in that number of events. Mathematically, this is the same as the 'multisim' method, but with only
        one or two universes.

        This particular method, whereby the response matrix is used, is required for xsec analyses.
        Here, the signal and background are treated differently. The covariance for the background
        is calculated as described above. But the covariance for the signal is calculated using the
        response matrix; the variation for a particular universe is calculated by multiplying the
        response matrix for that universe (S^{univ}_{sel} / S^{univ}_{total}) by the total true
        signal CV (central value) prediction.
        
        This function has written to be used with microfit.

        Parameters
        ----------
        rundata : dict
            Dictionary of dataframes for each sample. This is the output from the load_runs()
            function in data_loading.py.
        signame : str
            The name of the MC sample that contains your signal events. This must refer to one of the keys in rundata.
            For my analysis, this is "nue".
        var_name : str
            The name of the reconstructed variable you are calculating the covariance for.
        true_var_name : str
            The name of the truth-level variable corresponding to var_name. This is required, unless var_name and true_var_name
            share the same stem but with either "Reco" or "True" at the front of the string 
            (e.g. var_name = "RecoDeltaPT" and true_var_name = "TrueDeltaPT"), in which case true_var_name is optional.
        preselection : str
            The name of one of the sets of preselection cuts in selections.py in microfit.
        selection : str
            The name of one of the sets of selection cuts in selections.py in microfit.
        truth_def : str
            A set of cuts specifying the signal definition using truth-level variables.
        n_bins : int, optional
            Number of bins used in the distribution. If not provided, then bin_edges must be provided.
        x_range : tuple, optional
            Tuple of lower and upper limits. If not provided, then bin_edges must be provided.
        bin_edges : np.ndarray, optional
            Array of bin edges. If this is provided, the n_bins and limits are ignored.
        weightCV : str
            The central value weights for the CV histogram. In the PELEE framework, the column "weights" = "weightSplineTimesTune" * data_pot / mc_pot
            so additional scaling is not needed.
        base_weight_Var : str
            Name of the column containing the baseline weights of the events to be used for the variation histograms
            (as this function is to calculate the GENIE multisim, we need to use the weights without the GENIE tune). 
            In the PELEE framework, the column "weights_no_tune" = "weightSpline" * data_pot / mc_pot, so additional scaling is not needed.

        Returns
        -------
        cov : array_like
            Covariance matrix of the bin counts.
    """

    # For my own 1e1p TKI branch, I have named my variables in a certain way - not applicable in general
    if true_var_name is None:
        label = var_name.lstrip("Reco")
        true_var_name = "True" + label

    # Sometimes n_bins is none in the Binning object if using variable bin sizes
    if bin_edges is not None:
        n_bins = len(bin_edges) - 1
        bins = bin_edges
    elif x_range is None:
        bins = n_bins
    else:
        bins = np.linspace(x_range[0],x_range[1],n_bins+1)

    knob_v = ['knobRPA']# ,'knobCCMEC','knobAxFFCCQE','knobVecFFCCQE','knobDecayAngMEC','knobThetaDelta2Npi']
    knob_n = [2]# ,1,1,1,1,1]

    n_cv_tot = np.zeros(n_bins) # this will store the sum of the CV hists
    n_tot_v = [] # this will store the variation hists
    # sig_tot_v = []
    for u, knob in enumerate(knob_v):
        n_tot_v.append(np.zeros([ knob_n[u] ,n_bins]))
        # sig_tot_v.append(np.zeros([ knob_n[u] ,n_bins]))

    # First calculate the covariance for the background
    for key, df in rundata.items():
        if key not in ["data", "ext"]: # the calculation should exclude all data and only be performed on MC
            extra_query = ""
            if key == signame:
                extra_query = f"& ~({truth_def})" # removing the signal events
                        
            from microfit import selections as sel
            query = f"{sel.preselection_categories[preselection]['query']} and {sel.selection_categories[selection]['query']}"

            queried_df = df.query(query+extra_query, engine="python")
            variable = queried_df[var_name]
            CV_base_weight = queried_df[weightCV]
            Var_base_weight = queried_df[base_weight_Var]
            #syst_weights = queried_df[wname]

            # Calculate the background CV hist sum
            n_cv, bins = np.histogram(variable, bins=bins, weights=CV_base_weight)
            n_cv_tot += n_cv

            # Calculate the background variation hists and store them
            for n, knob in enumerate(knob_v):
                weight_up = queried_df[f"{knob}up"].values
                weight_up[np.isnan(weight_up)] = 1
                weight_up[weight_up > 100] = 1
                weight_up[weight_up < 0] = 1
                weight_up[weight_up == np.inf] = 1
                n_up, bins = np.histogram(variable, weights=weight_up * Var_base_weight, bins=bins)
                n_tot_v[n][0] += n_up

                if (knob_n[n] == 2):
                    weight_dn = queried_df[f"{knob}dn"].values
                    weight_dn[np.isnan(weight_dn)] = 1
                    weight_dn[weight_dn > 100] = 1
                    weight_dn[weight_dn < 0] = 1
                    weight_dn[weight_dn == np.inf] = 1
                    n_dn, bins = np.histogram(variable, weights=weight_dn * Var_base_weight, bins=bins)
                    n_tot_v[n][1] += n_dn
                
    bkg_vars_dict = dict()
    for n, knob in enumerate(knob_v):
         bkg_vars_dict[knob] = n_tot_v[n]
    # print("My bkg var dict: \n", bkg_vars_dict)
    # print()

    # Now calculate the covariance for the signal using the response matrix method
    df = rundata[signame]
    queried_df = df.query(f"{query} & ({truth_def})", engine="python")
    variable = queried_df[var_name]
    CV_base_weight = queried_df[weightCV]
    Var_base_weight = queried_df[base_weight_Var]

    # Calculate the signal CV hist sum
    n_cv, bins = np.histogram(variable, bins=bins, weights=CV_base_weight)
    n_cv_tot += n_cv
    print("My CV hist: ", n_cv_tot)
    print()
    # Calculate the signal variation hists and store them

    # Calculate the true signal CV that stays constant in all the universes
    true_sig = df.query(truth_def, engine="python")
    true_variable = true_sig[true_var_name]
    true_var_weightsCV  = true_sig[weightCV]
    t_cv, bins = np.histogram(true_variable, bins=bins, weights=true_var_weightsCV)

    if not df.empty:
         for n,knob in enumerate(knob_v):
            rmv_up, xb, yb = ResponseMatrix(df, truth_def, query ,true_var_name, var_name, bins, base_weight_Var, 0, f"{knob}up")
            rp_up = rmv_up.dot(t_cv)
            n_tot_v[n][0] += rp_up
            # sig_tot_v[n][0] += rp_up

            if (knob_n[n] == 2):
                rmv_dn, xb, yb = ResponseMatrix(df,truth_def, query, true_var_name, var_name, bins, base_weight_Var, 0, f"{knob}dn")
                rp_dn = rmv_dn.dot(tcv)
                n_tot_v[n][1] += rp_dn
                # sig_tot_v[n][1] += rp_dn_
        
     # Now finally compute the covariance
    # total_vars_dict = dict()
    # sig_vars_dict = dict()
    # for n,knob in enumerate(knob_v):
    #      total_vars_dict[knob] = n_tot_v[n]
    #      sig_vars_dict[knob] = sig_tot_v[n]

    # # print("My sig var dict: \n", sig_vars_dict)
    # # print()     
    # print("My total var dict: \n", total_vars_dict)
    # print()
    cov = np.zeros([len(n_cv_tot), len(n_cv_tot)])

    for n,knob in enumerate(knob_v):
        
        this_cov = np.zeros([len(n_cv_tot), len(n_cv_tot)])

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
    
################################################################################