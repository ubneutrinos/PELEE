from dataclasses import dataclass, fields
import numpy as np
import pandas as pd

from numbers import Number
from uncertainties import correlated_values, unumpy
import unblinding_far_sideband as far_sb
from .category_definitions import get_category_label, get_category_color
from .statistics import covariance, sideband_constraint_correction, error_propagation_division


class HistGenMixin:
    """Mixin class for histogram generators to store variable, binning, weight_column, and query."""

    def __init__(self, data_columns, binning, weight_column=None, query=None):
        self._weight_column = weight_column
        self._binning = binning
        self._query = query
        self.data_columns = data_columns

    @property
    def variable(self):
        return self._binning.variable

    @variable.setter
    def variable(self, variable):
        if variable not in self.data_columns:
            raise ValueError(f"Variable {variable} is not in the dataframe, cannot set binning")
        self._binning.variable = variable

    @property
    def binning(self):
        return self._binning

    @binning.setter
    def binning(self, binning):
        variable = binning.variable
        if variable not in self.data_columns:
            raise ValueError(f"Variable {variable} is not in the dataframe, cannot set binning")
        self._binning = binning

    @property
    def weight_column(self):
        return self._weight_column

    @weight_column.setter
    def weight_column(self, weight_column):
        if weight_column is not None:
            # it is possible to pass either the name of one single column or a list of
            # column names (where the weights are multiplied)
            if isinstance(weight_column, str):
                if weight_column not in self.data_columns:
                    raise ValueError(f"Weight column {weight_column} is not in the dataframe.")
            elif isinstance(weight_column, list):
                for col in weight_column:
                    if col not in self.data_columns:
                        raise ValueError(f"Weight column {col} is not in the dataframe.")
        self._weight_column = weight_column

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, query):
        if query is not None and not isinstance(query, str):
            raise ValueError("query must be a string.")
        self._query = query

    def _get_query(self, extra_query=None):
        query = self.query
        if extra_query is not None:
            if query is None:
                query = extra_query
            else:
                query = f"{query} & {extra_query}"
        return query

    def check_settings(self):
        """Check that the settings are valid."""
        if self.weight_column is not None:
            # it is possible to pass either the name of one single column or a list of
            # column names (where the weights are multiplied)
            if isinstance(self.weight_column, str):
                if self.weight_column not in self.data_columns:
                    raise ValueError(f"Weight column {self.weight_column} is not in the dataframe.")
            elif isinstance(self.weight_column, list):
                for col in self.weight_column:
                    if col not in self.data_columns:
                        raise ValueError(f"Weight column {col} is not in the dataframe.")
            else:
                raise ValueError("weight_column must be a string or a list of strings.")
        if self.query is not None and not isinstance(self.query, str):
            raise ValueError("query must be a string.")

    def update_settings(self, variable=None, binning=None, weight_column=None, query=None):
        """Update the settings of the histogram generator.

        Parameters
        ----------
        variable : str, optional
            Name of the column containing the data to be binned.
        binning : array_like, optional
            Bin edges of the histogram.
        weight_column : str, optional
            Name of the column containing the weights of the data points.
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.
        """

        if variable is not None:
            self.variable = variable
        if binning is not None:
            self.binning = binning
        if weight_column is not None:
            self.weight_column = weight_column
        if query is not None:
            self.query = query

        self.check_settings()


class RunHistGenerator(HistGenMixin):
    """Histogram generator for data and simulation runs."""

    def __init__(
        self,
        rundata_dict,
        binning,
        weight_column="weights",
        selection=None,
        preselection=None,
        data_pot=None,
        sideband_generator=None,
    ):
        """Create a histogram generator for data and simulation runs.

        This combines data and MC appropriately for the given run. It assumes also that,
        if truth-filtered samples are present, that the corresponding event types have
        already been removed from the 'mc' dataframe. It also assumes that the background sets
        have been scaled to the same POT as the data.

        Parameters
        ----------
        rundata_dict : dict
            Dictionary containing the dataframes for this run. The keys are the names of the
            datasets and the values are the dataframes. This must at least contain the keys
            "data", "mc", and "ext". This dictionary should be returned by the data_loader.
        binning : Binning
            Binning object containing the binning of the histogram.
        weight_column : str, optional
            Name of the column containing the weights of the data points.
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.
        data_pot : float, optional
            POT of the data sample. Required to reweight to a different target POT.
        sideband_generator : RunHistGenerator, optional
            Histogram generator for the sideband data. If provided, the sideband data will be
            used to constrain multisim uncertainties.
        """
        self.rundata_dict = rundata_dict
        self.data_pot = data_pot
        data_columns = rundata_dict["data"].columns
        if weight_column is None:
            weight_column = "weights"
        query = self.get_selection_query(selection, preselection)
        super().__init__(data_columns, binning, weight_column=weight_column, query=query)

        # ensure that the necessary keys are present
        if "data" not in self.rundata_dict.keys():
            raise ValueError("data key is missing from rundata_dict.")
        if "mc" not in self.rundata_dict.keys():
            raise ValueError("mc key is missing from rundata_dict.")
        if "ext" not in self.rundata_dict.keys():
            raise ValueError("ext key is missing from rundata_dict.")

        for k, df in rundata_dict.items():
            # if "dataset_name" in df.columns:
            #     raise ValueError("dataset_name column already exists in dataframe.")
            df["dataset_name"] = k
            df["dataset_name"] = df["dataset_name"].astype("category")
        # make one dataframe for all mc events
        self.df_mc = pd.concat([df for k, df in rundata_dict.items() if k not in ["data", "ext"]])
        self.df_ext = rundata_dict["ext"]
        self.df_data = rundata_dict["data"]
        self.sideband_generator = sideband_generator

    def get_selection_query(self, selection, preselection):
        if selection is None and preselection is None:
            return None
        presel_query = far_sb.preselection_categories[preselection]["query"]
        sel_query = far_sb.selection_categories[selection]["query"]

        if presel_query is None:
            query = sel_query
        elif sel_query is None:
            query = presel_query
        else:
            query = f"{presel_query} and {sel_query}"

        return query

    def get_data_hist(self, type="data", add_error_floor=False, scale_to_pot=None):
        """Get the histogram for the data (or EXT).

        Parameters
        ----------
        type : str, optional
            Type of data to return. Can be "data" or "ext".
        add_error_floor : bool, optional
            Whether to add an error floor to the histogram in bins with zero entries.
        scale_to_pot : float, optional
            POT to scale the data to. Only applicable to EXT data.

        Returns
        -------
        data_hist : numpy.ndarray
            Histogram of the data.
        """

        assert type in ["data", "ext"]
        scale_factor = 1.0
        if scale_to_pot is not None:
            if type == "data":
                raise ValueError("Cannot scale data to POT.")
            assert self.data_pot is not None, "Must provide data POT to scale EXT data."
            scale_factor = scale_to_pot / self.data_pot
        dataframe = self.df_data if type == "data" else self.df_ext
        # The weights here are all 1.0 for data, but may be scaled for EXT
        # to match the total number of triggers.
        hist_generator = HistogramGenerator(dataframe, self.binning, weight_column="weights", query=self.query)
        data_hist = hist_generator.generate(add_error_floor=add_error_floor)
        data_hist *= scale_factor
        data_hist.label = {"data": "Data", "ext": "EXT"}[type]
        data_hist.color = {"data": "k", "ext": "yellow"}[type]
        return data_hist

    def get_mc_hists(self, category_column="dataset_name", include_multisim_errors=False, scale_to_pot=None):
        """Get MC histograms that are split by event category.

        Parameters
        ----------
        category_column : str, optional
            Name of the column containing the event categories.
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.

        Returns
        -------
        mc_hists : dict
            Dictionary containing the histograms for each event category. Keys are the
            category names and values are the histograms.
        """

        if scale_to_pot is not None:
            assert self.data_pot is not None, "data_pot must be set to scale to a different POT."
        mc_hists = {}
        other_categories = []
        for category in self.df_mc[category_column].unique():
            extra_query = f"{category_column} == '{category}'"
            hist = self.get_mc_hist(
                include_multisim_errors=include_multisim_errors, extra_query=extra_query, scale_to_pot=scale_to_pot
            )
            if category_column == "dataset_name":
                hist.label = str(category)
            else:
                hist.label = get_category_label(category_column, category)
                hist.color = get_category_color(category_column, category)
                if hist.label == "Other":
                    other_categories.append(category)
            mc_hists[category] = hist
        # before we return the histogram dict, we want to sum all categories together
        # that were labeled as "Other"
        if len(other_categories) > 0:
            mc_hists["Other"] = sum([mc_hists.pop(cat) for cat in other_categories])
            mc_hists["Other"].label = "Other"
            mc_hists["Other"].color = "gray"
        return mc_hists

    def get_hist_generator(self, which, extra_query=None):
        assert which in ["mc", "data", "ext"]
        query = self._get_query(extra_query)
        dataframe = {"mc": self.df_mc, "data": self.df_data, "ext": self.df_ext}[which]
        hist_generator = HistogramGenerator(dataframe, self.binning, weight_column=self.weight_column, query=query)
        return hist_generator

    def get_mc_hist(self, include_multisim_errors=False, extra_query=None, scale_to_pot=None, use_sideband=False):
        """Produce a histogram from the MC dataframe.

        Parameters
        ----------
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
        """

        scale_factor = 1.0
        if scale_to_pot is not None:
            assert self.data_pot is not None, "data_pot must be set to scale to a different POT."
            scale_factor = scale_to_pot / self.data_pot
        hist_generator = self.get_hist_generator(which="mc", extra_query=extra_query)
        hist = hist_generator.generate()
        hist.label = "MC"
        use_sideband = use_sideband and self.sideband_generator is not None
        if include_multisim_errors:
            if use_sideband:
                # We need the hist-generator to calculate the universe histograms later
                sideband_hist_generator = self.sideband_generator.get_hist_generator(which="mc")
                # We calculate the nominal values and *full* covariance matrix for the sideband data
                sideband_mc_cv_hist = self.sideband_generator.get_mc_hist(include_multisim_errors=True)
                # The extended covariance matrix contains the covariance of the concatenated
                # central value and sideband data histograms.
                extended_cov_mat = np.zeros(
                    (hist.n_bins + sideband_mc_cv_hist.n_bins, hist.n_bins + sideband_mc_cv_hist.n_bins)
                )
                extended_central_value = np.concatenate((hist.nominal_values, sideband_mc_cv_hist.nominal_values))
                sideband_cov_mat = np.zeros((sideband_mc_cv_hist.n_bins, sideband_mc_cv_hist.n_bins))
            for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
                # The GENIE variations are applied instead of the central value tuning, so we need to use the
                # weights_no_tune column instead of the weights column.
                weight_column = "weights_no_tune" if ms_column == "weightsGenie" else "weights"
                cov_mat, universe_hists = hist_generator.calculate_multisim_uncertainties(
                    ms_column,
                    weight_column=weight_column,
                    central_value_hist=hist,
                    extra_query=extra_query,
                    return_histograms=True,
                )
                hist.add_covariance(cov_mat)
                if not use_sideband:
                    continue

                sb_cov_mat, sideband_universe_hists = sideband_hist_generator.calculate_multisim_uncertainties(
                    ms_column,
                    weight_column=weight_column,
                    # when calculating multisim uncertainties, the central value is that of MC alone
                    central_value_hist=sideband_mc_cv_hist,
                    return_histograms=True,
                )
                concat_observations = np.concatenate([universe_hists, sideband_universe_hists], axis=1)
                extended_cov_mat_contrib = covariance(concat_observations, extended_central_value)
                # sanity check: the upper left block of the extended covariance should be the same
                # as cov_mat, and the lower right block should be the same as sb_cov_mat
                assert np.all(np.equal(extended_cov_mat_contrib[: hist.n_bins, : hist.n_bins], cov_mat))
                assert np.all(np.equal(extended_cov_mat_contrib[hist.n_bins :, hist.n_bins :], sb_cov_mat))
                extended_cov_mat += extended_cov_mat_contrib
                sideband_cov_mat += sb_cov_mat

            cov_mat_unisim = hist_generator.calculate_unisim_uncertainties(
                central_value_hist=hist, extra_query=extra_query
            )
            hist.add_covariance(cov_mat_unisim)
            # If we are using a sideband, we can now calculate the correction to the central value
            # and the covariance matrix from the sideband data.
            if use_sideband:
                sideband_data_hist = self.sideband_generator.get_data_hist()
                sideband_ext_hist = self.sideband_generator.get_data_hist(type="ext")
                sideband_total_prediction = sideband_mc_cv_hist + sideband_ext_hist
                mu_offset, cov_corr = sideband_constraint_correction(
                    # The data that was actually observed
                    sideband_measurement=sideband_data_hist.nominal_values,
                    # to calculate the constraint, we want to compare the data to the total prediction
                    # in the sideband, which is the MC prediction plus the beam-off EXT data
                    sideband_central_value=sideband_total_prediction.nominal_values,
                    concat_covariance=extended_cov_mat,
                    # We also want to use the covariance matrix of the total prediction for the
                    # sideband, which includes the covariance of the MC and EXT data
                    sideband_covariance=sideband_total_prediction.cov_matrix,
                )
                # finally we add the corrections to the histogram
                hist += mu_offset
                hist.add_covariance(cov_corr)

        hist *= scale_factor
        return hist


class HistogramGenerator(HistGenMixin):
    def __init__(self, dataframe, binning, weight_column="weights", query=None):
        """Create a histogram generator for a given dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the data to be binned.
        weight_column : str or list of str, optional
            Name of the column containing the weights of the data points. If more than one
            weight column is given, the weights are multiplied in sequence.
        """
        self.dataframe = dataframe
        data_columns = dataframe.columns
        super().__init__(data_columns, binning, weight_column=weight_column, query=query)

    def generate(self, query=None, add_error_floor=False):
        """Generate a histogram from the dataframe.

        Parameters
        ----------
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.
        add_error_floor : bool, optional
            Add a minimum error of 1.4 to empty bins. This is motivated by a Bayesian
            prior of a unit step function as documented in
            https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=32714&filename=Monte_Carlo_Uncertainties.pdf&version=1
            and is currently only applied to EXT events.

        Returns
        -------
        histogram : Histogram
            Histogram object containing the binned data.
        """
        self.update_settings(variable=None, binning=None, weight_column=None, query=query)

        if self.query is not None:
            dataframe = self.dataframe.query(self.query)
        else:
            dataframe = self.dataframe

        weights = self.get_weights(weight_column=self.weight_column)
        bin_counts, bin_edges = np.histogram(dataframe[self.variable], bins=self.binning.bin_edges, weights=weights)
        variances, _ = np.histogram(dataframe[self.variable], bins=self.binning.bin_edges, weights=weights**2)
        if add_error_floor:
            variances[variances == 0] = 1.4**2
        return Histogram(self.binning, bin_counts, uncertainties=np.sqrt(variances))

    def get_weights(self, weight_column=None, limit_weight=True, query=None):
        """Get the weights of the dataframe.

        Parameters
        ----------
        weight_column : str or list of str, optional
            Override the weight column given at initialization.
        limit_weight : bool, optional
            Reset invalid weights to one.

        Returns
        -------
        weights : array_like
            Array of weights.
        """
        if self.query is not None:
            dataframe = self.dataframe.query(self.query)
        else:
            dataframe = self.dataframe
        if weight_column is None:
            weight_column = self.weight_column
        if weight_column is None:
            return np.ones(len(dataframe))
        # the weight column might be a list of columns that need to be multiplied
        if isinstance(weight_column, list):
            weights = np.ones(len(dataframe))
            for col in weight_column:
                weights *= dataframe[col]
        else:
            weights = self._limit_weights(dataframe[weight_column])
        return weights

    def _limit_weights(self, weights):
        weights = np.asarray(weights)
        weights[weights > 100] = 1.0
        weights[weights < 0] = 1.0
        weights[~np.isfinite(weights)] = 1.0
        return weights

    def calculate_multisim_uncertainties(
        self,
        multisim_weight_column,
        weight_rescale=1 / 1000,
        weight_column=None,
        central_value_hist=None,
        extra_query=None,
        return_histograms=False,
    ):
        """Calculate multisim uncertainties.

        Each of the given multisim weight columns is expected to contain a list of weights
        for every row that correspond to the weights of the fluctuated "universes". The
        histogram is regenerated for every universe and the covariance matrix is calculated
        from the resulting histograms. Optionally, a central value histogram can be given
        that is used to calculate the covariance matrix.

        Parameters
        ----------
        multisim_weight_columns : list of str
            List of names of the columns containing the multisim weights.
        weight_rescale : float, optional
            Rescale factor for the weights. Typically, multisim weights are stored as ints
            that are multiplied by a factor of 1000.
        weight_column : str, optional
            Name of the column containing the baseline weights of the events. If not given,
            the baseline weight that this histogram generator was initialized with is used.
        central_value_hist : Histogram, optional
            Histogram containing the central value of the multisim weights. If not given,
            the covariance is calculated from the mean of the histograms.
        extra_query : str, optional
            Query to apply to the dataframe before calculating the covariance matrix.
        return_histograms : bool, optional
            If True, return the histograms of the universes.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """

        self.check_settings()

        if multisim_weight_column not in self.dataframe.columns:
            raise ValueError(f"Weight column {multisim_weight_column} is not in the dataframe.")
        query = self._get_query(extra_query)
        dataframe = self.dataframe.query(query)
        multisim_weights = dataframe[multisim_weight_column].values
        # We have to make sure that there are no NaNs in the weights. Every row should contain
        # a list or np.ndarray of values of the same length. If there are NaNs, this indicates that the
        # selection might contain events that are acually not MC events (such as EXT). We
        # cannot calculate systematic errors in this case.
        # Check that they are all of type list or np.ndarray
        if not all(isinstance(x, (list, np.ndarray)) for x in multisim_weights):
            raise ValueError("Not all elements are lists or np.ndarrays.")
        # Check that all lists have the same length
        if not all(len(x) == len(multisim_weights[0]) for x in multisim_weights):
            raise ValueError("Not all lists have the same length.")
        df = pd.DataFrame(multisim_weights.tolist())
        # every column in df now contains the weights for one universe
        universe_histograms = []
        base_weights = self.get_weights(weight_column=weight_column, query=query)
        for column in df.columns:
            # create a histogram for each universe
            bincounts, _ = np.histogram(
                dataframe[self.variable],
                bins=self.binning.bin_edges,
                weights=base_weights * df[column].values * weight_rescale,
            )
            universe_histograms.append(bincounts)
        universe_histograms = np.array(universe_histograms)
        # calculate the covariance matrix from the histograms
        cov = covariance(
            universe_histograms, central_value_hist.nominal_values if central_value_hist is not None else None
        )
        if return_histograms:
            return cov, universe_histograms
        return cov

    def calculate_unisim_uncertainties(self, central_value_hist, extra_query=None):
        """Calculate unisim uncertainties.

        Unisim means that a single variation of a given analysis input parameter is performed according to its uncertainty.
        The difference in the number of selected events between this variation and the central value is taken as the
        uncertainty in that number of events. Mathematically, this is the same as the 'multisim' method, but with only
        one or two universes. The central value is in this case not optional.

        Parameters
        ----------
        central_value_hist : Histogram
            Central value histogram.
        extra_query : str, optional
            Extra query to apply to the dataframe before calculating the covariance matrix.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """

        query = self._get_query(extra_query)
        knob_v = ["knobRPA", "knobCCMEC", "knobAxFFCCQE", "knobVecFFCCQE", "knobDecayAngMEC", "knobThetaDelta2Npi"]
        # see table 23 from the technote
        knob_n_universes = [2, 1, 1, 1, 1, 1]
        # because all of these are GENIE knobs, we need to use the weights without the GENIE tune just as
        # for the GENIE multisim
        base_weight = "weights_no_tune"
        # When we have two universes, then there are two weight variations, knobXXXup and knobXXXdown. Otherwise, there
        # is only one weight variation, knobXXXup.
        total_cov = np.zeros((len(self.binning), len(self.binning)))
        base_weights = self.get_weights(weight_column=base_weight)
        dataframe = self.dataframe.query(query)
        for knob, n_universes in zip(knob_v, knob_n_universes):
            observations = []
            for universe in range(n_universes):
                # get the weight column for this universe
                weight_column_knob = f"{knob}up" if n_universes == 2 and universe == 0 else f"{knob}dn"
                universe_weights = self.get_weights(weight_column=weight_column_knob, query=query)
                # calculate the histogram for this universe
                bincounts, _ = np.histogram(
                    dataframe[self.variable],
                    bins=self.binning.bin_edges,
                    weights=base_weights * universe_weights,
                )
                observations.append(bincounts)
            observations = np.array(observations)
            # calculate the covariance matrix from the histograms
            cov = covariance(observations, central_value_hist.nominal_values)
            # add it to the total covariance matrix
            total_cov += cov
        return total_cov


class Histogram:
    def __init__(
        self,
        binning,
        bin_counts,
        uncertainties=None,
        covariance_matrix=None,
        label=None,
        plot_color=None,
        tex_string=None,
    ):
        """Create a histogram object.

        Parameters
        ----------
        binning : Binning
            Binning of the histogram.
        bin_counts : array_like
            Bin counts of the histogram.
        uncertainties : array_like, optional
            Uncertainties of the bin counts.
        covariance_matrix : array_like, optional
            Covariance matrix of the bin counts.
        label : str, optional
            Label of the histogram. This is distinct from the label of the x-axis, which is
            set in the Binning object that is passed to the constructor.
        plot_color : str, optional
            Color of the histogram, used for plotting.
        tex_string : str, optional
            TeX string used to label the histogram in a plot.
        """

        self.binning = binning
        self.bin_counts = bin_counts
        assert self.binning.n_bins == len(self.bin_counts), "bin_counts must have the same length as binning."
        self._label = label
        self._plot_color = plot_color
        self._tex_string = tex_string

        if covariance_matrix is not None:
            self.cov_matrix = np.array(covariance_matrix)
            self.bin_counts = np.array(correlated_values(bin_counts, self.cov_matrix))
        elif uncertainties is not None:
            self.cov_matrix = np.diag(np.array(uncertainties) ** 2)
            self.bin_counts = unumpy.uarray(bin_counts, uncertainties)
        else:
            raise ValueError("Either uncertainties or covariance_matrix must be provided.")

    def draw(self, ax, as_errorbars=False, **plot_kwargs):
        """Draw the histogram on a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw the histogram on.
        as_errorbars : bool, optional
            Whether to draw the histogram as errorbars.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to the matplotlib step function.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if self.binning.is_log:
            ax.set_yscale("log")
        ax.set_xlabel(self.binning.label)
        ax.set_ylabel("Events")
        bin_counts = self.nominal_values
        bin_edges = self.binning.bin_edges
        label = plot_kwargs.pop("label", self.tex_string)
        color = plot_kwargs.pop("color", self.color)
        if as_errorbars:
            ax.errorbar(
                self.binning.bin_centers,
                bin_counts,
                yerr=self.std_devs,
                linestyle="none",
                marker=".",
                label=label,
                color=color,
                **plot_kwargs,
            )
            return ax
        errband_alpha = plot_kwargs.pop("alpha", 0.5)
        # Be sure to repeat the last bin count
        bin_counts = np.append(bin_counts, bin_counts[-1])
        p = ax.step(bin_edges, bin_counts, where="post", label=label, color=color, **plot_kwargs)

        # plot uncertainties as a shaded region
        uncertainties = self.std_devs
        uncertainties = np.append(uncertainties, uncertainties[-1])
        # ensure that error band has the same color as the plot we made earlier unless otherwise specified
        color = p[0].get_color()

        ax.fill_between(
            bin_edges,
            np.clip(bin_counts - uncertainties, 0, None),
            bin_counts + uncertainties,
            alpha=errband_alpha,
            step="post",
            color=color,
            **plot_kwargs,
        )

    def _repr_html_(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self.draw(ax)
        ax.legend()
        return fig._repr_html_()

    def to_dict(self):
        """Convert the histogram to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the histogram.
        """

        return {
            "binning": self.binning.__dict__,
            "bin_counts": self.nominal_values,
            "covariance_matrix": self.cov_matrix,
            "label": self._label,
            "plot_color": self._plot_color,
            "tex_string": self._tex_string,
        }

    @classmethod
    def from_dict(cls, dictionary):
        """Create a histogram from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary representation of the histogram.

        Returns
        -------
        Histogram
            Histogram object.
        """

        dictionary["binning"] = Binning(**dictionary["binning"])
        return cls(**dictionary)

    def __eq__(self, other):
        """Compare two histograms.

        Parameters
        ----------
        other : Histogram
            Other histogram to compare to.

        Returns
        -------
        bool
            Whether the histograms are equal.
        """

        return (
            self.binning == other.binning
            and np.all(self.nominal_values == other.nominal_values)
            and np.all(self.cov_matrix == other.cov_matrix)
            and self.label == other.label
            and self.color == other.color
            and self.tex_string == other.tex_string
        )

    @property
    def bin_centers(self):
        return self.binning.bin_centers

    @property
    def n_bins(self):
        return len(self.binning)

    @property
    def color(self):
        # We let the plotter handle the case when this is None, in which case
        # it will assign a color automatically.
        return self._plot_color

    @color.setter
    def color(self, value):
        self._plot_color = value

    @property
    def label(self):
        if self._label is None:
            return ""
        else:
            return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def tex_string(self):
        # if we don't have a tex string defined, we use the label
        if self._tex_string is None:
            return self.label
        else:
            return self._tex_string

    @tex_string.setter
    def tex_string(self, value):
        self._tex_string = value

    @property
    def nominal_values(self):
        return unumpy.nominal_values(self.bin_counts)

    @property
    def std_devs(self):
        return unumpy.std_devs(self.bin_counts)

    @property
    def corr_matrix(self):
        # convert the covariance matrix into a correlation matrix
        # ignore division by zero error
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.cov_matrix / np.outer(self.std_devs, self.std_devs)

    def sum(self):
        return np.sum(self.nominal_values)

    def add_covariance(self, cov_mat):
        """Add a covariance matrix to the uncertainties of the histogram.

        The covariance matrix is added to the existing covariance matrix. This can be used
        to add systematic uncertainties to a histogram. The uncertainties of the
        bin counts are updated accordingly.

        Parameters
        ----------
        cov_mat : array_like
            Covariance matrix to be added to the histogram.
        """

        self.cov_matrix += cov_mat
        self.bin_counts = np.array(correlated_values(self.nominal_values, self.cov_matrix))

    def fluctuate(self, seed=None):
        """Fluctuate bin counts according to uncertainties and return a new histogram with the fluctuated counts."""
        # take full correlation into account
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.multivariate_normal(unumpy.nominal_values(self.bin_counts), self.cov_matrix)
        # clip bin counts from below
        fluctuated_bin_counts[fluctuated_bin_counts < 0] = 0
        return Histogram(self.binning, fluctuated_bin_counts, covariance_matrix=self.cov_matrix)

    def fluctuate_poisson(self, seed=None):
        """Fluctuate bin counts according to Poisson uncertainties and return a new histogram with the fluctuated counts."""
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.poisson(unumpy.nominal_values(self.bin_counts))
        return Histogram(self.binning, fluctuated_bin_counts, uncertainties=np.sqrt(fluctuated_bin_counts))

    def __repr__(self):
        return f"Histogram(binning={self.binning}, bin_counts={self.bin_counts}, covariance={self.cov_matrix}, label={self.label}, tex={self.tex_string})"

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            # We can add or subtract an ndarray to a Histogram as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot add ndarray to histogram: ndarray has wrong length.")
            new_bin_counts = self.bin_counts + other
            return Histogram(
                self.binning,
                unumpy.nominal_values(new_bin_counts),
                covariance_matrix=self.cov_matrix,
                label=self.label,
                tex_string=self.tex_string,
            )
        assert self.binning == other.binning, "Cannot add histograms with different binning."
        new_bin_counts = self.bin_counts + other.bin_counts
        label = self.label if self.label == other.label else "+".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        return Histogram(
            self.binning,
            unumpy.nominal_values(new_bin_counts),
            covariance_matrix=new_cov_matrix,
            label=label,
            tex_string=self.tex_string,
        )

    def __radd__(self, other):
        # This function adds support for sum() to work with histograms. sum() starts with 0, so we need to handle that case.
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        # We can add or subtract an ndarray to a Histogram as long as the length of the ndarray is the same as the number of bins

        # if other is an ndarray
        if isinstance(other, np.ndarray):
            if len(other) != len(self.bin_counts):
                raise ValueError(
                    f"Cannot subtract ndarray of length {len(other)} from histogram with {self.n_bins} bins."
                )
            new_bin_counts = self.bin_counts - other
            return Histogram(
                self.binning,
                unumpy.nominal_values(new_bin_counts),
                covariance_matrix=self.cov_matrix,
                label=self.label,
                tex_string=self.tex_string,
            )
        # otherwise, if other is also a Histogram
        assert self.binning == other.binning, "Cannot add histograms with different binning."
        new_bin_counts = self.bin_counts - other.bin_counts
        label = self.label if self.label == other.label else "-".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        return Histogram(
            self.binning,
            unumpy.nominal_values(new_bin_counts),
            covariance_matrix=new_cov_matrix,
            label=label,
            tex_string=self.tex_string,
        )

    def __truediv__(self, other):
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values / other
            new_cov_matrix = self.cov_matrix / other**2
            return Histogram(
                self.binning,
                new_bin_counts,
                covariance_matrix=new_cov_matrix,
                label=self.label,
                # need to bypass the getter method to do this correctly
                tex_string=self._tex_string,
            )
        assert self.binning == other.binning, "Cannot divide histograms with different binning."
        new_bin_counts, new_cov_matrix = error_propagation_division(
            self.nominal_values, other.nominal_values, self.cov_matrix, other.cov_matrix
        )
        label = self.label if self.label == other.label else "/".join([self.label, other.label])
        return Histogram(
            self.binning,
            new_bin_counts,
            covariance_matrix=new_cov_matrix,
            label=label,
            tex_string=self.tex_string,
        )

    def __mul__(self, other):
        # we only support multiplication by numbers that scale the entire histogram
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.cov_matrix * other**2
            return Histogram(
                self.binning,
                new_bin_counts,
                covariance_matrix=new_cov_matrix,
                label=self.label,
                # need to bypass the getter method to do this correctly
                tex_string=self._tex_string,
            )
        else:
            raise NotImplementedError("Histogram multiplication is only supported for numeric types.")

    def __rmul__(self, other):
        # we only support multiplication by numbers that scale the entire histogram
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.cov_matrix * other**2
            return Histogram(
                self.binning,
                new_bin_counts,
                covariance_matrix=new_cov_matrix,
                label=self.label,
                # need to bypass the getter method to do this correctly
                tex_string=self._tex_string,
            )
        else:
            raise NotImplementedError("Histogram multiplication is only supported for numeric types.")


@dataclass
class Binning:
    """Binning for a variable

    Attributes:
    -----------
    variable : str
        Name of the variable being binned
    bin_edges : np.ndarray
        Array of bin edges
    label : str
        Label for the binned variable. This will be used to label the x-axis in plots.
    is_log : bool, optional
        Whether the binning is logarithmic or not (default is False)
    """

    variable: str
    bin_edges: np.ndarray
    label: str
    is_log: bool = False

    def __eq__(self, other):
        if isinstance(other, Binning):
            for field in fields(self):
                attr_self = getattr(self, field.name)
                attr_other = getattr(other, field.name)
                if isinstance(attr_self, np.ndarray) and isinstance(attr_other, np.ndarray):
                    if not np.array_equal(attr_self, attr_other):
                        return False
                else:
                    if attr_self != attr_other:
                        return False
            return True
        return False

    def __post_init__(self):
        if isinstance(self.bin_edges, list):
            self.bin_edges = np.array(self.bin_edges)

    def __len__(self):
        return self.n_bins

    @classmethod
    def from_bounds(cls, var, n_bins, bounds, label, is_log=False):
        """Create a Binning object from bounds

        Parameters:
        -----------
        var : str
            Name of the variable being binned
        n_bins : int
            Number of bins
        bounds : tuple
            Tuple of lower and upper bounds
        label : str
            Label for the binned variable. This will be used to label the x-axis in plots.
        is_log : bool, optional
            Whether the binning is logarithmic or not (default is False)

        Returns:
        --------
        Binning
            A Binning object with the specified bounds
        """
        if is_log:
            bin_edges = np.geomspace(*bounds, n_bins + 1)
        else:
            bin_edges = np.linspace(*bounds, n_bins + 1)
        return cls(var, bin_edges, label, is_log=is_log)

    @property
    def n_bins(self):
        """Number of bins"""
        return len(self.bin_edges) - 1

    @property
    def bin_centers(self):
        """Array of bin centers"""
        if self.is_log:
            return np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        else:
            return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2



