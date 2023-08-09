import numpy as np
import unittest
import pandas as pd
from uncertainties import correlated_values, unumpy
from .category_definitions import get_category_label, get_category_color


class HistGenMixin:
    """Mixin class for histogram generators to store variable, binning, weight_column, and query."""

    def __init__(self, data_columns, weight_column=None, variable=None, binning=None, query=None):
        self._weight_column = weight_column
        self._variable = variable
        self._binning = binning
        self._query = query
        self.data_columns = data_columns

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, variable):
        if variable is not None and variable not in self.data_columns:
            raise ValueError(f"Variable {variable} is not in the dataframe.")
        self._variable = variable

    @property
    def binning(self):
        return self._binning

    @binning.setter
    def binning(self, binning):
        if binning is not None and not isinstance(binning, np.ndarray):
            raise ValueError("binning must be a numpy array.")
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

    def check_settings(self):
        """Check that the settings are valid."""
        if self.variable is not None and self.variable not in self.data_columns:
            raise ValueError(f"Variable {self.variable} is not in the dataframe.")
        if self.binning is not None and not isinstance(self.binning, np.ndarray):
            raise ValueError("binning must be a numpy array.")
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

    def __init__(self, rundata_dict, weight_column=None, variable=None, binning=None, query=None):
        """Create a histogram generator for data and simulation runs.

        This combines data and MC appropriately for the given run. It assumes also that,
        if truth-filtered samples are present, that the corresponding event types have
        already been removed from the 'mc' dataframe.

        Parameters
        ----------
        rundata_dict : dict
            Dictionary containing the dataframes for this run. The keys are the names of the
            datasets and the values are the dataframes. This must at least contain the keys
            "data", "mc", and "ext". This dictionary should be returned by the data_loader.
        weight_column : str, optional
            Name of the column containing the weights of the data points.
        variable : str, optional
            Name of the column containing the data to be binned.
        binning : array_like, optional
            Bin edges of the histogram.
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.
        """
        self.rundata_dict = rundata_dict
        data_columns = rundata_dict["data"].columns
        if weight_column is None:
            weight_column = "weights"
        super().__init__(data_columns, weight_column=weight_column, variable=variable, binning=binning, query=query)

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

    def get_data_hist(self, type="data"):
        """Get the histogram for the data (or EXT).

        Returns
        -------
        data_hist : numpy.ndarray
            Histogram of the data.
        """

        assert type in ["data", "ext"]
        dataframe = self.df_data if type == "data" else self.df_ext
        # The weights here are all 1.0 for data, but may be scaled for EXT
        # to match the total number of triggers.
        hist_generator = HistogramGenerator(
            dataframe, weight_column="weights", variable=self.variable, binning=self.binning, query=self.query
        )
        data_hist = hist_generator.generate()
        data_hist.label = {"data": "Data", "ext": "EXT"}[type]
        data_hist.color = {"data": "k", "ext": "yellow"}[type]
        return data_hist

    def get_mc_hists(self, category_column="dataset_name", include_multisim_errors=False):
        """Get MC histograms that are split by event category.

        Parameters
        ----------
        category_column : str, optional
            Name of the column containing the event categories.
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration.

        Returns
        -------
        mc_hists : dict
            Dictionary containing the histograms for each event category. Keys are the
            category names and values are the histograms.
        """

        mc_hists = {}
        other_categories = []
        for category in self.df_mc[category_column].unique():
            if self.query is not None:
                query = f"{category_column} == '{category}' & {self.query}"
            else:
                query = f"{category_column} == '{category}'"
            hist_generator = HistogramGenerator(
                self.df_mc, weight_column=self.weight_column, variable=self.variable, binning=self.binning, query=query
            )
            hist = hist_generator.generate()

            if include_multisim_errors:
                for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
                    # The GENIE variations are applied instead of the central value tuning, so we need to use the
                    # weights_no_tune column instead of the weights column.
                    weight_column = "weights_no_tune" if ms_column == "weightsGenie" else "weights"
                    cov_mat = hist_generator.calculate_multisim_uncertainties(
                        ms_column, central_value_hist=hist, weight_column=weight_column
                    )
                    hist.add_covariance(cov_mat)
                # cov_mat = hist_generator.calculate_unisim_uncertainties(central_value_hist=hist)
                # hist.add_covariance(cov_mat)
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


class HistogramGenerator(HistGenMixin):
    def __init__(self, dataframe, weight_column=None, variable=None, binning=None, query=None):
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
        super().__init__(data_columns, weight_column=weight_column, variable=variable, binning=binning, query=query)

    def generate(self, variable=None, binning=None, weight_column=None, query=None, add_error_floor=False):
        """Generate a histogram from the dataframe.

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
        self.update_settings(variable, binning, weight_column, query)

        if self.query is not None:
            dataframe = self.dataframe.query(self.query)

        weights = self.get_weights(weight_column=weight_column)
        bin_counts, bin_edges = np.histogram(dataframe[self.variable], bins=self.binning, weights=weights)
        variances, _ = np.histogram(dataframe[self.variable], bins=self.binning, weights=weights**2)
        if add_error_floor:
            variances[variances == 0] = 1.4**2
        return Histogram(bin_edges, bin_counts, uncertainties=np.sqrt(variances))

    def _cov(self, observations, central_value=None):
        """Calculate the covariance matrix of the given observations.

        Optionally, a central value can be given that will be used instead of the mean of the
        observations. This is useful for calculating the covariance matrix of the multisim
        uncertainties where the central value is the nominal MC prediction. Note that the
        calculation of the covariance matrix has to be normalized by (N - 1) if there
        is no central value given, which is done internally by numpy.cov.

        Parameters
        ----------
        observations : array_like
            Array of observations.
        central_value : array_like, optional
            Central value of the observations.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the observations.
        """

        # Make sure that the observations and the central value are both ndarray
        observations = np.asarray(observations)
        if central_value is not None:
            central_value = np.asarray(central_value)
        # make sure the central value, if given, has the right length
        if central_value is not None:
            if central_value.shape[0] != observations.shape[1]:
                raise ValueError("Central value has wrong length.")
        # calculate covariance matrix
        if central_value is None:
            return np.cov(observations, rowvar=False)
        else:
            cov = np.zeros((observations.shape[1], observations.shape[1]))
            for i in range(observations.shape[1]):
                for j in range(observations.shape[1]):
                    cov[i, j] = np.sum(
                        (observations[:, i] - central_value[i]) * (observations[:, j] - central_value[j])
                    )
            # Here, we normalize by 1 / N, rather than 1 / (N - 1) as done by numpy.cov, because we
            # used the given central value rather than calculating it from the observations.
            return cov / observations.shape[0]

    def get_weights(self, weight_column=None, limit_weight=True):
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

        dataframe = self.dataframe.query(self.query)
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
        self, multisim_weight_column, weight_rescale=1 / 1000, weight_column=None, central_value_hist=None
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

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """

        self.check_settings()

        if multisim_weight_column not in self.dataframe.columns:
            raise ValueError(f"Weight column {multisim_weight_column} is not in the dataframe.")
        dataframe = self.dataframe.query(self.query)
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
        base_weights = self.get_weights(weight_column=weight_column)
        for column in df.columns:
            # create a histogram for each universe
            bincounts, _ = np.histogram(
                dataframe[self.variable], bins=self.binning, weights=base_weights * df[column].values * weight_rescale
            )
            universe_histograms.append(bincounts)
        universe_histograms = np.array(universe_histograms)
        # calculate the covariance matrix from the histograms
        return self._cov(
            universe_histograms, central_value_hist.nominal_values if central_value_hist is not None else None
        )

    def calculate_unisim_uncertainties(self, central_value_hist):
        """Calculate unisim uncertainties.

        Unisim means that a single variation of a given analysis input parameter is performed according to its uncertainty.
        The difference in the number of selected events between this variation and the central value is taken as the
        uncertainty in that number of events. Mathematically, this is the same as the 'multisim' method, but with only
        one or two universes. The central value is in this case not optional.

        Parameters
        ----------
        central_value_hist : Histogram
            Central value histogram.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """

        knob_v = ["knobRPA", "knobCCMEC", "knobAxFFCCQE", "knobVecFFCCQE", "knobDecayAngMEC", "knobThetaDelta2Npi"]
        # see table 23 from the technote
        knob_n_universes = [2, 1, 1, 1, 1, 1]
        # because all of these are GENIE knobs, we need to use the weights without the GENIE tune just as 
        # for the GENIE multisim
        base_weight = "weights_no_tune"
        # When we have two universes, then there are two weight variations, knobXXXup and knobXXXdown. Otherwise, there
        # is only one weight variation, knobXXXup.
        total_cov = np.zeros((len(self.binning) - 1, len(self.binning) - 1))
        import pdb; pdb.set_trace()
        for knob, n_universes in zip(knob_v, knob_n_universes):
            observations = []
            for universe in range(n_universes):
                # get the weight column for this universe
                weight_column_knob = f"{knob}up" if n_universes == 2 and universe == 0 else f"{knob}dn"
                # calculate the histogram for this universe
                bincounts, _ = np.histogram(
                    self.dataframe[self.variable],
                    bins=self.binning,
                    weights=self.dataframe[base_weight].values * self.dataframe[weight_column_knob].values,
                )
                observations.append(bincounts)
            observations = np.array(observations)
            # calculate the covariance matrix from the histograms
            cov = self._cov(observations, central_value_hist.nominal_values)
            # add it to the total covariance matrix
            total_cov += cov
        return total_cov

class Histogram:
    def __init__(
        self,
        bin_edges,
        bin_counts,
        uncertainties=None,
        covariance_matrix=None,
        label=None,
        plot_color=None,
        tex_string=None,
        log_scale=False,
    ):
        """Create a histogram object.

        Parameters
        ----------
        bin_edges : array_like
            Bin edges of the histogram.
        bin_counts : array_like
            Bin counts of the histogram.
        uncertainties : array_like, optional
            Uncertainties of the bin counts.
        covariance_matrix : array_like, optional
            Covariance matrix of the bin counts.
        label : str, optional
            Label of the histogram.
        plot_color : str, optional
            Color of the histogram, used for plotting.
        tex_string : str, optional
            TeX string of the histogram.
        log_scale : bool, optional
            Whether the histogram is plotted on a logarithmic scale.
        """

        if len(bin_edges) != len(bin_counts) + 1:
            raise ValueError("bin_edges must have one more element than bin_counts.")
        self.bin_edges = np.array(bin_edges)
        self._label = label
        self._plot_color = plot_color
        self._tex_string = tex_string
        self.log_scale = log_scale

        if covariance_matrix is not None:
            self.cov_matrix = np.array(covariance_matrix)
            self.bin_counts = np.array(correlated_values(bin_counts, self.cov_matrix))
        elif uncertainties is not None:
            self.cov_matrix = np.diag(np.array(uncertainties) ** 2)
            self.bin_counts = unumpy.uarray(bin_counts, uncertainties)
        else:
            raise ValueError("Either uncertainties or covariance_matrix must be provided.")

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
    def bin_centers(self):
        if self.log_scale:
            return np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        else:
            return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def nominal_values(self):
        return unumpy.nominal_values(self.bin_counts)

    @property
    def std_devs(self):
        return unumpy.std_devs(self.bin_counts)

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

    def check_bin_edges(self, other):
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Histograms have different bin edges, cannot perform operation.")

    def fluctuate(self, seed=None):
        """Fluctuate bin counts according to uncertainties and return a new histogram with the fluctuated counts."""
        # take full correlation into account
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.multivariate_normal(unumpy.nominal_values(self.bin_counts), self.cov_matrix)
        return Histogram(self.bin_edges, fluctuated_bin_counts, covariance_matrix=self.cov_matrix)

    def _error_propagation_division(self, x1, x2, C1, C2):
        """
        Compute the result of element-wise division of x1 by x2 and the associated covariance matrix.

        Parameters
        ----------
        x1 : array_like
            First array to be divided.
        x2 : array_like
            Second array to divide by.
        C1 : array_like
            Covariance matrix of x1.
        C2 : array_like
            Covariance matrix of x2.

        Returns
        -------
        y : array_like
            Result of element-wise division of x1 by x2.
        Cy : array_like
            Covariance matrix of y.
        """

        # Element-wise division to get y
        y = x1 / x2

        n = len(x1)
        Cy = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements (variance)
                    Cy[i, i] = y[i] ** 2 * (C1[i, i] / x1[i] ** 2 + C2[i, i] / x2[i] ** 2)
                else:
                    # Off-diagonal elements (covariance)
                    Cy[i, j] = y[i] * y[j] * (C1[i, j] / (x1[i] * x1[j]) + C2[i, j] / (x2[i] * x2[j]))

        return y, Cy

    def __repr__(self):
        return f"Histogram(\nbin_edges: {self.bin_edges},\nbin_counts: {self.bin_counts},\ncovariance: {self.cov_matrix},\nlabel: {self.label}, tex: {self.tex_string})"

    def __add__(self, other):
        self.check_bin_edges(other)
        new_bin_counts = self.bin_counts + other.bin_counts
        label = self.label if self.label == other.label else "+".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        return Histogram(
            self.bin_edges,
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
        self.check_bin_edges(other)
        new_bin_counts = self.bin_counts - other.bin_counts
        label = self.label if self.label == other.label else "-".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        return Histogram(
            self.bin_edges,
            unumpy.nominal_values(new_bin_counts),
            covariance_matrix=new_cov_matrix,
            label=label,
            tex_string=self.tex_string,
        )

    def __truediv__(self, other):
        self.check_bin_edges(other)
        new_bin_counts, new_cov_matrix = self._error_propagation_division(
            self.nominal_values, other.nominal_values, self.cov_matrix, other.cov_matrix
        )
        label = self.label if self.label == other.label else "/".join([self.label, other.label])
        return Histogram(
            self.bin_edges,
            new_bin_counts,
            covariance_matrix=new_cov_matrix,
            label=label,
            tex_string=self.tex_string,
        )


class TestHistogram(unittest.TestCase):
    def test_uncorrelated(self):
        bin_edges = np.array([0, 1, 2, 3])
        bin_counts1 = np.array([1, 2, 3])
        uncertainties1 = np.array([0.1, 0.2, 0.3])
        bin_counts2 = np.array([4, 5, 6])
        uncertainties2 = np.array([0.4, 0.5, 0.6])

        hist1 = Histogram(bin_edges, bin_counts1, uncertainties1, label="hist1", tex_string="hist1")
        hist2 = Histogram(bin_edges, bin_counts2, uncertainties2, label="hist2", tex_string="hist2")

        hist_sum = hist1 + hist2
        hist_diff = hist1 - hist2

        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_sum.bin_counts), np.array([5, 7, 9]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_sum.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )
        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_diff.bin_counts), np.array([-3, -3, -3]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_diff.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )

    def test_correlated(self):
        bin_edges = np.array([0, 1, 2, 3])
        bin_counts1 = np.array([1, 2, 3])
        covariance_matrix1 = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        bin_counts2 = np.array([4, 5, 6])
        covariance_matrix2 = np.array([[0.16, 0.20, 0.24], [0.20, 0.25, 0.30], [0.24, 0.30, 0.36]])

        hist1 = Histogram(
            bin_edges, bin_counts1, covariance_matrix=covariance_matrix1, label="hist1", tex_string="hist1"
        )
        hist2 = Histogram(
            bin_edges, bin_counts2, covariance_matrix=covariance_matrix2, label="hist2", tex_string="hist2"
        )

        hist_sum = hist1 + hist2
        hist_diff = hist1 - hist2

        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_sum.bin_counts), np.array([5, 7, 9]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_sum.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )
        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_diff.bin_counts), np.array([-3, -3, -3]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_diff.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )

    def test_fluctuation(self):
        # Generate a histogram with a covariance matrix
        bin_edges = np.array([0, 1, 2, 3])
        bin_counts = np.array([1, 2, 3])
        covariance_matrix = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        hist = Histogram(bin_edges, bin_counts, covariance_matrix=covariance_matrix, label="hist", tex_string="hist")

        # fluctuate the histogram and check that the fluctuated bin counts are distributed according to the covariance matrix
        fluctuated_counts = []
        for i in range(10000):
            fluctuated_hist = hist.fluctuate(seed=i)
            fluctuated_counts.append(fluctuated_hist.nominal_values)
        fluctuated_counts = np.array(fluctuated_counts)

        # calculate covariance matrix of fluctuated counts with numpy
        cov_matrix = np.cov(fluctuated_counts, rowvar=False)

        # this should be close to the expectation value
        np.testing.assert_array_almost_equal(cov_matrix, covariance_matrix, decimal=2)

    def test_division(self):
        bin_edges = np.array([0, 1, 2, 3])
        # We want to test the division away from zero to avoid division by zero errors
        bin_counts1 = np.array([1, 2, 3]) + 10
        covariance_matrix1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]])
        # covariance_matrix1 = np.array([[0.1, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.9]])
        bin_counts2 = np.array([4, 5, 6]) + 10
        covariance_matrix2 = np.array([[0.16, 0.20, 0.24], [0.20, 0.25, 0.30], [0.24, 0.30, 0.36]])
        # covariance_matrix2 = np.array([[0.16, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.36]])

        hist1 = Histogram(
            bin_edges, bin_counts1, covariance_matrix=covariance_matrix1, label="hist1", tex_string="hist1"
        )
        hist2 = Histogram(
            bin_edges, bin_counts2, covariance_matrix=covariance_matrix2, label="hist2", tex_string="hist2"
        )

        fluctuated_divisions = []
        # To test error propagation, we fluctuate hist1 and hist2 and divide them. The covariance matrix
        # of the fluctuated divisions should be close to the expected covariance matrix that we get
        # from the division function.
        for i in range(10000):
            fluctuated_hist1 = hist1.fluctuate(seed=i)
            # It's important not to repeat seeds here, otherwise the values will be correlated
            # when they should not be.
            fluctuated_hist2 = hist2.fluctuate(seed=i + 10000)
            fluctuated_division = fluctuated_hist1 / fluctuated_hist2
            fluctuated_divisions.append(fluctuated_division.nominal_values)
        fluctuated_divisions = np.array(fluctuated_divisions)

        # calculate covariance matrix of fluctuated divisions with numpy
        cov_matrix = np.cov(fluctuated_divisions, rowvar=False)

        # get expectation histogram
        expected_div_hist = hist1 / hist2
        # check nominal values
        np.testing.assert_array_almost_equal(
            fluctuated_divisions.mean(axis=0), expected_div_hist.nominal_values, decimal=3
        )
        # check covariance matrix
        np.testing.assert_array_almost_equal(cov_matrix, expected_div_hist.cov_matrix, decimal=4)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
