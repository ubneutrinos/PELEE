from dataclasses import dataclass, fields
import hashlib
from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging

from numbers import Number
from uncertainties import correlated_values, unumpy
from . import selections
from .category_definitions import get_category_label, get_category_color
from .statistics import (
    covariance,
    sideband_constraint_correction,
    error_propagation_division,
    error_propagation_multiplication,
    chi_square,
)
from .parameters import Parameter, ParameterSet


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

    def __post_init__(self):
        if isinstance(self.bin_edges, list):
            self.bin_edges = np.array(self.bin_edges)

    def __len__(self):
        return self.n_bins

    @classmethod
    def from_config(cls, variable, n_bins, limits, label, is_log=False):
        """Create a Binning object from a typical binning configuration

        Parameters:
        -----------
        variable : str
            Name of the variable being binned
        n_bins : int
            Number of bins
        limits : tuple
            Tuple of lower and upper limits
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
            bin_edges = np.geomspace(*limits, n_bins + 1)
        else:
            bin_edges = np.linspace(*limits, n_bins + 1)
        return cls(variable, bin_edges, label, is_log=is_log)

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

    def draw_covariance_matrix(self, ax=None, as_correlation=True, **plot_kwargs):
        """Draw the covariance matrix on a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw the covariance matrix on.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to the matplotlib imshow function.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.set_title(f"{'Correlation' if as_correlation else 'Covariance'} matrix")
        X, Y = np.meshgrid(self.binning.bin_centers, self.binning.bin_centers)
        colormap = plot_kwargs.pop("cmap", "RdBu_r")
        if as_correlation:
            plot_kwargs["vmin"] = -1
            plot_kwargs["vmax"] = 1
            pc = ax.pcolormesh(X, Y, self.corr_matrix, cmap=colormap, **plot_kwargs)
        else:
            pc = ax.pcolormesh(X, Y, self.cov_matrix, cmap=colormap, **plot_kwargs)
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label("Correlation" if as_correlation else "Covariance")
        ax.set_xlabel(self.binning.label)
        ax.set_ylabel(self.binning.label)

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

    def copy(self):
        """Create a copy of the histogram.

        Returns
        -------
        Histogram
            Copy of the histogram.
        """

        state = self.to_dict()
        return Histogram.from_dict(state)

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
        return f"Histogram(binning={self.binning}, bin_counts={self.bin_counts}, label={self.label}, tex={self.tex_string})"

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            # We can add or subtract an ndarray to a Histogram as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot add ndarray to histogram: ndarray has wrong length.")
            new_bin_counts = self.nominal_values + other
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            return Histogram.from_dict(state)

        # otherwise, if other is also a Histogram
        assert self.binning == other.binning, (
            "Cannot add histograms with different binning. "
            f"self.binning = {self.binning}, other.binning = {other.binning}"
        )
        new_bin_counts = self.nominal_values + other.nominal_values
        label = self.label if self.label == other.label else "+".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return Histogram.from_dict(state)

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
            new_bin_counts = self.nominal_values - other
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            return Histogram.from_dict(state)
        # otherwise, if other is also a Histogram
        assert self.binning == other.binning, (
            "Cannot subtract histograms with different binning. "
            f"self.binning = {self.binning}, other.binning = {other.binning}"
        )
        new_bin_counts = self.nominal_values - other.nominal_values
        label = self.label if self.label == other.label else "-".join([self.label, other.label])
        new_cov_matrix = self.cov_matrix + other.cov_matrix
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return Histogram.from_dict(state)

    def __truediv__(self, other):
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values / other
            new_cov_matrix = self.cov_matrix / other**2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return Histogram.from_dict(state)
        elif isinstance(other, np.ndarray):
            # We can divide a histogram by an ndarray as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot divide histogram by ndarray: ndarray has wrong length.")
            with np.errstate(divide="ignore", invalid="ignore"):
                new_bin_counts = self.nominal_values / other
                # The covariance matrix also needs to be scaled according to
                # C_{ij}' = C_{ij} / a_i / a_j
                # where a_i is the ith element of the ndarray
                new_cov_matrix = self.cov_matrix / np.outer(other, other)
            # replace infs with zero
            new_cov_matrix[~np.isfinite(new_cov_matrix)] = 0
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return Histogram.from_dict(state)
        assert self.binning == other.binning, "Cannot divide histograms with different binning."
        new_bin_counts, new_cov_matrix = error_propagation_division(
            self.nominal_values, other.nominal_values, self.cov_matrix, other.cov_matrix
        )
        label = self.label if self.label == other.label else "/".join([self.label, other.label])
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return Histogram.from_dict(state)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            # We can multiply a histogram by an ndarray as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot multiply histogram by ndarray: ndarray has wrong length.")
            new_bin_counts = self.nominal_values * other
            # The covariance matrix also needs to be scaled according to
            # C_{ij}' = C_{ij} * a_i * a_j
            # where a_i is the ith element of the ndarray
            new_cov_matrix = self.cov_matrix * np.outer(other, other)
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return Histogram.from_dict(state)
        elif isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.cov_matrix * other**2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return Histogram.from_dict(state)
        elif isinstance(other, Histogram):
            assert self.binning == other.binning, "Cannot multiply histograms with different binning."
            new_bin_counts, new_cov_matrix = error_propagation_multiplication(
                self.nominal_values, other.nominal_values, self.cov_matrix, other.cov_matrix
            )
            label = self.label if self.label == other.label else "*".join([self.label, other.label])
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            state["label"] = label
            return Histogram.from_dict(state)
        else:
            raise NotImplementedError(f"Histogram multiplication not supprted for type {type(other)}")

    def __rmul__(self, other):
        # we only support multiplication by numbers that scale the entire histogram
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.cov_matrix * other**2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return Histogram.from_dict(state)
        else:
            raise NotImplementedError("Histogram multiplication is only supported for numeric types.")

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
        rundata_dict: Dict[str, pd.DataFrame],
        binning: Binning,
        weight_column: Optional[str] = "weights",
        selection: Optional[str] = None,
        preselection: Optional[str] = None,
        data_pot: Optional[float] = None,
        sideband_generator: Optional["RunHistGenerator"] = None,
        uncertainty_defaults: Optional[Dict[str, bool]] = None,
        parameters: Optional[ParameterSet] = None,
        mc_hist_generator_cls: Optional[type] = None,
        **mc_hist_generator_kwargs,
    ) -> None:
        """Create a histogram generator for data and simulation runs.

        This combines data and MC appropriately for the given run. It assumes also that,
        if truth-filtered samples are present, that the corresponding event types have
        already been removed from the 'mc' dataframe. It also assumes that the background sets
        have been scaled to the same POT as the data.

        Parameters
        ----------
        rundata_dict : Dict[str, pd.DataFrame]
            Dictionary containing the dataframes for this run. The keys are the names of the
            datasets and the values are the dataframes. This must at least contain the keys
            "data", "mc", and "ext". This dictionary should be returned by the data_loader.
        binning : Binning
            Binning object containing the binning of the histogram.
        weight_column : str, optional
            Name of the column containing the weights of the data points.
        selection : str, optional
            Query to be applied to the dataframe before generating the histogram.
        preselection : str, optional
            Query to be applied to the dataframe before the selection is applied.
        data_pot : float, optional
            POT of the data sample. Required to reweight to a different target POT.
        sideband_generator : RunHistGenerator, optional
            Histogram generator for the sideband data. If provided, the sideband data will be
            used to constrain multisim uncertainties.
        uncertainty_defaults : Dict[str, bool], optional
            Dictionary containing default configuration of the uncertainty calculation, i.e.
            whether to use the sideband, include multisim errors etc.
        parameters : ParameterSet, optional
            Set of parameters for the analysis. These parameters will be passed through to the
            histogram generator for MC. The default HistogramGenerator ignores all parameters.
            The parameters are passed through by reference, so any changes to the parameters
            will be reflected in the histogram generator automatically.
        mc_hist_generator_cls : type, optional
            Class to use for the MC histogram generator. If None, the default HistogramGenerator
            class is used.
        **mc_hist_generator_kwargs
            Additional keyword arguments that are passed to the MC histogram generator on initialization.
        """
        self.rundata_dict = rundata_dict
        self.data_pot = data_pot
        data_columns = rundata_dict["mc"].columns
        if weight_column is None:
            weight_column = "weights"
        query = self.get_selection_query(selection, preselection)
        self.selection = selection
        self.preselection = preselection
        super().__init__(data_columns, binning, weight_column=weight_column, query=query)

        # ensure that the necessary keys are present
        if "data" not in self.rundata_dict.keys():
            raise ValueError("data key is missing from rundata_dict.")
        if "mc" not in self.rundata_dict.keys():
            raise ValueError("mc key is missing from rundata_dict.")
        if "ext" not in self.rundata_dict.keys():
            raise ValueError("ext key is missing from rundata_dict.")

        for k, df in rundata_dict.items():
            if df is None:
                continue
            df["dataset_name"] = k
            df["dataset_name"] = df["dataset_name"].astype("category")
        mc_hist_generator_cls = HistogramGenerator if mc_hist_generator_cls is None else mc_hist_generator_cls
        # make one dataframe for all mc events
        df_mc = pd.concat([df for k, df in rundata_dict.items() if k not in ["data", "ext"]])
        df_ext = rundata_dict["ext"]
        df_data = rundata_dict["data"]
        self.parameters = parameters
        if self.parameters is None:
            self.parameters = ParameterSet([])  # empty parameter set
        else:
            assert isinstance(self.parameters, ParameterSet), "parameters must be a ParameterSet."
        self.mc_hist_generator = mc_hist_generator_cls(
            df_mc,
            binning,
            weight_column=weight_column,
            query=query,
            parameters=self.parameters,
            **mc_hist_generator_kwargs,
        )
        self.ext_hist_generator = HistogramGenerator(df_ext, binning, weight_column=weight_column, query=query)
        if df_data is not None:
            self.data_hist_generator = HistogramGenerator(df_data, binning, weight_column=weight_column, query=query)
        else:
            self.data_hist_generator = None
        self.sideband_generator = sideband_generator
        self.uncertainty_defaults = dict() if uncertainty_defaults is None else uncertainty_defaults

    def get_selection_query(self, selection, preselection, extra_queries=None):
        """Get the query for the given selection and preselection.

        Optionally, add any extra queries to the selection query. These will
        be joined with an 'and' operator.

        Parameters
        ----------
        selection : str
            Name of the selection category.
        preselection : str
            Name of the preselection category.
        extra_queries : list of str, optional
            List of additional queries to apply to the dataframe.

        Returns
        -------
        query : str
            Query to apply to the dataframe.
        """

        if selection is None and preselection is None:
            return None
        presel_query = selections.preselection_categories[preselection]["query"]
        sel_query = selections.selection_categories[selection]["query"]

        if presel_query is None:
            query = sel_query
        elif sel_query is None:
            query = presel_query
        else:
            query = f"{presel_query} and {sel_query}"

        if extra_queries is not None:
            for q in extra_queries:
                query = f"{query} and {q}"
        return query

    def get_data_hist(self, type="data", add_error_floor=None, scale_to_pot=None):
        """Get the histogram for the data (or EXT).

        Parameters
        ----------
        type : str, optional
            Type of data to return. Can be "data" or "ext".
        add_error_floor : bool, optional
            Add a minimum error of 1.4 to empty bins. This is motivated by a Bayesian
            prior of a unit step function as documented in
            https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=32714&filename=Monte_Carlo_Uncertainties.pdf&version=1
            and is currently only applied to EXT events.
        scale_to_pot : float, optional
            POT to scale the data to. Only applicable to EXT data.

        Returns
        -------
        data_hist : numpy.ndarray
            Histogram of the data.
        """

        assert type in ["data", "ext"]
        add_error_floor = (
            self.uncertainty_defaults.get("add_ext_error_floor", False) if add_error_floor is None else add_error_floor
        )
        # The error floor is never added for data, overriding anything else
        if type == "data":
            add_error_floor = False
        scale_factor = 1.0
        if scale_to_pot is not None:
            if type == "data":
                raise ValueError("Cannot scale data to POT.")
            assert self.data_pot is not None, "Must provide data POT to scale EXT data."
            scale_factor = scale_to_pot / self.data_pot
        hist_generator = self.get_hist_generator(which=type)
        if hist_generator is None:
            return None
        data_hist = hist_generator.generate()
        if add_error_floor:
            prior_errors = np.ones(data_hist.n_bins) * 1.4**2
            prior_errors[data_hist.nominal_values > 0] = 0
            data_hist.add_covariance(np.diag(prior_errors))
        data_hist *= scale_factor
        data_hist.label = {"data": "Data", "ext": "EXT"}[type]
        data_hist.color = {"data": "k", "ext": "yellow"}[type]
        return data_hist

    def get_mc_hists(self, category_column="dataset_name", include_multisim_errors=None, scale_to_pot=None):
        """Get MC histograms that are split by event category.

        Parameters
        ----------
        category_column : str, optional
            Name of the column containing the event categories.
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
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
        include_multisim_errors = (
            self.uncertainty_defaults.get("include_multisim_errors", False)
            if include_multisim_errors is None
            else include_multisim_errors
        )
        mc_hists = {}
        other_categories = []
        for category in self.mc_hist_generator.dataframe[category_column].unique():
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

    def get_hist_generator(self, which):
        assert which in ["mc", "data", "ext"]
        hist_generator = {
            "mc": self.mc_hist_generator,
            "data": self.data_hist_generator,
            "ext": self.ext_hist_generator,
        }[which]
        return hist_generator

    def get_mc_hist(self, include_multisim_errors=None, extra_query=None, scale_to_pot=None, use_sideband=None):
        """Produce a histogram from the MC dataframe.

        Parameters
        ----------
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
            Overrides the default setting.
        """

        scale_factor = 1.0
        if scale_to_pot is not None:
            assert self.data_pot is not None, "data_pot must be set to scale to a different POT."
            scale_factor = scale_to_pot / self.data_pot
        include_multisim_errors = (
            self.uncertainty_defaults.get("include_multisim_errors", False)
            if include_multisim_errors is None
            else include_multisim_errors
        )
        use_sideband = self.uncertainty_defaults.get("use_sideband", False) if use_sideband is None else use_sideband
        hist_generator = self.get_hist_generator(which="mc")
        use_sideband = use_sideband and self.sideband_generator is not None
        if use_sideband:
            sideband_generator = self.sideband_generator.get_hist_generator(which="mc")
            sideband_total_prediction = self.sideband_generator.get_total_prediction(include_multisim_errors=True)
            sideband_observed_hist = self.sideband_generator.get_data_hist(type="data")
            if sideband_observed_hist is None:
                raise RuntimeError(
                    "The sideband generator contains no data. Make sure to set `blinded=False` when loading the sideband data."
                )
        else:
            sideband_generator = None
            sideband_total_prediction = None
            sideband_observed_hist = None
        hist = hist_generator.generate(
            include_multisim_errors=include_multisim_errors,
            use_sideband=use_sideband,
            sideband_generator=sideband_generator,
            sideband_total_prediction=sideband_total_prediction,
            sideband_observed_hist=sideband_observed_hist,
            extra_query=extra_query,
        )
        hist.label = "MC"

        hist *= scale_factor
        return hist

    def get_total_prediction(
        self,
        include_multisim_errors=None,
        extra_query=None,
        scale_to_pot=None,
        use_sideband=None,
        add_ext_error_floor=None,
    ):
        """Get the total prediction from MC and EXT.

        Parameters
        ----------
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
            Overrides the default setting.
        add_ext_error_floor : bool, optional
            Whether to add an error floor to the histogram in bins with zero entries.
        """
        mc_prediction = self.get_mc_hist(
            include_multisim_errors=include_multisim_errors,
            extra_query=extra_query,
            scale_to_pot=scale_to_pot,
            use_sideband=use_sideband,
        )
        ext_prediction = self.get_data_hist(type="ext", scale_to_pot=scale_to_pot, add_error_floor=add_ext_error_floor)
        total_prediction = mc_prediction + ext_prediction
        return total_prediction

    def get_chi_square(self):
        """Get the chi square between the data and the total prediction.

        Returns
        -------
        chi_square : float
            Chi square between the data and the total prediction.
        """
        data_hist = self.get_data_hist(type="data")
        total_prediction = self.get_total_prediction()
        chi_sq = chi_square(data_hist.nominal_values, total_prediction.nominal_values, total_prediction.cov_matrix)
        return chi_sq


class HistogramGenerator(HistGenMixin):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        binning: Binning,
        weight_column: str = "weights",
        query: Optional[str] = None,
        parameters: Optional[ParameterSet] = None,
        enable_cache: bool = True,
        cache_total_covariance: bool = True,
    ):
        """Create a histogram generator for a given dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the data to be binned.
        binning : Binning
            Binning object containing the binning of the histogram.
        weight_column : str or list of str, optional
            Name of the column containing the weights of the data points. If more than one weight
            column is given, the weights are multiplied in sequence.
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.
        parameters : ParameterSet, optional
            Set of parameters for the analysis that are used to weight or otherwise manipulate the
            histograms. The default HistogramGenerator ignores all parameters, but sub-classes can
            override this behavior. When this class is used inside a RunHistGenerator, the
            parameters are passed through by reference, so any changes to the parameters of the
            RunHistGenerator will be reflected in the histogram generator automatically.
        enable_cache : bool, optional
            Whether to enable caching of histograms. The cache stores histograms in a dictionary
            where the keys are the hash of the query and the values are the histograms. The cache is
            invalidated if the parameters change. The cache also stores the histograms of the
            different multisim universes and the unisim histograms. Do not change the dataframe
            after creating the HistogramGenerator with this setting enabled.
        cache_total_covariance : bool, optional
            If True, the total covariance matrix is cached. This is only used if enable_cache is
            True. This skips the entire calculation of the sideband correction. Use with caution:
            This assumes that there are no parameters that affect only the sideband (which is the
            case for the LEE analysis, hence the default is True). If you are using this class in an
            analysis where you have a parameter that is _only_ affecting the sideband, this needs to
            be set to False. If the parameter affects both (for instance, an overall spectral index
            correction), it is fine to leave it as True because a recalculation of the sideband will
            also trigger a recalculation of this histogram.
        """
        self.dataframe = dataframe
        data_columns = dataframe.columns
        self.parameters = parameters
        if self.parameters is None:
            self.parameters = ParameterSet([])  # empty parameter set, still needed to check cache
        self.parameters_last_evaluated = None  # used to invalidate the cache
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Creating histogram generator for with binning: {binning}")
        self.enable_cache = enable_cache
        self.cache_total_covariance = cache_total_covariance
        self._invalidate_cache()
        super().__init__(data_columns, binning, weight_column=weight_column, query=query)

    def _invalidate_cache(self):
        """Invalidate the cache."""
        self.hist_cache = dict()
        # we keep caches around for every combination of multisim and sideband (even though
        #  the combination of without mulitim and with sideband is kinda nonsense and not used)
        self.hist_cache["without_multisim"] = {"with_sideband": dict(), "without_sideband": dict()}
        self.hist_cache["with_multisim"] = {"with_sideband": dict(), "without_sideband": dict()}
        self.unisim_hist_cache = dict()
        self.multisim_hist_cache = dict()
        self.multisim_hist_cache["weightsReint"] = dict()
        self.multisim_hist_cache["weightsFlux"] = dict()
        self.multisim_hist_cache["weightsGenie"] = dict()
        self.parameters_last_evaluated = self.parameters.copy()

    def _return_empty_hist(self):
        """Return an empty histogram."""
        return Histogram(self.binning, np.zeros(self.binning.n_bins), np.zeros(self.binning.n_bins))

    def generate(
        self,
        query=None,
        include_multisim_errors=False,
        use_sideband=False,
        extra_query=None,
        sideband_generator=None,
        sideband_total_prediction=None,
        sideband_observed_hist=None,
    ):
        """Generate a histogram from the dataframe.

        Parameters
        ----------
        query : str, optional
            Query to be applied to the dataframe before generating the histogram.

        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
            Overrides the default setting.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        sideband_generator : HistogramGenerator, optional
            Histogram generator for the sideband data. If provided, the sideband data will be
            used to constrain multisim uncertainties.
        sideband_total_prediction : Histogram, optional
            Histogram containing the total prediction in the sideband. If provided, the sideband
            data will be used to constrain multisim uncertainties.
        sideband_observed_hist : Histogram, optional
            Histogram containing the observed data in the sideband. If provided, the sideband
            data will be used to constrain multisim uncertainties.

        Returns
        -------
        histogram : Histogram
            Histogram object containing the binned data.
        """
        self.update_settings(variable=None, binning=None, weight_column=None, query=query)

        if use_sideband:
            assert sideband_generator is not None
            assert sideband_total_prediction is not None
            assert sideband_observed_hist is not None

        query = self._get_query(extra_query=extra_query)

        calculate_hist = True

        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self.logger.debug("Parameters changed, invalidating cache.")
                self._invalidate_cache()
            if query is None:
                hash = "None"
            else:
                hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            hist_cache = self.hist_cache["with_multisim" if include_multisim_errors else "without_multisim"][
                "with_sideband" if use_sideband else "without_sideband"
            ]
            if hash in hist_cache:
                self.logger.debug("Histogram found in cache.")
                if self.cache_total_covariance:
                    self.logger.debug("Using cached total covariance matrix.")
                    return hist_cache[hash].copy()
                hist = hist_cache[hash].copy()
                calculate_hist = False
        if calculate_hist:
            if query is not None:
                dataframe = self.dataframe.query(query, engine="python")
                if len(dataframe) == 0:
                    self.logger.debug("Query returned no events, returning empty histogram.")
                    hist = self._return_empty_hist()
                    if self.enable_cache:
                        hist_cache[hash] = hist.copy()
                    return hist
            else:
                dataframe = self.dataframe
            self.logger.debug(f"Generating histogram with query: {query}")
            self.logger.debug(f"Total number of events after filtering: {len(dataframe)}")
            weights = self.get_weights(weight_column=self.weight_column, query=query)
            bin_counts, bin_edges = np.histogram(
                dataframe[self.variable], bins=self.binning.bin_edges, weights=weights
            )
            variances, _ = np.histogram(dataframe[self.variable], bins=self.binning.bin_edges, weights=weights**2)
            hist = Histogram(self.binning, bin_counts, uncertainties=np.sqrt(variances))
            if self.enable_cache:
                hist_cache[hash] = hist.copy()
        self.logger.debug(f"Generated histogram: {hist}")
        if include_multisim_errors:
            self.logger.debug("Calculating multisim uncertainties")
            if use_sideband:
                # initialize extended covariance matrix
                n_bins = hist.n_bins
                sb_n_bins = sideband_observed_hist.n_bins
                extended_cov = np.zeros((n_bins + sb_n_bins, n_bins + sb_n_bins))

            # calculate multisim histograms
            for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
                cov_mat, universe_hists = self.calculate_multisim_uncertainties(
                    ms_column,
                    extra_query=extra_query,
                    return_histograms=True,
                )
                hist.add_covariance(cov_mat)

                if use_sideband:
                    extended_cov += self.multiband_covariance([self, sideband_generator], ms_column)

            # calculate unisim histograms
            self.logger.debug("Calculating unisim uncertainties")
            cov_mat_unisim = self.calculate_unisim_uncertainties(central_value_hist=hist, extra_query=extra_query)
            hist.add_covariance(cov_mat_unisim)

            if use_sideband:
                # calculate constraint correction
                mu_offset, cov_corr = sideband_constraint_correction(
                    sideband_measurement=sideband_observed_hist.nominal_values,
                    sideband_central_value=sideband_total_prediction.nominal_values,
                    concat_covariance=extended_cov,
                    sideband_covariance=sideband_total_prediction.cov_matrix,
                )
                self.logger.debug(f"Sideband constraint correction: {mu_offset}")
                self.logger.debug(f"Sideband constraint covariance correction: {cov_corr}")
                # add corrections to histogram
                hist += mu_offset
                hist.add_covariance(cov_corr)
        if self.enable_cache and self.cache_total_covariance:
            hist_cache[hash] = hist.copy()
        return hist

    @classmethod
    def multiband_covariance(cls, hist_generators, ms_column):
        """Calculate the covariance matrix for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the covariance matrix of the
        multisim universes for the given multisim weight column. The underlying assumption
        is that the weights listed in the multisim column are from the same universes
        in the same order for all histograms.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        ms_column : str
            Name of the multisim weight column.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        # types = [type(hg) for hg in hist_generators]
        # assert all(isinstance(hg, cls) for hg in hist_generators), f"Must provide a list of HistogramGenerator objects. Types are {types}."

        universe_hists = []
        central_values = []
        for hg in hist_generators:
            cov_mat, universe_hist = hg.calculate_multisim_uncertainties(ms_column, return_histograms=True)
            universe_hists.append(universe_hist)
            central_values.append(hg.generate().nominal_values)

        concatenated_cv = np.concatenate(central_values)
        concatenated_universes = np.concatenate(universe_hists, axis=1)
        cov_mat = covariance(concatenated_universes, concatenated_cv)
        return cov_mat

    @classmethod
    def multiband_unisim_covariance(cls, hist_generators):
        """Calculate the covariance matrix for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the covariance matrix of the
        unisim universes. The underlying assumption
        is that the weights listed in the unisim column are from the same universes
        in the same order for all histograms.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        universe_hist_dicts = []
        concatenated_cv = []
        for hg in hist_generators:
            concatenated_cv.append(hg.generate().nominal_values)
            universe_hist_dicts.append(hg.calculate_unisim_uncertainties(return_histograms=True)[1])
        concatenated_cv = np.concatenate(concatenated_cv)
        knobs = list(universe_hist_dicts[0].keys())
        summed_cov_mat = np.zeros((len(concatenated_cv), len(concatenated_cv)))
        for knob in knobs:
            assert all(
                knob in hist_dict for hist_dict in universe_hist_dicts
            ), f"Knob {knob} not found in all histograms."
            concatenated_universes = np.concatenate([hist_dict[knob] for hist_dict in universe_hist_dicts], axis=1)
            cov_mat = covariance(concatenated_universes, concatenated_cv, allow_approximation=True, tolerance=1e-10)
            summed_cov_mat += cov_mat
        return summed_cov_mat

    def adjust_weights(self, dataframe, base_weights):
        """Reweight events according to the parameters.

        This method is intended to be overridden by subclasses. The default implementation
        does not change the weights. Subclasses may safely assume that the dataframe
        has already been filtered for the given query and has the same length as the
        base_weights array.
        """

        return base_weights

    def get_weights(self, weight_column=None, limit_weight=True, query=None):
        """Get the weights of the dataframe after filtering for the given query.

        Parameters
        ----------
        weight_column : str or list of str, optional
            Override the weight column given at initialization.
        limit_weight : bool, optional
            Reset invalid weights to one.
        query : str, optional
            Query to apply to the dataframe before calculating the weights.

        Returns
        -------
        weights : array_like
            Array of weights

        Notes
        -----
        This method is _not_ intended to be overridden by subclasses. Instead, subclasses
        should override the adjust_weights() method if they wish to change the weights
        according to the parameters.
        """
        if query is not None:
            dataframe = self.dataframe.query(query, engine="python")
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
        return self.adjust_weights(dataframe, weights)

    def _limit_weights(self, weights):
        weights = np.asarray(weights)
        weights[weights > 100] = 1.0
        weights[weights < 0] = 1.0
        weights[~np.isfinite(weights)] = 1.0
        if np.sum(~np.isfinite(weights)) > 0:
            self.logger.debug(f"Found {np.sum(~np.isfinite(weights))} invalid weights (set to one).")
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
            the central value is produced by calling the generate() method.
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
        assert multisim_weight_column in [
            "weightsGenie",
            "weightsFlux",
            "weightsReint",
        ], "Unknown multisim weight column."
        if weight_column is None:
            weight_column = "weights_no_tune" if multisim_weight_column == "weightsGenie" else "weights"
        query = self._get_query(extra_query)
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self._invalidate_cache()
            if query is None:
                hash = "None"
            else:
                hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            if hash in self.multisim_hist_cache[multisim_weight_column]:
                self.logger.debug(f"Multisim histogram found in cache.")
                if return_histograms:
                    return self.multisim_hist_cache[multisim_weight_column][hash]
                else:
                    return self.multisim_hist_cache[multisim_weight_column][hash][0]
        dataframe = self.dataframe.query(query, engine="python")
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
            universe_weights = self._limit_weights(df[column].values * weight_rescale)
            bincounts, _ = np.histogram(
                dataframe[self.variable],
                bins=self.binning.bin_edges,
                weights=base_weights * universe_weights,
            )
            universe_histograms.append(bincounts)
        universe_histograms = np.array(universe_histograms)
        if central_value_hist is None:
            central_value_hist = self.generate(query=query)
        # calculate the covariance matrix from the histograms
        cov = covariance(universe_histograms, central_value_hist.nominal_values)
        self.logger.debug(f"Calculated covariance matrix for {multisim_weight_column}.")
        self.logger.debug(f"Bin-wise error contribution: {np.sqrt(np.diag(cov))}")
        if self.enable_cache:
            self.multisim_hist_cache[multisim_weight_column][hash] = (cov, universe_histograms)
        if return_histograms:
            return cov, universe_histograms
        return cov

    def calculate_unisim_uncertainties(self, central_value_hist=None, extra_query=None, return_histograms=False):
        """Calculate unisim uncertainties.

        Unisim means that a single variation of a given analysis input parameter is performed according to its uncertainty.
        The difference in the number of selected events between this variation and the central value is taken as the
        uncertainty in that number of events. Mathematically, this is the same as the 'multisim' method, but with only
        one or two universes. The central value is in this case not optional.

        Parameters
        ----------
        central_value_hist : Histogram, optional
            Central value histogram.
        extra_query : str, optional
            Extra query to apply to the dataframe before calculating the covariance matrix.
        return_histograms : bool, optional
            If True, return the histograms of the universes.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.
        """

        query = self._get_query(extra_query)
        if query is None:
            hash = "None"
        else:
            hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self._invalidate_cache()
        knob_v = ["knobRPA", "knobCCMEC", "knobAxFFCCQE", "knobVecFFCCQE", "knobDecayAngMEC", "knobThetaDelta2Npi"]
        # see table 23 from the technote
        knob_n_universes = [2, 1, 1, 1, 1, 1]
        # because all of these are GENIE knobs, we need to use the weights without the GENIE tune just as
        # for the GENIE multisim
        base_weight = "weights_no_tune"
        # When we have two universes, then there are two weight variations, knobXXXup and knobXXXdown. Otherwise, there
        # is only one weight variation, knobXXXup.
        total_cov = np.zeros((len(self.binning), len(self.binning)))
        base_weights = self.get_weights(weight_column=base_weight, query=query)
        dataframe = self.dataframe.query(query, engine="python")
        if central_value_hist is None:
            central_value_hist = self.generate(query=query)
        observation_dict = dict()
        for knob, n_universes in zip(knob_v, knob_n_universes):
            observations = []
            if self.enable_cache and (hash not in self.unisim_hist_cache):
                self.unisim_hist_cache[hash] = dict()
            if self.enable_cache and knob in self.unisim_hist_cache[hash]:
                self.logger.debug(f"Unisim histogram for knob {knob} found in cache.")
                observations = self.unisim_hist_cache[hash][knob]
            else:
                for universe in range(n_universes):
                    # get the weight column for this universe
                    weight_column_knob = f"{knob}up" if n_universes == 2 and universe == 0 else f"{knob}dn"
                    # it is important to use the raw weights here that are not manipulated when this
                    # class is sub-classed and the get_weights function does some re-weighting.
                    universe_weights = self._limit_weights(dataframe.query(query, engine="python")[weight_column_knob])
                    # calculate the histogram for this universe
                    bincounts, _ = np.histogram(
                        dataframe[self.variable],
                        bins=self.binning.bin_edges,
                        weights=base_weights * universe_weights,
                    )
                    self.logger.debug(
                        f"Calculated histogram for {knob}up, universe {universe} with bin counts:\n{bincounts}."
                    )
                    if not np.all(np.isfinite(bincounts)):
                        raise ValueError(
                            f"Not all bin counts are finite for {knob}up, universe {universe}. Bin counts are: {bincounts}."
                        )
                    observations.append(bincounts)
                observations = np.array(observations)
                if self.enable_cache:
                    self.unisim_hist_cache[hash][knob] = observations
            observation_dict[knob] = observations
            # calculate the covariance matrix from the histograms
            cov = covariance(
                observations,
                central_value_hist.nominal_values,
                allow_approximation=True,
                debug_name=knob,
                tolerance=1e-10,
            )
            self.logger.debug(f"Bin-wise error contribution for knob {knob}: {np.sqrt(np.diag(cov))}")
            # add it to the total covariance matrix
            total_cov += cov
        if return_histograms:
            return total_cov, observation_dict
        return total_cov

    def _resync_parameters(self):
        """Not needed since there are no parameters in this class."""
        pass

    def _check_shared_parameters(self, parameters):
        return True



