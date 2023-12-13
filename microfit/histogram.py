from dataclasses import dataclass, fields, field
import hashlib
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging

from numbers import Number
from uncertainties import correlated_values, unumpy

from microfit.fileio import from_json
from . import selections
from .category_definitions import get_category_label, get_category_color
from .statistics import (
    covariance,
    sideband_constraint_correction,
    error_propagation_division,
    error_propagation_multiplication,
    chi_square,
    is_psd,
    fronebius_nearest_psd,
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
    selection_query : str, optional
        Query to be applied to the dataframe before generating the histogram.
    """

    variable: str
    bin_edges: np.ndarray
    label: Optional[str] = None
    is_log: bool = False
    selection_query: Optional[str] = None

    def __eq__(self, other):
        for field in fields(self):
            attr_self = getattr(self, field.name)
            attr_other = getattr(other, field.name)
            if field.name == "label":
                # There may be situations where a label is undefined (for instance, when
                # loading detector systematics). In this case, we don't want to compare
                # the labels.
                if attr_self is None or attr_other is None:
                    continue
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
    def from_config(cls, variable, n_bins, limits, label=None, is_log=False):
        """Create a Binning object from a typical binning configuration

        Parameters:
        -----------
        variable : str
            Name of the variable being binned
        n_bins : int
            Number of bins
        limits : tuple
            Tuple of lower and upper limits
        label : str, optional
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

    def to_dict(self):
        """Return a dictionary representation of the binning."""
        return self.__dict__

    @classmethod
    def from_dict(cls, state):
        """Create a Binning object from a dictionary representation of the binning."""
        return cls(**state)


class Histogram:
    def __init__(
        self,
        binning: Binning,
        bin_counts: np.ndarray,
        uncertainties=None,
        covariance_matrix=None,
        label=None,
        plot_color=None,
        plot_hatch=None,
        tex_string=None,
        check_psd=True,
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
        
        Notes
        -----
        The histograms defined by this class are strictly 1-dimensional. However, the binning
        may be one that has been un-rolled from a multi-channel binning.
        """

        self.binning = binning
        self.bin_counts = bin_counts
        assert self.binning.n_bins == len(
            self.bin_counts
        ), "bin_counts must have the same length as binning."
        self._label = label
        self._plot_color = plot_color
        self._plot_hatch = plot_hatch
        self._tex_string = tex_string

        if covariance_matrix is not None:
            self.covariance_matrix = np.array(covariance_matrix)
            if check_psd:
                if not is_psd(self.covariance_matrix, ignore_zeros=True):
                    raise ValueError(
                        "Non-zero part of covariance matrix must be positive semi-definite."
                    )
            std_devs = np.sqrt(np.diag(self.covariance_matrix))
            self.bin_counts = unumpy.uarray(bin_counts, std_devs)
        elif uncertainties is not None:
            # assert that uncertainties are 1D
            assert len(uncertainties.shape) == 1, "uncertainties must be 1-dimensional."
            self.covariance_matrix = np.diag(np.array(uncertainties) ** 2)
            self.bin_counts = unumpy.uarray(bin_counts, uncertainties)
        else:
            raise ValueError("Either uncertainties or covariance_matrix must be provided.")

    def draw_covariance_matrix(
        self, ax=None, as_correlation=True, as_fractional=False, **plot_kwargs
    ):
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
        label = None
        if as_correlation:
            plot_kwargs["vmin"] = -1
            plot_kwargs["vmax"] = 1
            pc = ax.pcolormesh(X, Y, self.correlation_matrix, cmap=colormap, **plot_kwargs)
            label = "Correlation"
        elif as_fractional:
            # plot fractional covariance matrix
            fractional_covar = self.covariance_matrix / np.outer(
                self.nominal_values, self.nominal_values
            )
            max_val = np.max(np.abs(fractional_covar))
            plot_kwargs["vmin"] = -max_val
            plot_kwargs["vmax"] = max_val
            pc = ax.pcolormesh(X, Y, fractional_covar, cmap=colormap, **plot_kwargs)
            label = "Fractional covariance"
        else:
            pc = ax.pcolormesh(X, Y, self.covariance_matrix, cmap=colormap, **plot_kwargs)
            label = "Covariance"
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel(self.binning.label)
        ax.set_ylabel(self.binning.label)

    def draw(self, ax, as_errorbars=False, show_errors=True, **plot_kwargs):
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
        hatch = plot_kwargs.pop("hatch", self.hatch)
        if as_errorbars:
            ax.errorbar(
                self.binning.bin_centers,
                bin_counts,
                yerr=self.std_devs,
                linestyle="none",
                marker=".",
                label=label,
                color=color,
                hatch=hatch,
                **plot_kwargs,
            )
            return ax
        errband_alpha = plot_kwargs.pop("alpha", 0.5)
        # Be sure to repeat the last bin count
        bin_counts = np.append(bin_counts, bin_counts[-1])
        p = ax.step(bin_edges, bin_counts, where="post", label=label, color=color, **plot_kwargs)
        if not show_errors:
            return ax
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
        return ax

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
            "covariance_matrix": self.covariance_matrix,
            "label": self._label,
            "plot_color": self._plot_color,
            "plot_hatch": self._plot_hatch,
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
        # When we make a copy, we trust that this has already been checked.
        # The check is a somewhat expensive operation. We make this optimization here
        # because the caching mechanism in the HistogramGenerator would be bottlenecked
        # by the expense of the initialization function otherwise.
        return self.__class__.from_dict(state, check_psd=False)

    @classmethod
    def from_dict(cls, dictionary, check_psd=True):
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
        dictionary["check_psd"] = check_psd
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
            and np.allclose(self.covariance_matrix, other.covariance_matrix)
            and self.label == other.label
            and self.color == other.color
            and self.tex_string == other.tex_string
        )

    @property
    def bin_centers(self):
        return self.binning.bin_centers

    @property
    def n_bins(self):
        return self.binning.n_bins

    @property
    def color(self):
        # We let the plotter handle the case when this is None, in which case
        # it will assign a color automatically.
        return self._plot_color

    @color.setter
    def color(self, value):
        self._plot_color = value

    @property
    def hatch(self):
        return self._plot_hatch

    @hatch.setter
    def hatch(self, value):
        self._plot_hatch = value

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
    def covariance_matrix(self):
        return self._covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(self, value):
        self._covariance_matrix = value
        # also update the bin counts
        self.bin_counts = unumpy.uarray(self.nominal_values, np.sqrt(np.diag(value)))

    @property
    def correlation_matrix(self):
        # convert the covariance matrix into a correlation matrix
        # ignore division by zero error
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.covariance_matrix / np.outer(self.std_devs, self.std_devs)

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

        self.covariance_matrix += cov_mat
        self.bin_counts = np.array(correlated_values(self.nominal_values, self.covariance_matrix))

    def fluctuate(self, seed=None):
        """Fluctuate bin counts according to uncertainties and return a new histogram with the fluctuated counts."""
        # take full correlation into account
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.multivariate_normal(
            unumpy.nominal_values(self.bin_counts), self.covariance_matrix
        )
        # clip bin counts from below
        fluctuated_bin_counts[fluctuated_bin_counts < 0] = 0
        return self.__class__(
            self.binning, fluctuated_bin_counts, covariance_matrix=self.covariance_matrix,
        )

    def fluctuate_poisson(self, seed=None):
        """Fluctuate bin counts according to Poisson uncertainties and return a new histogram with the fluctuated counts."""
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.poisson(unumpy.nominal_values(self.bin_counts))
        return self.__class__(
            self.binning, fluctuated_bin_counts, uncertainties=np.sqrt(fluctuated_bin_counts),
        )

    def __repr__(self):
        return f"Histogram(binning={self.binning}, bin_counts={self.bin_counts}, label={self.label}, tex={self.tex_string})"

    def __abs__(self):
        return np.abs(self.nominal_values)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            # We can add or subtract an ndarray to a Histogram as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot add ndarray to histogram: ndarray has wrong length.")
            new_bin_counts = self.nominal_values + other
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            return self.__class__.from_dict(state)

        # otherwise, if other is also a Histogram
        assert self.binning == other.binning, (
            "Cannot add histograms with different binning. "
            f"self.binning = {self.binning}, other.binning = {other.binning}"
        )
        new_bin_counts = self.nominal_values + other.nominal_values
        label = self.label if self.label == other.label else "+".join([self.label, other.label])
        new_cov_matrix = self.covariance_matrix + other.covariance_matrix
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return self.__class__.from_dict(state)

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
            return self.__class__.from_dict(state)
        # otherwise, if other is also a Histogram
        assert self.binning == other.binning, (
            "Cannot subtract histograms with different binning. "
            f"self.binning = {self.binning}, other.binning = {other.binning}"
        )
        new_bin_counts = self.nominal_values - other.nominal_values
        label = self.label if self.label == other.label else "-".join([self.label, other.label])
        new_cov_matrix = self.covariance_matrix + other.covariance_matrix
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return self.__class__.from_dict(state)

    def __truediv__(self, other):
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values / other
            new_cov_matrix = self.covariance_matrix / other ** 2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        elif isinstance(other, np.ndarray):
            # We can divide a histogram by an ndarray as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot divide histogram by ndarray: ndarray has wrong length.")
            with np.errstate(divide="ignore", invalid="ignore"):
                new_bin_counts = self.nominal_values / other
                # The covariance matrix also needs to be scaled according to
                # C_{ij}' = C_{ij} / a_i / a_j
                # where a_i is the ith element of the ndarray
                new_cov_matrix = self.covariance_matrix / np.outer(other, other)
            # replace infs with zero
            new_cov_matrix[~np.isfinite(new_cov_matrix)] = 0
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        assert self.binning == other.binning, "Cannot divide histograms with different binning."
        new_bin_counts, new_cov_matrix = error_propagation_division(
            self.nominal_values,
            other.nominal_values,
            self.covariance_matrix,
            other.covariance_matrix,
        )
        label = self.label if self.label == other.label else "/".join([self.label, other.label])
        state = self.to_dict()
        state["bin_counts"] = new_bin_counts
        state["covariance_matrix"] = new_cov_matrix
        state["label"] = label
        return self.__class__.from_dict(state)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            # We can multiply a histogram by an ndarray as long as the length of the ndarray matches the number of bins in the histogram.
            if len(other) != len(self.bin_counts):
                raise ValueError("Cannot multiply histogram by ndarray: ndarray has wrong length.")
            new_bin_counts = self.nominal_values * other
            # The covariance matrix also needs to be scaled according to
            # C_{ij}' = C_{ij} * a_i * a_j
            # where a_i is the ith element of the ndarray
            new_cov_matrix = self.covariance_matrix * np.outer(other, other)
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        elif isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.covariance_matrix * other ** 2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        elif isinstance(other, Histogram):
            assert (
                self.binning == other.binning
            ), "Cannot multiply histograms with different binning."
            new_bin_counts, new_cov_matrix = error_propagation_multiplication(
                self.nominal_values,
                other.nominal_values,
                self.covariance_matrix,
                other.covariance_matrix,
            )
            label = self.label if self.label == other.label else "*".join([self.label, other.label])
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            state["label"] = label
            return self.__class__.from_dict(state)
        else:
            raise NotImplementedError(
                f"Histogram multiplication not supprted for type {type(other)}"
            )

    def __rmul__(self, other):
        # we only support multiplication by numbers that scale the entire histogram
        if isinstance(other, Number):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.covariance_matrix * other ** 2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        else:
            raise NotImplementedError(
                "Histogram multiplication is only supported for numeric types."
            )


@dataclass
class MultiChannelBinning:
    """Binning for multiple channels.

    This can be used to define multi-channel binnings of the same data as well as
    binnings for different selections. For every channel, we may have a different 
    Binning that may have its own selection query. Using this binning definition,
    the HistogramGenerator can make multi-channel histograms that may include 
    correlations between bins and channels.

    Note that this is different from a multi-dimensional binning. The binnings
    defined in this class are still 1-dimensional, but the HistogramGenerator
    can create an unrolled 1-dimensional histogram from all the channels that
    includes correlations between bins of different channels.
    """

    binnings: List[Binning]
    is_log: bool = False
    __iter_index: int = field(default=0, init=False, repr=False)

    @property
    def labels(self) -> List[str]:
        """Labels of all channels."""
        return [b.label for b in self.binnings]

    @property
    def n_channels(self):
        """Number of channels."""
        return len(self.binnings)

    @property
    def n_bins(self):
        """Number of bins in all channels."""
        return sum([len(b) for b in self.binnings])

    @property
    def consecutive_bin_edges(self) -> List[np.ndarray]:
        """Get the bin edges of all channels.

        The bin edges of all channels are concatenated
        into a single array, which can be passed to np.histogramdd as follows:
        >>> bin_edges = consecutive_bin_edges
        >>> hist, _ = np.histogramdd(data, bins=bin_edges)
        
        Returns
        -------
        List[np.ndarray]
            List of arrays of bin edges of all channels.
        """
        return [b.bin_edges for b in self.binnings]

    @property
    def variables(self) -> List[str]:
        """Get the variables of all channels.
        
        Returns
        -------
        List[str]
            List of variables of all channels.
        """
        return [b.variable for b in self.binnings]

    @property
    def selection_queries(self) -> List[str]:
        """Get the selection queries of all channels.
        
        Returns
        -------
        List[str]
            List of selection queries of all channels.
        """
        return [b.selection_query for b in self.binnings]

    def get_unrolled_binning(self) -> Binning:
        """Get an unrolled binning of all channels.
        
        The bins will just be labeled as 'Global bin number' and the 
        variable will be 'none'. An unrolled binning cannot be used to 
        create a histogram directly, but it can be used for plotting.
        """
        bin_edges = np.arange(self.n_bins + 1)
        return Binning("none", bin_edges, "Global bin number")

    def _idx_channel(self, key: Union[int, str]) -> int:
        """Get the index of a channel from the key.
        The key may be a numeric index or a string label.
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return self.labels.index(key)
        else:
            raise ValueError(f"Invalid key {key}. Must be an integer or a string.")

    def _channel_bin_idx(self, key: Union[int, str]) -> List[int]:
        """Get the indices of bins in the unrolled binning belonging to a channel.

        The bin counts from the channel can be extracted from the full bin counts
        using these indices as follows:
        >>> idx = _channel_bin_idx(key)
        >>> channel_bin_counts = bin_counts[idx]
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        bin_idx : int
            Index of the bin in the channel.
        
        Returns
        -------
        List[int]
            List of indices of bins belonging to the channel.
        """
        idx = self._idx_channel(key)
        binning = self.binnings[idx]
        start_idx = sum([len(b) for b in self.binnings[:idx]])
        return list(range(start_idx, start_idx + len(binning)))

    def get_binning(self, key: Union[int, str]) -> Binning:
        """Get the binning of a given channel.
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        
        Returns
        -------
        Binning
            Binning of the channel.
        """
        return self.binnings[self._idx_channel(key)]

    def __iter__(self):
        self.__iter_index = 0
        return self

    def __next__(self) -> Binning:
        if self.__iter_index >= len(self.binnings):
            raise StopIteration
        else:
            binning = self.binnings[self.__iter_index]
            self.__iter_index += 1
            return binning

    def __len__(self):
        return self.n_channels


class MultiChannelHistogram(Histogram):
    """A histogram that combines multiple channels with a single covariance matrix.
    
    The purpose of this class is to hold several histograms that may be correlated,
    and to extract sub-channels from the full histogram.
    """

    def __init__(self, binning: MultiChannelBinning, *args, **kwargs) -> None:
        """Initialize a MultiChannelHistogram.
        
        Parameters
        ----------
        binning : MultiChannelBinning
            Binning of the histogram.
        bin_counts : np.ndarray
            Bin counts of the histogram. The length of the array must match the sum of the
            number of bins in each channel. This class assumes that the bin counts passed
            correspond to the concatenated bin counts of each channel, where the order
            of the channels is the same as the order of the binning.
        covariance_matrix : np.ndarray
            Covariance matrix of the histogram. It is a square matrix whose size is equal
            to that of the bin counts. The covariance matrix must be positive semi-definite.
        kwargs
            Additional keyword arguments passed to the Histogram constructor.
        """

        assert isinstance(binning, MultiChannelBinning), "binning must be a MultiChannelBinning."
        super().__init__(binning, *args, **kwargs)

    @property
    def channels(self) -> List[str]:
        """List of channels."""
        return self.binning.labels

    def channel_bin_counts(self, key: Union[int, str]) -> np.ndarray:
        """Get the bin counts of a given channel.
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        
        Returns
        -------
        np.ndarray
            Array of bin counts of the channel.
        """
        return self.bin_counts[self.binning._channel_bin_idx(key)]

    def channel_covariance_matrix(self, key: Union[int, str]) -> np.ndarray:
        """Get the covariance matrix of a given channel.
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        
        Returns
        -------
        np.ndarray
            Covariance matrix of the channel.
        """
        return self.covariance_matrix[
            np.ix_(self.binning._channel_bin_idx(key), self.binning._channel_bin_idx(key))
        ]

    def __getitem__(self, key: Union[int, str]) -> Histogram:
        """Get the histogram of a given channel.
        
        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        
        Returns
        -------
        Histogram
            Histogram of the channel.
        """
        idx = self.binning._idx_channel(key)
        return Histogram(
            self.binning.binnings[idx],
            unumpy.nominal_values(self.channel_bin_counts(idx)),
            covariance_matrix=self.channel_covariance_matrix(idx),
        )

    def replace_channel_histogram(self, key: Union[int, str], histogram: Histogram) -> None:
        """Replace the histogram of a given channel.
        
        This does not change the covariance _between_ channels.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        histogram : Histogram
            Histogram to replace the channel histogram with.
        """
        idx = self.binning._idx_channel(key)
        self.bin_counts[self._channel_bin_idx(idx)] = histogram.bin_counts
        self.covariance_matrix[
            np.ix_(self._channel_bin_idx(idx), self._channel_bin_idx(idx))
        ] = histogram.covariance_matrix

    def get_unrolled_histogram(self) -> Histogram:
        """Get an unrolled histogram of all channels.
        
        Returns
        -------
        Histogram
            Histogram with the concatenated bin counts and covariance matrix of all channels.
        """

        state = self.to_dict()
        state["binning"] = self.binning.get_unrolled_binning().__dict__
        return Histogram.from_dict(state)

    def __repr__(self):
        return f"MultiChannelHistogram(binning={self.binning}, bin_counts={self.bin_counts}, label={self.label}, tex={self.tex_string})"

    def draw(self, ax, as_errorbars=False, show_errors=True, **plot_kwargs):
        # call the draw method of the unrolled histogram
        return self.get_unrolled_histogram().draw(ax, as_errorbars, show_errors, **plot_kwargs)

    def draw_covariance_matrix(self, ax=None, as_correlation=True, **plot_kwargs):
        return self.get_unrolled_histogram().draw_covariance_matrix(
            ax, as_correlation, **plot_kwargs
        )

    @classmethod
    def from_dict(cls, dictionary, check_psd=True):
        """Create a MultiChannelHistogram from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary representation of the histogram.

        Returns
        -------
        MultiChannelHistogram
            MultiChannelHistogram object.
        """

        dictionary["binning"] = MultiChannelBinning(**dictionary["binning"])
        dictionary["check_psd"] = check_psd
        return cls(**dictionary)


class RunHistGenerator:
    """Histogram generator for data and simulation runs."""

    def __init__(
        self,
        rundata_dict: Dict[str, pd.DataFrame],
        binning: Binning,
        selection: Optional[str] = None,
        preselection: Optional[str] = None,
        data_pot: Optional[float] = None,
        sideband_generator: Optional["RunHistGenerator"] = None,
        uncertainty_defaults: Optional[Dict[str, bool]] = None,
        parameters: Optional[ParameterSet] = None,
        detvar_data_path: Optional[str] = None,
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
        detvar_data_path : str, optional
            Path to the JSON file containing histograms of detector variations. If provided,
            the detector variations will be treated as unisim variations for additional uncertainty.
        mc_hist_generator_cls : type, optional
            Class to use for the MC histogram generator. If None, the default HistogramGenerator
            class is used.
        **mc_hist_generator_kwargs
            Additional keyword arguments that are passed to the MC histogram generator on initialization.
        """
        self.data_pot = data_pot
        query = self.get_selection_query(selection, preselection)
        self.selection = selection
        self.preselection = preselection
        self.binning = binning
        self.logger = logging.getLogger(__name__)
        self.detvar_data = None
        if detvar_data_path is not None:
            # check that path exists
            if not os.path.exists(detvar_data_path):
                raise ValueError(f"Path {detvar_data_path} does not exist.")
            self.detvar_data = from_json(detvar_data_path)
            if not self.detvar_data["binning"] == self.binning:
                raise ValueError(
                    "Binning of detector variations does not match binning of main histogram."
                )
            if not self.detvar_data["selection"] == self.selection:
                raise ValueError(
                    "Selection of detector variations does not match selection of main histogram."
                )
            if not self.detvar_data["preselection"] == self.preselection:
                raise ValueError(
                    "Preselection of detector variations does not match preselection of main histogram."
                )

        # ensure that the necessary keys are present
        if "data" not in rundata_dict.keys():
            raise ValueError("data key is missing from rundata_dict (may be None).")
        if "mc" not in rundata_dict.keys():
            raise ValueError("mc key is missing from rundata_dict.")
        if "ext" not in rundata_dict.keys():
            raise ValueError("ext key is missing from rundata_dict (may be None).")

        for k, df in rundata_dict.items():
            if df is None:
                continue
            df["dataset_name"] = k
            df["dataset_name"] = df["dataset_name"].astype("category")
        mc_hist_generator_cls = (
            HistogramGenerator if mc_hist_generator_cls is None else mc_hist_generator_cls
        )
        # make one dataframe for all mc events
        df_mc = pd.concat([df for k, df in rundata_dict.items() if k not in ["data", "ext"]])
        df_ext = rundata_dict["ext"]
        df_data = rundata_dict["data"]
        if query is not None:
            # The Python engine is necessary because the queries tend to have too many inputs
            # for numexpr to handle.
            df_mc = df_mc.query(query, engine="python")
            if df_ext is not None:
                df_ext = df_ext.query(query, engine="python")
            if df_data is not None:
                df_data = df_data.query(query, engine="python")
            # the query has already been applied, so we can set it to None
            query = None

        self.parameters = parameters
        if self.parameters is None:
            self.parameters = ParameterSet([])  # empty parameter set
        else:
            assert isinstance(self.parameters, ParameterSet), "parameters must be a ParameterSet."
        self.mc_hist_generator = mc_hist_generator_cls(
            df_mc,
            binning,
            parameters=self.parameters,
            detvar_data=self.detvar_data,
            **mc_hist_generator_kwargs,
        )
        if df_ext is not None:
            self.ext_hist_generator = HistogramGenerator(df_ext, binning)
        else:
            self.ext_hist_generator = None
        if df_data is not None:
            self.data_hist_generator = HistogramGenerator(df_data, binning)
        else:
            self.data_hist_generator = None
        self.sideband_generator = sideband_generator
        self.uncertainty_defaults = dict() if uncertainty_defaults is None else uncertainty_defaults

    @classmethod
    def get_selection_query(cls, selection, preselection, extra_queries=None):
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
            self.uncertainty_defaults.get("add_ext_error_floor", False)
            if add_error_floor is None
            else add_error_floor
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
            prior_errors = np.ones(data_hist.n_bins) * 1.4 ** 2
            prior_errors[data_hist.nominal_values > 0] = 0
            data_hist.add_covariance(np.diag(prior_errors))
        data_hist *= scale_factor
        data_hist.label = {"data": "Data", "ext": "EXT"}[type]
        data_hist.color = "k"
        data_hist.hatch = {"data": None, "ext": "///"}[type]
        return data_hist

    def get_mc_hists(
        self,
        category_column="dataset_name",
        include_multisim_errors=None,
        scale_to_pot=None,
        channel=None,
    ):
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
                include_multisim_errors=include_multisim_errors,
                extra_query=extra_query,
                scale_to_pot=scale_to_pot,
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

    def get_mc_hist(
        self,
        include_multisim_errors=None,
        extra_query=None,
        scale_to_pot=None,
        use_sideband=None,
        add_precomputed_detsys=False,
    ):
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
        add_precomputed_detsys : bool, optional
            Whether to add the precomputed detector systematics to the histogram covariance.
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
        use_sideband = (
            self.uncertainty_defaults.get("use_sideband", False)
            if use_sideband is None
            else use_sideband
        )
        hist_generator = self.get_hist_generator(which="mc")
        use_sideband = use_sideband and self.sideband_generator is not None
        if use_sideband:
            sideband_generator = self.sideband_generator.get_hist_generator(which="mc")
            sideband_total_prediction = self.sideband_generator.get_total_prediction(
                include_multisim_errors=True
            )
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
            add_precomputed_detsys=add_precomputed_detsys,
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
        add_precomputed_detsys=False,
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
        add_precomputed_detsys : bool, optional
            Whether to add the precomputed detector systematics to the histogram covariance.
        """
        mc_prediction = self.get_mc_hist(
            include_multisim_errors=include_multisim_errors,
            extra_query=extra_query,
            scale_to_pot=scale_to_pot,
            use_sideband=use_sideband,
            add_precomputed_detsys=add_precomputed_detsys,
        )
        if self.ext_hist_generator is not None:
            ext_prediction = self.get_data_hist(
                type="ext", scale_to_pot=scale_to_pot, add_error_floor=add_ext_error_floor,
            )
            total_prediction = mc_prediction + ext_prediction
        else:
            total_prediction = mc_prediction
        return total_prediction

    def get_chi_square(self, **kwargs):
        """Get the chi square between the data and the total prediction.

        Returns
        -------
        chi_square : float
            Chi square between the data and the total prediction.
        """
        data_hist = self.get_data_hist(type="data")
        total_prediction = self.get_total_prediction(**kwargs)
        chi_sq = chi_square(
            data_hist.nominal_values,
            total_prediction.nominal_values,
            total_prediction.covariance_matrix,
        )
        return chi_sq


class HistogramGenerator:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        binning: Union[Binning, MultiChannelBinning],
        parameters: Optional[ParameterSet] = None,
        detvar_data: Optional[Union[Dict[str, Any], str]] = None,
        enable_cache: bool = True,
        cache_total_covariance: bool = True,
    ):
        """Create a histogram generator for a given dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the data to be binned.
        binning : Binning or MultiChannelBinning
            Binning object containing the binning of the histogram. If a MultiChannelBinning is
            passed, the histogram will be unrolled and the covariance matrix will contain
            correlations between bins of different channels. The `generate` function will 
            then return a MultiChannelHistogram.
        weight_column : str or list of str, optional
            Name of the column containing the weights of the data points. If more than one weight
            column is given, the weights are multiplied in sequence.
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
        self.parameters = parameters
        # Hardcode this as default because currently there is no situation where it would be
        # anything else. The only exceptions are when we calculate systematics, but even
        # then we also hard-coded which column must be used when.
        self.weight_column = "weights"
        if self.parameters is None:
            self.parameters = ParameterSet([])  # empty parameter set, still needed to check cache
        self.parameters_last_evaluated = None  # used to invalidate the cache
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Creating histogram generator for with binning: {binning}")
        self.binning = binning
        self.enable_cache = enable_cache
        self.cache_total_covariance = cache_total_covariance
        self.detvar_data = detvar_data
        # in case a string was passed to detvar_data, we load it from the file
        if isinstance(self.detvar_data, str):
            self.detvar_data = from_json(self.detvar_data)
        # check that binning matches to detvar_data
        if self.detvar_data is not None:
            if not self.detvar_data["binning"] == self.binning:
                raise ValueError(
                    "Binning of detector variations does not match binning of main histogram."
                )

        self._invalidate_cache()

    def _generate_hash(*args, **kwargs):
        hash_obj = hashlib.md5()
        data = str(args) + str(kwargs)
        hash_obj.update(data.encode("utf-8"))
        return hash_obj.hexdigest()

    def _invalidate_cache(self):
        """Invalidate the cache."""
        self.hist_cache = dict()
        self.unisim_hist_cache = dict()
        self.multisim_hist_cache = dict()
        self.multisim_hist_cache["weightsReint"] = dict()
        self.multisim_hist_cache["weightsFlux"] = dict()
        self.multisim_hist_cache["weightsGenie"] = dict()
        self.parameters_last_evaluated = self.parameters.copy()

    def _return_empty_hist(self):
        """Return an empty histogram."""
        if "binnings" in self.binning.__dict__:
            return MultiChannelHistogram(
                self.binning,
                np.zeros(self.binning.n_bins),
                uncertainties=np.zeros(self.binning.n_bins),
            )
        return Histogram(self.binning, np.zeros(self.binning.n_bins), np.zeros(self.binning.n_bins))

    def _get_query_mask(self, dataframe, query):
        """Get the boolean mask corresponding to the query."""
        if query is None:
            return np.ones(len(dataframe), dtype=bool)
        query_df = dataframe.query(query)
        query_indices = query_df.index
        mask = dataframe.index.isin(query_indices)
        return mask

    def _histogram_multi_channel(
        self, dataframe: pd.DataFrame, weight_column: Optional[Union[str, List[str]]] = None,
    ) -> MultiChannelHistogram:
        """Generate a histogram for multiple channels from the dataframe.

        The histograms contained in the MultiChannelHistogram are linked by one large covariance
        matrix. The covariance includes only the purely statistical uncertainties (i.e. no multisim
        or unisim uncertainties). The variance in each bin is calculated as the sum of the weights
        squared. The correlation between bins of different channels is calculated as the sum of
        squares of the weights that appear in both bins, which is equivalent to the 2D histogram of
        the squared weights in the two channels.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the data to be binned. This function assumes that any
            other filtering has already been applied.
        weight_column : str or list of str, optional
            Name of the column containing the weights of the data points. If more than one weight
            column is given, the weights are multiplied in sequence. If None, the weight_column
            passed to the constructor is used. If that is also None, the weights are set to 1.

        Returns
        -------
        histogram : MultiChannelHistogram
            MultiChannelHistogram object containing the binned data.
        """

        binning = self.binning
        return_single_channel = False
        if not hasattr(binning, "variables"):
            binning = MultiChannelBinning([binning])
            return_single_channel = True
        sample = dataframe[binning.variables].to_numpy()
        selection_masks = []
        for i, query in enumerate(binning.selection_queries):
            selection_masks.append(self._get_query_mask(dataframe, query))
        bin_edges = binning.consecutive_bin_edges
        weights = self.get_weights(dataframe, weight_column)

        channel_bin_counts = []
        channel_bin_variances = []
        # for the bin-counts and the variances, we just have to make 1D histograms for each channel
        for i in range(binning.n_channels):
            channel_sample = sample[selection_masks[i], i]
            channel_weights = weights[selection_masks[i]]
            channel_bin_counts.append(
                np.histogram(channel_sample, bins=bin_edges[i], weights=channel_weights)[0]
            )
            channel_bin_variances.append(
                np.histogram(channel_sample, bins=bin_edges[i], weights=channel_weights ** 2)[0]
            )
        # now we build the covariance
        covariance_matrix = np.diag(np.concatenate(channel_bin_variances))
        # for off-diagonal blocks, we need to calculate 2D histograms of the squared weights
        # for each pair of channels
        for i in range(binning.n_channels):
            for j in range(i + 1, binning.n_channels):
                mask = selection_masks[i] & selection_masks[j]
                if np.sum(mask) == 0:
                    continue
                hist, _, _ = np.histogram2d(
                    sample[mask, i],
                    sample[mask, j],
                    bins=[bin_edges[i], bin_edges[j]],
                    weights=weights[mask] ** 2,
                )
                # put the 2D histogram into the off-diagonal block of the covariance matrix
                covariance_matrix[
                    np.ix_(binning._channel_bin_idx(i), binning._channel_bin_idx(j))
                ] = hist
                # copy the transpose into the other off-diagonal block
                covariance_matrix[
                    np.ix_(binning._channel_bin_idx(j), binning._channel_bin_idx(i))
                ] = hist.T
        covariance_matrix, dist = fronebius_nearest_psd(covariance_matrix, return_distance=True)
        if dist > 1e-3:
            raise RuntimeError(f"Nearest PSD distance is {dist} away, which is too large.")
        if return_single_channel:
            return Histogram(
                binning.binnings[0], channel_bin_counts[0], covariance_matrix=covariance_matrix,
            )
        return MultiChannelHistogram(
            binning, np.concatenate(channel_bin_counts), covariance_matrix=covariance_matrix,
        )

    def _multi_channel_universes(
        self,
        dataframe: pd.DataFrame,
        base_weight_column: str,
        multisim_weight_column: str,
        weight_rescale: float = 1 / 1000.0,
    ) -> List[np.ndarray]:
        """Generate histograms for each universe for multiple channels from the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the data to be binned. This function assumes that any
            other filtering has already been applied.
        base_weight_column : str
            Name of the column containing the base weights.
        multisim_weight_column : str
            Name of the column containing the universe weights.

        Returns
        -------
        bin_counts : List[np.ndarray]
            List of bin counts for each universe.
        """

        binning = self.binning
        if not hasattr(binning, "variables"):
            binning = MultiChannelBinning([binning])
        sample = dataframe[binning.variables].to_numpy()
        selection_masks = []
        for i, query in enumerate(binning.selection_queries):
            selection_masks.append(self._get_query_mask(dataframe, query))
        bin_edges = binning.consecutive_bin_edges
        base_weights = self.get_weights(dataframe, base_weight_column)
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
        for column in df.columns:
            # create a histogram for each universe
            universe_weights = self._limit_weights(df[column].values * weight_rescale)
            channel_bin_counts = []
            for i in range(binning.n_channels):
                channel_sample = sample[selection_masks[i], i]
                channel_weights = (
                    base_weights[selection_masks[i]] * universe_weights[selection_masks[i]]
                )
                channel_bin_counts.append(
                    np.histogram(channel_sample, bins=bin_edges[i], weights=channel_weights)[0]
                )
            universe_histograms.append(np.concatenate(channel_bin_counts))
        return np.array(universe_histograms)

    def generate(
        self,
        include_multisim_errors=False,
        use_sideband=False,
        extra_query=None,
        sideband_generator=None,
        sideband_total_prediction=None,
        sideband_observed_hist=None,
        add_precomputed_detsys=False,
    ):
        """Generate a histogram from the dataframe.

        Parameters
        ----------
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
        add_precomputed_detsys : bool, optional
            Whether to include the detector systematics in the covariance matrix.

        Returns
        -------
        histogram : Histogram
            Histogram object containing the binned data.
        """

        if use_sideband:
            assert sideband_generator is not None
            assert sideband_total_prediction is not None
            assert sideband_observed_hist is not None
        if add_precomputed_detsys:
            assert self.detvar_data is not None, "No detector variations provided."
        calculate_hist = True
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self.logger.debug("Parameters changed, invalidating cache.")
                self._invalidate_cache()
            hash = self._generate_hash(
                extra_query, add_precomputed_detsys, use_sideband, include_multisim_errors,
            )
            if hash in self.hist_cache:
                self.logger.debug("Histogram found in cache.")
                if self.cache_total_covariance:
                    self.logger.debug("Using cached total covariance matrix.")
                    return self.hist_cache[hash].copy()
                hist = self.hist_cache[hash].copy()
                calculate_hist = False
        if calculate_hist:
            if extra_query is not None:
                dataframe = self.dataframe.query(extra_query, engine="python")
            else:
                dataframe = self.dataframe
            if len(dataframe) == 0:
                self.logger.debug("No events in dataframe, returning empty histogram.")
                hist = self._return_empty_hist()
                if self.enable_cache:
                    self.hist_cache[hash] = hist.copy()
                return hist
            hist = self._histogram_multi_channel(dataframe)
            if self.enable_cache:
                self.hist_cache[hash] = hist.copy()
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
                    ms_column, extra_query=extra_query, return_histograms=True,
                )
                hist.add_covariance(cov_mat)

                if use_sideband:
                    extended_cov += self.multiband_covariance(
                        [self, sideband_generator], ms_column, extra_queries=[extra_query, None],
                    )

            # calculate unisim histograms
            self.logger.debug("Calculating unisim uncertainties")
            cov_mat_unisim = self.calculate_unisim_uncertainties(
                central_value_hist=hist, extra_query=extra_query
            )
            hist.add_covariance(cov_mat_unisim)

            if use_sideband:
                # calculate constraint correction
                mu_offset, cov_corr = sideband_constraint_correction(
                    sideband_measurement=sideband_observed_hist.nominal_values,
                    sideband_central_value=sideband_total_prediction.nominal_values,
                    concat_covariance=extended_cov,
                    sideband_covariance=sideband_total_prediction.covariance_matrix,
                )
                self.logger.debug(f"Sideband constraint correction: {mu_offset}")
                self.logger.debug(f"Sideband constraint covariance correction: {cov_corr}")
                # add corrections to histogram
                hist += mu_offset
                if not is_psd(hist.covariance_matrix + cov_corr):
                    raise RuntimeError("Covariance matrix is not PSD after correction.")
                hist.add_covariance(cov_corr)
        if add_precomputed_detsys:
            det_cov = self.calculate_detector_covariance()
            if det_cov is not None:
                hist.add_covariance(det_cov)
        if self.enable_cache and self.cache_total_covariance:
            self.hist_cache[hash] = hist.copy()
        return hist

    @classmethod
    def multiband_covariance(cls, hist_generators, ms_column, extra_queries=None):
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
        extra_queries : list of str, optional
            List of additional queries to apply to the dataframe.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        # types = [type(hg) for hg in hist_generators]
        # assert all(isinstance(hg, cls) for hg in hist_generators), f"Must provide a list of HistogramGenerator objects. Types are {types}."

        universe_hists = []
        central_values = []
        if extra_queries is None:
            extra_queries = [None] * len(hist_generators)
        for hg, extra_query in zip(hist_generators, extra_queries):
            cov_mat, universe_hist = hg.calculate_multisim_uncertainties(
                ms_column, return_histograms=True, extra_query=extra_query
            )
            universe_hists.append(universe_hist)
            central_values.append(
                hg.generate(extra_query=extra_query, include_multisim_errors=False).nominal_values
            )

        concatenated_cv = np.concatenate(central_values)
        concatenated_universes = np.concatenate(universe_hists, axis=1)
        cov_mat = covariance(
            concatenated_universes, concatenated_cv, allow_approximation=True, tolerance=1e-10,
        )
        return cov_mat

    @classmethod
    def multiband_unisim_covariance(cls, hist_generators, extra_queries=None):
        """Calculate the covariance matrix for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the covariance matrix of the
        unisim universes. The underlying assumption
        is that the weights listed in the unisim column are from the same universes
        in the same order for all histograms.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        extra_queries : list of str, optional
            List of additional queries to apply to the dataframe.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        universe_hist_dicts = []
        concatenated_cv = []
        if extra_queries is None:
            extra_queries = [None] * len(hist_generators)
        for hg, extra_query in zip(hist_generators, extra_queries):
            concatenated_cv.append(hg.generate(extra_query=extra_query).nominal_values)
            universe_hist_dicts.append(
                hg.calculate_unisim_uncertainties(return_histograms=True, extra_query=extra_query)[
                    1
                ]
            )
        concatenated_cv = np.concatenate(concatenated_cv)
        knobs = list(universe_hist_dicts[0].keys())
        summed_cov_mat = np.zeros((len(concatenated_cv), len(concatenated_cv)))
        for knob in knobs:
            assert all(
                knob in hist_dict for hist_dict in universe_hist_dicts
            ), f"Knob {knob} not found in all histograms."
            concatenated_universes = np.concatenate(
                [hist_dict[knob] for hist_dict in universe_hist_dicts], axis=1
            )
            cov_mat = covariance(
                concatenated_universes, concatenated_cv, allow_approximation=True, tolerance=1e-10,
            )
            summed_cov_mat += cov_mat
        return summed_cov_mat

    @classmethod
    def multiband_detector_covariance(cls, hist_generators):
        """Calculate the covariance matrix for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the covariance matrix of the
        detector variations.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        # Before we start, we have to make sure that the same truth-filtered sets were used
        # in the detector systematics of all histograms. Otherwise, we could not combine
        # the universes in a meaningful way.
        # The detvar_data dictionary contains a dictionary of all the filter queries that
        # were used under the key `filter_queries`.
        reference_filter_queries = hist_generators[0].detvar_data["filter_queries"]
        for hg in hist_generators:
            assert (
                hg.detvar_data["filter_queries"] == reference_filter_queries
            ), "Not all histograms have the same filter queries."

        universe_hist_dicts = []
        for hg in hist_generators:
            universe_hist_dicts.append(hg.calculate_detector_covariance(return_histograms=True)[1])
        datasets = list(universe_hist_dicts[0].keys())
        knobs = list(universe_hist_dicts[0][datasets[0]].keys())
        total_bins = sum([len(hg.binning) for hg in hist_generators])
        summed_cov_mat = np.zeros((total_bins, total_bins))
        for dataset in datasets:
            for knob in knobs:
                assert all(
                    knob in hist_dict[dataset] for hist_dict in universe_hist_dicts
                ), f"Knob {knob} not found in all histograms."
                concatenated_universes = np.concatenate(
                    [hist_dict[dataset][knob] for hist_dict in universe_hist_dicts], axis=1,
                )
                # The detector universes are special, because the central value has already been subtracted
                # by construction. Therefore, we can use the zero vector as the central value.
                cov_mat = covariance(
                    concatenated_universes,
                    np.zeros(total_bins),
                    allow_approximation=True,
                    tolerance=1e-10,
                )
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

    def get_weights(self, dataframe, weight_column=None, limit_weight=True):
        """Get the weights of the dataframe after filtering for the given query.

        Parameters
        ----------
        weight_column : str or list of str, optional
            Name of the column containing the weights of the data points. If more than one weight
            column is given, the weights are multiplied in sequence. If None, the weight_column
            passed to the constructor is used. If that is also None, the weights are set to 1.
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
        if weight_column is None:
            weight_column = self.weight_column
        if weight_column is None:
            return np.ones(len(dataframe))
        if isinstance(weight_column, str):
            weight_column = [weight_column]
        weights = np.ones(len(dataframe))
        for col in weight_column:
            weights *= dataframe[col]
        if limit_weight:
            weights = self._limit_weights(weights)
        return self.adjust_weights(dataframe, weights)

    def _limit_weights(self, weights):
        weights = np.asarray(weights)
        weights[weights > 100] = 0.0
        weights[weights < 0] = 0.0
        weights[~np.isfinite(weights)] = 0.0
        if np.sum(~np.isfinite(weights)) > 0:
            self.logger.debug(
                f"Found {np.sum(~np.isfinite(weights))} invalid weights (set to one)."
            )
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
        multisim_weight_column : str
            Name of the column containing the multisim weights of the events.
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

        if multisim_weight_column not in self.dataframe.columns:
            raise ValueError(f"Weight column {multisim_weight_column} is not in the dataframe.")
        assert multisim_weight_column in [
            "weightsGenie",
            "weightsFlux",
            "weightsReint",
        ], "Unknown multisim weight column."
        if weight_column is None:
            weight_column = (
                "weights_no_tune" if multisim_weight_column == "weightsGenie" else "weights"
            )
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self._invalidate_cache()
            if extra_query is None:
                hash = "None"
            else:
                hash = hashlib.sha256(extra_query.encode("utf-8")).hexdigest()
            if hash in self.multisim_hist_cache[multisim_weight_column]:
                self.logger.debug(
                    f"Multisim histogram for {multisim_weight_column} found in cache."
                )
                if return_histograms:
                    return self.multisim_hist_cache[multisim_weight_column][hash]
                else:
                    return self.multisim_hist_cache[multisim_weight_column][hash][0]
        if extra_query is not None:
            dataframe = self.dataframe.query(extra_query, engine="python")
        else:
            dataframe = self.dataframe
        universe_histograms = self._multi_channel_universes(
            dataframe, weight_column, multisim_weight_column, weight_rescale=weight_rescale,
        )
        if central_value_hist is None:
            central_value_hist = self._histogram_multi_channel(dataframe)
        # calculate the covariance matrix from the histograms
        cov = covariance(universe_histograms, central_value_hist.nominal_values, allow_approximation=True, tolerance=1e-10)
        self.logger.debug(f"Calculated covariance matrix for {multisim_weight_column}.")
        self.logger.debug(f"Bin-wise error contribution: {np.sqrt(np.diag(cov))}")
        if self.enable_cache:
            self.multisim_hist_cache[multisim_weight_column][hash] = (
                cov,
                universe_histograms,
            )
        if return_histograms:
            return cov, universe_histograms
        return cov

    def calculate_unisim_uncertainties(
        self, central_value_hist=None, extra_query=None, return_histograms=False
    ):
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

        if extra_query is None:
            hash = "None"
        else:
            hash = hashlib.sha256(extra_query.encode("utf-8")).hexdigest()
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self._invalidate_cache()
        knob_v = [
            "knobRPA",
            "knobCCMEC",
            "knobAxFFCCQE",
            "knobVecFFCCQE",
            "knobDecayAngMEC",
            "knobThetaDelta2Npi",
        ]
        # see table 23 from the technote
        knob_n_universes = [2, 1, 1, 1, 1, 1]
        # because all of these are GENIE knobs, we need to use the weights without the GENIE tune just as
        # for the GENIE multisim
        base_weight_column = "weights_no_tune"
        # When we have two universes, then there are two weight variations, knobXXXup and knobXXXdown. Otherwise, there
        # is only one weight variation, knobXXXup.
        total_cov = np.zeros((self.binning.n_bins, self.binning.n_bins))
        dataframe = None
        observation_dict = dict()
        for knob, n_universes in zip(knob_v, knob_n_universes):
            observations = []
            if self.enable_cache and (hash not in self.unisim_hist_cache):
                self.unisim_hist_cache[hash] = dict()
            if self.enable_cache and knob in self.unisim_hist_cache[hash]:
                self.logger.debug(f"Unisim histogram for knob {knob} found in cache.")
                observations = self.unisim_hist_cache[hash][knob]
                if central_value_hist is None:
                    central_value_hist = self.unisim_hist_cache[hash]["central_value"]
            else:
                # only when we didn't find a knob in the cache, we want to query the dataframe
                # as this is also an expensive operation
                if dataframe is None:
                    if extra_query is not None:
                        dataframe = self.dataframe.query(extra_query, engine="python")
                    else:
                        dataframe = self.dataframe
                    # At this point we also want to calculate the central value histogram
                    if central_value_hist is None:
                        central_value_hist = self._histogram_multi_channel(dataframe)
                for universe in range(n_universes):
                    # get the weight column for this universe
                    weight_column_knob = (
                        f"{knob}up" if n_universes == 2 and universe == 0 else f"{knob}dn"
                    )
                    bincounts = self._histogram_multi_channel(
                        dataframe,
                        # multiply the base weights with the knob weights
                        weight_column=[base_weight_column, weight_column_knob],
                    ).nominal_values
                    observations.append(bincounts)
                observations = np.array(observations)
                if self.enable_cache:
                    self.unisim_hist_cache[hash][knob] = observations
                    self.unisim_hist_cache[hash]["central_value"] = central_value_hist
            observation_dict[knob] = observations
            # calculate the covariance matrix from the histograms
            cov = covariance(
                observations,
                central_value_hist.nominal_values,
                allow_approximation=True,
                debug_name=knob,
                tolerance=1e-10,
            )
            self.logger.debug(
                f"Bin-wise error contribution for knob {knob}: {np.sqrt(np.diag(cov))}"
            )
            # add it to the total covariance matrix
            total_cov += cov
        if return_histograms:
            return total_cov, observation_dict
        return total_cov

    def calculate_detector_covariance(self, only_diagonal=False, return_histograms=False):
        """Get the covariance matrix for detector uncertainties.

        This function follows the recommendation outlined in:
        A. Ashkenazi, et al., "Detector Systematics supporting note", DocDB 27009

        The relevant quote is:
            We recommend running the full analysis chain on each one of the samples, and adding the
            difference between each sample and the central value in quadrature. An exception should be
            made to the recombination and wire modification dEdx variations. The recommendation is to
            use only the one that has the larger impact to obtain the uncertainty.
        
        In newer samples, the dEdx variations are not included anymore, so we do not need to worry
        about them here.
        """
        if self.detvar_data is None:
            return None
        variations = [
            "lydown",
            "lyatt",
            "lyrayleigh",
            "sce",
            "recomb2",
            "wiremodx",
            "wiremodyz",
            "wiremodthetaxz",
            "wiremodthetayz",
        ]
        variation_hist_data = self.detvar_data["variation_hist_data"]
        # Detector variations are calculated separately for each truth-filtered set. We can not
        # assume, however, that the truth-filtered sets are identical to those used in this RunHistGenerator.
        # Instead, we use the filter queries that are part of the detector variation data to select the
        # correct samples.
        filter_queries = self.detvar_data["filter_queries"]
        cov_mat = np.zeros((self.binning.n_bins, self.binning.n_bins))
        observation_dict = {}

        for dataset, query in filter_queries.items():
            self.logger.debug(
                f"Getting detector covariance for dataset {dataset} with query {query}."
            )
            observation_dict[dataset] = {}
            variation_hists = variation_hist_data[dataset]
            this_analysis_hist = self.generate(extra_query=query, add_precomputed_detsys=False)
            cv_hist = this_analysis_hist.nominal_values
            variation_cv_hist = variation_hists["cv"].nominal_values

            # make sure every variation is in the dictionary
            for v in variations:
                if v not in variation_hists:
                    raise ValueError(f"Variation {v} is missing from the detector variations.")
            # We are taking the *relative* error from the detector variation data and scale it
            # to the bin counts of the histogram of this analysis. This means that, for example,
            # if the error in a bin was 10% in the detector variation data, and the bin count
            # in this analysis was 100, the error in this analysis will be 10.
            with np.errstate(divide="ignore", invalid="ignore"):
                variation_diffs = {
                    v: (h.nominal_values - variation_cv_hist) * (cv_hist / variation_cv_hist)
                    for v, h in variation_hists.items()
                }
            # set nan values to zero. These can occur when bins are empty, which we can safely ignore.
            for v, h in variation_diffs.items():
                h[~np.isfinite(h)] = 0.0
                observation_dict[dataset][v] = h.reshape(1, -1)
                # We have just one observation and the central value is zero since it was already subtracted
                cov_mat += covariance(
                    h.reshape(1, -1),
                    central_value=np.zeros_like(h),
                    # As with all unisim variations, small deviations from the PSD case are expected
                    allow_approximation=True,
                    tolerance=1e-10,
                    debug_name=f"detector_{v}",
                )
        if only_diagonal:
            cov_mat = np.diag(np.diag(cov_mat))
        if return_histograms:
            return cov_mat, observation_dict
        return cov_mat

    def _resync_parameters(self):
        """Not needed since there are no parameters in this class."""
        pass

    def _check_shared_parameters(self, parameters):
        return True
