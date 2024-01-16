from typing import List, Optional, Union
from matplotlib.axes import Axes
import numpy as np
from scipy.linalg import block_diag

from numbers import Real, Real
from uncertainties import correlated_values, unumpy

from microfit.statistics import (
    error_propagation_division,
    error_propagation_multiplication,
    is_psd,
    sideband_constraint_correction,
)
from microfit.histogram import Binning, MultiChannelBinning


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
            set in the Binning object that is passed to the constructor. This label should
            represent the category of the histogram, e.g. "Signal" or "Background" and
            will be used in the legend of a plot (unless a tex_string is provided).
        plot_color : str, optional
            Color of the histogram, used for plotting.
        tex_string : str, optional
            TeX string used to label the histogram in the legend of a plot.

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

        if as_correlation and as_fractional:
            raise ValueError("Cannot draw covariance matrix as correlation and as fractional.")
        if ax is None:
            ax = plt.gca()
        ax.set_title(f"{'Correlation' if as_correlation else 'Covariance'} matrix")
        X, Y = np.meshgrid(self.binning.bin_centers, self.binning.bin_centers)
        colormap = plot_kwargs.pop("cmap", "RdBu_r")
        label = None
        if as_correlation:
            vmin = plot_kwargs.pop("vmin", -1)
            vmax = plot_kwargs.pop("vmax", 1)
            pc = ax.pcolormesh(
                X, Y, self.correlation_matrix, cmap=colormap, vmin=vmin, vmax=vmax, **plot_kwargs
            )
            label = "Correlation"
        elif as_fractional:
            # plot fractional covariance matrix
            fractional_covar = self.covariance_matrix / np.outer(
                self.nominal_values, self.nominal_values
            )
            max_val = np.max(np.abs(fractional_covar))
            vmin = plot_kwargs.pop("vmin", -max_val)
            vmax = plot_kwargs.pop("vmax", max_val)
            pc = ax.pcolormesh(
                X, Y, fractional_covar, cmap=colormap, vmin=vmin, vmax=vmax, **plot_kwargs
            )
            label = "Fractional covariance"
        else:
            max_val = np.max(np.abs(self.covariance_matrix))
            vmin = plot_kwargs.pop("vmin", -max_val)
            vmax = plot_kwargs.pop("vmax", max_val)
            pc = ax.pcolormesh(
                X, Y, self.covariance_matrix, cmap=colormap, vmin=vmin, vmax=vmax, **plot_kwargs
            )
            label = "Covariance"
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel(self.binning.variable_tex)  # type: ignore
        ax.set_ylabel(self.binning.variable_tex)  # type: ignore
        return ax

    def draw(
        self,
        ax: Optional[Axes] = None,
        as_errorbars: bool = False,
        show_errors: bool = True,
        with_ylabel: bool = True,
        **plot_kwargs,
    ) -> Axes:
        """Draw the histogram on a matplotlib axis.

        Parameters
        ----------
        ax : Axes
            Axis to draw the histogram on.
        as_errorbars : bool, optional
            Whether to draw the histogram as errorbars.
        with_ylabel : bool, optional
            Whether to draw the y-axis label.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to the matplotlib step function.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        assert ax is not None
        if self.binning.is_log:
            ax.set_yscale("log")
        ax.set_xlabel(self.binning.variable_tex)  # type: ignore
        if with_ylabel:
            ax.set_ylabel("Events")
        bin_counts = self.nominal_values
        bin_edges = self.binning.bin_edges
        label = plot_kwargs.pop("label", self.tex_string)
        color = plot_kwargs.pop("color", self.color)
        hatch = plot_kwargs.pop("hatch", self.hatch)
        ax.set_xlim((bin_edges[0], bin_edges[-1]))
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
        return fig._repr_html_()  # type: ignore

    def to_dict(self):
        """Convert the histogram to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the histogram.
        """

        return {
            "binning": self.binning.to_dict(),
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
            self.binning,
            fluctuated_bin_counts,
            covariance_matrix=self.covariance_matrix,
        )

    def fluctuate_poisson(self, seed=None):
        """Fluctuate bin counts according to Poisson uncertainties and return a new histogram with the fluctuated counts."""
        rng = np.random.default_rng(seed)
        fluctuated_bin_counts = rng.poisson(unumpy.nominal_values(self.bin_counts))
        return self.__class__(
            self.binning,
            fluctuated_bin_counts,
            uncertainties=np.sqrt(fluctuated_bin_counts),
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
        assert isinstance(other, Histogram), "Can only add Histograms or np.ndarrays to Histograms."
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
        if isinstance(other, Real):
            new_bin_counts = self.nominal_values / other
            new_cov_matrix = self.covariance_matrix / other**2
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
        elif isinstance(other, Real):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.covariance_matrix * other**2
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
        if isinstance(other, Real):
            new_bin_counts = self.nominal_values * other
            new_cov_matrix = self.covariance_matrix * other**2
            state = self.to_dict()
            state["bin_counts"] = new_bin_counts
            state["covariance_matrix"] = new_cov_matrix
            return self.__class__.from_dict(state)
        else:
            raise NotImplementedError(
                "Histogram multiplication is only supported for numeric types."
            )

    def __len__(self):
        return self.n_bins


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
        super().__init__(binning.get_unrolled_binning(), *args, **kwargs)
        # The attribute self.binning is set in the Histogram constructor, but we overwrite
        # it here again.
        self.binning = binning

    def roll_channels(self, shift: int) -> None:
        """Roll the channels of the histogram.

        Parameters
        ----------
        shift : int
            Number of channels to roll. Positive values roll the channels to the right,
            negative values roll the channels to the left.
        """
        # We roll the bins sequentially because the number of bins in each channel may be different.
        sign = np.sign(shift)
        for i in range(abs(shift)):
            # If the sign is positive, we need to roll the bin counts by the
            # number of bins in the last channel. If it is negative, we need to
            # roll by the number of bins in the first channel.
            first_last = 0 if sign < 0 else -1
            shift_bins = self.binning[first_last].n_bins * sign
            self.bin_counts = np.roll(self.bin_counts, shift_bins)
            # We also need to roll the covariance matrix accordingly in 2D
            self.covariance_matrix = np.roll(self.covariance_matrix, shift_bins, axis=0)
            self.covariance_matrix = np.roll(self.covariance_matrix, shift_bins, axis=1)
            self.binning.roll_channels(sign)

    def roll_channel_to_first(self, key: Union[int, str]) -> None:
        """Roll a given channel to the first position.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        idx = self.binning._idx_channel(key)
        self.roll_channels(-idx)

    def roll_channel_to_last(self, key: Union[int, str]) -> None:
        """Roll a given channel to the last position.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        idx = self.binning._idx_channel(key)
        # We have to keep in mind that the index of the last channel is len(self.binning) - 1,
        # so we need to roll by len(self.binning) - (idx + 1) bins.
        # For example, if the binning is already at the last position, then idx = len(self.binning) - 1,
        # and we need to roll by 0 bins.
        shift = len(self.binning) - (idx + 1)
        if shift != 0:
            self.roll_channels(shift)

    def update_with_measurement(
        self, key: Union[int, str], measurement: np.ndarray, central_value: Optional[np.ndarray] = None
    ) -> "MultiChannelHistogram":
        """Perform the Bayesian update of the histogram given a measurement of one channel.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        measurement : np.ndarray
            Measurement of the channel.
        central_value : np.ndarray, optional
            Central value of the sideband. If provided, this overrides the expectation value
            of the sideband that is stored in the histogram.

        Returns
        -------
        MultiChannelHistogram
            Updated histogram containing all channels except the one that was measured.
        """

        # By construction, the constraint correction function expects the sideband to be the
        # last block of bins in the histogram. We therefore roll the channel to the last position.
        self.roll_channel_to_last(key)
        idx = self.binning._idx_channel(key)
        # This should always be the case, since we use the function above to roll the channel to the last position.
        assert idx == len(self.binning) - 1, "Channel must be the last channel in the histogram."
        sideband_central_value = self[idx].nominal_values if central_value is None else central_value
        concat_covariance_matrix = self.covariance_matrix
        assert len(measurement) == len(
            sideband_central_value
        ), "Measurement must have the same length as the bin counts of the channel."
        delta_mu, delta_cov = sideband_constraint_correction(
            measurement,
            sideband_central_value,
            concat_covariance=concat_covariance_matrix,
        )
        # We need to generate a new MultiChannelHistogram where we have removed the channel that we measured.
        constrained_histogram = self.copy()
        del constrained_histogram[idx]
        # Update bin counts and covariance
        constrained_histogram.bin_counts += delta_mu
        constrained_histogram.covariance_matrix += delta_cov
        return constrained_histogram

    @property
    def channels(self) -> List[Union[str, None]]:
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

    def __getitem__(self, key: Union[int, str, List[Union[int, str]]]) -> Histogram:
        """Get the histogram of a given channel or a list of channels.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.

        Returns
        -------
        Histogram or MultiChannelHistogram
            Histogram of the channel(s). If a single channel is requested, a Histogram is returned.
            If a list of channels is requested, a MultiChannelHistogram is returned containing
            only the requested channels with their correlations.
        """
        if isinstance(key, list):
            # If we request a list of channels, we return a MultiChannelHistogram containing only
            # the requested channels.
            new_binning = MultiChannelBinning(
                [self.binning[i] for i in key],
            )
            new_bin_counts = np.concatenate(
                [unumpy.nominal_values(self.channel_bin_counts(i)) for i in key]
            )
            new_covariance_matrix = block_diag(
                *[self.channel_covariance_matrix(i) for i in key],
            )
            # Now we need to also fill in the off-diagonal blocks of the covariance matrix,
            # which correspond to the correlations between the channels.
            # We do this by looping over all pairs of channels and filling in the corresponding
            # block of the covariance matrix.
            for i, channel_i in enumerate(key):
                for j, channel_j in enumerate(key):
                    if i == j:
                        continue
                    # We can make use of the Binning to translate the indices from one
                    # covariance matrix to the other.
                    new_covariance_matrix[
                        np.ix_(
                            new_binning._channel_bin_idx(channel_i),
                            new_binning._channel_bin_idx(channel_j),
                        )
                    ] += self.covariance_matrix[
                        np.ix_(
                            self.binning._channel_bin_idx(channel_i),
                            self.binning._channel_bin_idx(channel_j),
                        )
                    ]
            return MultiChannelHistogram(
                new_binning,
                new_bin_counts,
                covariance_matrix=new_covariance_matrix,
                label=self.label,
                tex_string=self.tex_string,
                plot_color=self.color,
                plot_hatch=self.hatch,
            )
        idx = self.binning._idx_channel(key)
        return Histogram(
            self.binning.binnings[idx],
            unumpy.nominal_values(self.channel_bin_counts(idx)),
            covariance_matrix=self.channel_covariance_matrix(idx),
            plot_color=self.color,
            plot_hatch=self.hatch,
            label=self.label,
            tex_string=self.tex_string,
        )

    def __delitem__(self, key: Union[int, str]) -> None:
        """Delete the histogram of a given channel.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        idx = self.binning._idx_channel(key)
        self.bin_counts = np.delete(self.bin_counts, self.binning._channel_bin_idx(idx))
        self.covariance_matrix = np.delete(
            self.covariance_matrix, self.binning._channel_bin_idx(idx), axis=0
        )
        self.covariance_matrix = np.delete(
            self.covariance_matrix, self.binning._channel_bin_idx(idx), axis=1
        )
        self.binning.delete_channel(idx)

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
        self.bin_counts[self.binning._channel_bin_idx(idx)] = histogram.bin_counts
        self.covariance_matrix[
            np.ix_(self.binning._channel_bin_idx(idx), self.binning._channel_bin_idx(idx))
        ] = histogram.covariance_matrix
    
    def append_empty_channel(self, binning: Binning) -> None:
        """Append an empty channel to the histogram.

        Parameters
        ----------
        binning : Binning
            Binning of the new channel.
        """
        assert not isinstance(binning, MultiChannelBinning), "Cannot append a MultiChannelBinning."
        self.binning.binnings.append(binning)
        self.bin_counts = np.append(self.bin_counts, np.zeros(binning.n_bins))
        self.covariance_matrix = block_diag(self.covariance_matrix, np.zeros((binning.n_bins, binning.n_bins)))

    def get_unrolled_histogram(self) -> Histogram:
        """Get an unrolled histogram of all channels.

        Returns
        -------
        Histogram
            Histogram with the concatenated bin counts and covariance matrix of all channels.
        """

        state = self.to_dict()
        state["binning"] = self.binning.get_unrolled_binning().to_dict()
        return Histogram.from_dict(state)

    def __repr__(self):
        return f"MultiChannelHistogram(binning={self.binning}, bin_counts={self.bin_counts}, label={self.label}, tex={self.tex_string})"

    def draw(self, ax=None, as_errorbars=False, show_errors=True, **plot_kwargs):
        # call the draw method of the unrolled histogram
        unrolled_hist = self.get_unrolled_histogram()
        ax = unrolled_hist.draw(ax, as_errorbars, show_errors, **plot_kwargs)
        channel_n_bins = np.cumsum([0] + [len(b) for b in self.binning])
        channel_labels = [b.label for b in self.binning]
        for n_bins in channel_n_bins[1:-1]:
            ax.axvline(n_bins, color="k", linestyle="--")

        # Add text boxes for each channel label
        for i, label in enumerate(channel_labels):
            if label is None:
                continue
            ax.text(
                (channel_n_bins[i] + channel_n_bins[i + 1]) / 2,
                0.0,
                label,
                ha="center",
                va="bottom",
            )

        return ax

    def draw_covariance_matrix(self, ax=None, as_correlation=True, **plot_kwargs):
        ax = self.get_unrolled_histogram().draw_covariance_matrix(ax, as_correlation, **plot_kwargs)
        channel_n_bins = np.cumsum([0] + [len(b) for b in self.binning])
        channel_labels = [b.label for b in self.binning]
        for n_bins in channel_n_bins[1:-1]:
            ax.axvline(n_bins, color="k", linestyle="--")
            ax.axhline(n_bins, color="k", linestyle="--")
        # turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # set tick marks at every bin
        ax.set_xticks(np.arange(self.n_bins) + 0.5, minor=False)
        ax.set_yticks(np.arange(self.n_bins) + 0.5, minor=False)
        ax.tick_params(axis="both", which="both", direction="in")
        # remove labels for x and y axes
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Add text boxes for each channel label
        for i, label in enumerate(channel_labels):
            ax.text(
                (channel_n_bins[i] + channel_n_bins[i + 1]) / 2,
                -0.5,
                label,
                ha="center",
                va="top",
            )
            ax.text(
                -0.5,
                (channel_n_bins[i] + channel_n_bins[i + 1]) / 2,
                label,
                ha="right",
                va="center",
                rotation=90,
            )

        return ax

    def copy(self) -> "MultiChannelHistogram":
        """Create a copy of the MultiChannelHistogram object.

        Returns
        -------
        MultiChannelHistogram
            Copied MultiChannelHistogram object.
        """
        return MultiChannelHistogram.from_dict(self.to_dict())

    def __iter__(self):
        for i in range(len(self.binning)):
            yield self[i]

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

        dictionary["binning"] = MultiChannelBinning.from_dict(dictionary["binning"])
        dictionary["check_psd"] = check_psd
        return cls(**dictionary)

    @classmethod
    def from_histograms(cls, histograms: List[Union[Histogram, "MultiChannelHistogram"]]):
        """Create a MultiChannelHistogram from a list of Histograms.

        Parameters
        ----------
        binning : MultiChannelBinning
            Binning of the histogram.
        histograms : list
            List of Histograms.

        Returns
        -------
        MultiChannelHistogram
            MultiChannelHistogram object.
        """
        binning = MultiChannelBinning.join(*[h.binning for h in histograms])
        bin_counts = np.concatenate([h.nominal_values for h in histograms])
        covariance_matrix = block_diag(*[h.covariance_matrix for h in histograms])
        combined_hist = cls(binning, bin_counts, covariance_matrix=covariance_matrix)
        properties = ["label", "color", "hatch", "tex_string"]
        # For every property that is shared between all input histograms, set
        # the corresponding property of the combined histogram.
        for prop in properties:
            values = [getattr(h, prop) for h in histograms]
            if len(set(values)) == 1:
                setattr(combined_hist, prop, values[0])
        return combined_hist
