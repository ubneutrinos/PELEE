from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

import pandas as pd
from scipy import integrate
from sklearn.neighbors import KernelDensity

from microfit.histogram import (
    Binning,
    Histogram,
    MultiChannelBinning,
    MultiChannelHistogram,
)
from microfit.statistics import covariance


class Transformer:
    """A base class for transformations that can be applied to data before fitting a KDE.
    
    In order for a transformation to be valid, it must be monotonous and its derivative must be
    non-zero everywhere within the support of the function.

    We can use the transformation to fit a KDE to a transformed variable :math:`Y` instead of the
    original variable :math:`X`. The KDE of :math:`Y` can then be used to estimate the PDF of
    :math:`X` using the following formula:

    .. math::
        f_X(x) = f_Y\\left(g(x)\\right) \\left| \\frac{d}{dx} \\left(g(x)\\right) \\right|

    Since the KernelDensity class returns the log of the PDF, we can use the following formula
    to get the log of the PDF of :math:`X`:

    .. math::
        \\log f_X(x) = \\log f_Y\\left(g(x)\\right) + \\log \\left| \\frac{d}{dx} \\left(g(x)\\right) \\right|

    For this reason, we implement the log of the derivative of the transformation function as
    a separate method.
    """

    def __init__(self):
        self._scale = None
        pass

    def fit(self, data):
        """Fit the transformation to the given data."""
        self._scale = 1.0
        pass

    def transform(self, x):
        """The transformation function, :math:`g(x)`."""
        return x

    def transform_log_derivative(self, x):
        return 0


class BoundTransformer(Transformer):
    def __init__(self, bounds=(None, None)):
        self.bounds = bounds
        # if both bounds are not none, use inverse logit transformation
        if all(b is not None for b in bounds):
            self.transform_func = self._transform_logit
            self.transform_log_derivative_func = self._transform_logit_log_derivative
        elif any(b is not None for b in bounds):
            self.transform_func = self._transform_logarithmic
            self.transform_log_derivative_func = self._transform_logarithmic_log_derivative
        else:
            self.transform_func = lambda x: x
            self.transform_log_derivative_func = lambda x: 0
        self._scale = None

    def _rescale_to_bounds(self, x):
        if all(b is None for b in self.bounds):
            return x, 1.0
        if self.bounds[0] is None:
            if np.any(x > self.bounds[1]):
                raise ValueError("Data is outside bounds.")
            y = -x + self.bounds[1]
            return np.clip(y, 1e-10, None), 1.0
        elif self.bounds[1] is None:
            if np.any(x < self.bounds[0]):
                raise ValueError("Data is outside bounds.")
            y = x - self.bounds[0]
            return np.clip(y, 1e-10, None), 1.0
        else:
            if np.any(x < self.bounds[0]) or np.any(x > self.bounds[1]):
                raise ValueError("Data is outside bounds.")
            y = x - self.bounds[0]
            scale = 1 / (self.bounds[1] - self.bounds[0])
            y *= scale
            return np.clip(y, 1e-10, 1 - 1e-10), scale

    def _transform_logarithmic(self, x):
        # This transformation is used when only one bound is specified.
        # We can treat lower and upper bounds by flipping data appropriately.
        assert any(b is not None for b in self.bounds)
        assert not all(b is not None for b in self.bounds)
        x, _ = self._rescale_to_bounds(x)
        return np.log(x)

    def _transform_logarithmic_log_derivative(self, x):
        assert any(b is not None for b in self.bounds)
        assert not all(b is not None for b in self.bounds)
        x, scale = self._rescale_to_bounds(x)
        return np.log(scale) - np.log(x)

    def _transform_logit(self, x):
        assert all(b is not None for b in self.bounds)
        x, _ = self._rescale_to_bounds(x)
        # Apply inverse logit transformation to map data from [0, 1] to the real line
        transformed_data = np.log(x / (1 - x))
        return transformed_data

    def _transform_logit_log_derivative(self, x):
        assert all(b is not None for b in self.bounds)
        x, scale = self._rescale_to_bounds(x)
        # derivative of the inverse logit transformation function
        return np.log(scale) - (np.log(x) + np.log(1 - x))

    def fit(self, data):
        data, self._scale = self._rescale_to_bounds(data)
        return self

    def transform(self, x):
        assert self._scale is not None, "Transformer has not been fit yet."
        return self.transform_func(x)

    def transform_log_derivative(self, x):
        assert self._scale is not None, "Transformer has not been fit yet."
        return self.transform_log_derivative_func(x)


class TransformedKDE:
    def __init__(self, transformer: Transformer, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.transformer = transformer
        assert self.transformer._scale is not None, "Transformer has not been fit yet."
        self.kde = KernelDensity(bandwidth=bandwidth)

    def fit(self, X, sample_weight=None):
        assert X.shape[1] == 1, "TransformedKDE only supports 1D data."
        transformed_data = self.transformer.transform(X[:, 0]).reshape(-1, 1)
        self.kde.fit(transformed_data, sample_weight=sample_weight)
        return self

    def score_samples(self, x):
        assert x.shape[1] == 1, "TransformedKDE only supports 1D data."
        transformed_x = self.transformer.transform(x[:, 0]).reshape(-1, 1)
        kde_values = self.kde.score_samples(transformed_x)
        kde_values += self.transformer.transform_log_derivative(x[:, 0])

        return kde_values


class SmoothHistogramMixin:
    """Mixin class for smoothing histograms.
    
    This class is mixed into the HistogramGenerator to add the functionality of
    histogram smoothing via kernel density estimation.

    This class can be used as a standalone class to smooth a histogram.
    """

    def __init__(self, binning: Union[Binning, MultiChannelBinning]) -> None:
        self.binning = binning

    def _get_query_mask(self, dataframe, query):
        """Get the boolean mask corresponding to the query.
        
        This method will be overridden by the HistogramGenerator class
        with something more sophisticated.
        """

        return np.ones(len(dataframe), dtype=bool)

    def get_weights(self, dataframe, weight_column):
        """Get the weights from the dataframe.
        
        This method will be overridden by the HistogramGenerator class
        with something more sophisticated.
        """
        if weight_column is None:
            return np.ones(len(dataframe))
        return dataframe[weight_column].to_numpy()

    def _integrate_kde(
        self,
        kde: Union[KernelDensity, TransformedKDE],
        bin_edges: np.ndarray,
        points_per_bin: int = 10,
    ):
        # To get smoothed bin counts, we want to integrate the PDF between bin edges.
        # Fortunately, the KDE already comes with that functionality.
        smoothed_hist = np.zeros(len(bin_edges) - 1)
        for i in range(len(smoothed_hist)):
            lower, upper = bin_edges[i], bin_edges[i + 1]

            def integrand(x):
                return np.exp(kde.score_samples(np.array(x).reshape(-1, 1)))

            x = np.linspace(lower, upper, points_per_bin)
            smoothed_hist[i] = integrate.trapz(integrand(x), x)
        return smoothed_hist

    def _autoselect_bound_transformation(self, data: np.ndarray, bin_edges: np.ndarray) -> str:
        """Automatically select the bound transformation to use.
        """
        bound_transformation = None
        if len(data) < 5:
            # When the number of samples is too small, we cannot reliably
            # determine the best bound transformation to use.
            return "none"
        if np.any(data >= bin_edges[-1]) and np.all(data > bin_edges[0]):
            bound_transformation = "lower"
        if np.any(data <= bin_edges[0]) and np.all(data < bin_edges[-1]):
            bound_transformation = "upper"
        if np.all(data < bin_edges[-1]) and np.all(data > bin_edges[0]):
            bound_transformation = "both"
        if np.any(data <= bin_edges[0]) and np.any(data >= bin_edges[-1]):
            bound_transformation = "none"
        assert bound_transformation in ["none", "lower", "upper", "both"]
        return bound_transformation

    def _compute_kde_histogram(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        bins: Union[int, np.ndarray],
        density: bool = False,
        bw_method: Union[str, float] = "silverman",
        points_per_bin: int = 100,
        bound_transformation: str = "none",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        assert data.ndim == 1, "KDE only supports 1D data."
        hist, bin_edges = np.histogram(data, bins=bins, weights=weights, density=density)
        # When the unsmoothed histogram has no entries, we always want to return zero. 
        # There are some variables where an "invalid value" was actually a value just outside
        # the binning range, for instance, an energy value that might be set to zero. If we
        # were to let the KDE work on these, we would allow these invalid values to contribute
        # to the smoothed histogram.
        if sum(hist) == 0.0:
            return hist, bin_edges, 0.0
        mask = weights > 0
        assert bound_transformation in ["none", "lower", "upper", "both", "auto"]
        if bound_transformation == "auto":
            bound_transformation = self._autoselect_bound_transformation(data, bin_edges)
        bounds = (bin_edges[0], bin_edges[-1])
        if bound_transformation == "both":
            # When using the bound transformer, we exclude any data that is outside the bounds
            bounds = (bin_edges[0], bin_edges[-1])
            mask = np.logical_and(mask, np.logical_and(data >= bounds[0], data <= bounds[1]))
        elif bound_transformation == "lower":
            bounds = (bin_edges[0], None)
            mask = np.logical_and(mask, data >= bin_edges[0])
        elif bound_transformation == "upper":
            bounds = (None, bin_edges[-1])
            mask = np.logical_and(mask, data <= bin_edges[-1])
        else:
            bounds = (None, None)
        nonzero_weights = weights[mask]
        X = np.array(data)[mask].reshape(-1, 1)
        # To calculate the bandwidth of the KDE, we need to transform the data
        # into the space that the KDE will be fit to.
        data_transformer = BoundTransformer(bounds=bounds).fit(X)
        data_std = np.std(data_transformer.transform(X))
        transformed_bin_edges = data_transformer.transform(bin_edges)
        # For non-linear transformations, the bin width is not constant. We take
        # the smallest to be conservative. Note that the transformed bin edges
        # may be in reversed order (when there is only an upper bound), so we take the absolute value.
        bin_width = np.min(np.abs(np.diff(transformed_bin_edges)))
        # If we just have one sample, the std is zero and the KDE calculation would fail.
        # To deal with this, we will assume that the data_std is never smaller than one
        # bin width.
        data_std = max(data_std, bin_width)
        bandwidth = bw_method
        if isinstance(bw_method, str):
            if bw_method == "scott":
                bw_method_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            elif bw_method == "silverman":
                bw_method_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))
            else:
                raise ValueError(f"Unknown bw_method {bw_method}.")
            bandwidth = bw_method_ * data_std
        assert isinstance(bandwidth, float)
        kde = TransformedKDE(data_transformer, bandwidth=bandwidth).fit(
            X, sample_weight=nonzero_weights
        )
        assert kde is not None
        # To get smoothed bin counts, we want to integrate the PDF between bin edges.
        smoothed_hist = self._integrate_kde(kde, bin_edges, points_per_bin=points_per_bin)
        if not density:
            smoothed_hist *= np.sum(nonzero_weights)
        # Ignore type of kde.bandwidth for now. We are using an older version of scikit-learn
        # that does have the attribute. In a newer version, the attribute should be kde.bandwidth_
        return smoothed_hist, bin_edges, kde.bandwidth  # type: ignore

    def _compute_kde_histogram_bootstrap(
        self,
        dataframe: pd.DataFrame,
        binning: Union[Binning, MultiChannelBinning],
        weight_column: Optional[Union[str, List[str]]] = "weights",
        n_samples: int = 100,
        seed: int = 0,
        method: str = "poisson",
        bound_transformation: str = "auto",
        calculate_covariance: bool = True,
        **smooth_hist_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Get the smoothed histogram and covariance matrix using bootstrap resampling."""

        return_kernel_width = smooth_hist_kwargs.pop("return_kernel_width", False)
        bw_method = smooth_hist_kwargs.pop("bw_method", "silverman")
        if isinstance(binning, Binning):
            binning = MultiChannelBinning([binning])
        selection_masks = []
        for i, query in enumerate(binning.selection_queries):
            selection_masks.append(self._get_query_mask(dataframe, query))
        data = dataframe[binning.variables].values
        weights = self.get_weights(dataframe, weight_column)
        if np.sum(weights) == 0:
            central_value = np.zeros(binning.n_bins)
            covariance_matrix = np.zeros((binning.n_bins, binning.n_bins))
            channel_bw = [0.0] * len(binning)
            return central_value, covariance_matrix, channel_bw
        (
            central_value,
            channel_bw,
            channel_bound_transformation,
        ) = self._compute_kde_histogram_multi_channel(
            data,
            weights,
            binning,
            return_kernel_width=True,
            bw_method=bw_method,
            selection_masks=selection_masks,
            bound_transformation=bound_transformation,
            **smooth_hist_kwargs,
        )
        if not calculate_covariance:
            covariance_matrix = np.zeros((len(central_value), len(central_value)))
            return central_value, covariance_matrix, channel_bw
        assert isinstance(channel_bw, list)
        bootstrap_samples = []
        # We set the seed once so that the bootstrap samples are reproducible,
        # and reuse the same rng object to ensure that there is no collision between
        # the bootstrap samples and the random numbers used in the smoothing.
        rng = np.random.default_rng(seed)
        for j in range(n_samples):
            bootstrap_data, bootstrap_weights = self._get_bootstrap_sample(
                data, weights, rng=rng, method=method
            )
            nonzero_mask = bootstrap_weights > 0
            if sum(nonzero_mask) == 0:
                bootstrap_samples.append(np.zeros_like(central_value))
                continue
            # The KDE bandwidth is set to be equal to the one used to create the central value
            # histogram. This ensures that the bootstrap samples are smoothed in the same way
            # and avoids pathological situations where, for instance, the bootstrap samples
            # are all using the same index and therefore the spread is zero.
            sample_selection_masks = [
                selection_masks[i][nonzero_mask] for i in range(len(selection_masks))
            ]
            bootstrap_samples.append(
                self._compute_kde_histogram_multi_channel(
                    bootstrap_data[nonzero_mask],
                    bootstrap_weights[nonzero_mask],
                    binning=binning,
                    bw_method=channel_bw,
                    selection_masks=sample_selection_masks,
                    bound_transformation=channel_bound_transformation,
                    **smooth_hist_kwargs,
                )
            )
        bootstrap_samples = np.array(bootstrap_samples)

        covariance_matrix = covariance(
            bootstrap_samples, central_value, allow_approximation=True, tolerance=1e-10
        )
        return central_value, covariance_matrix, channel_bw

    def _compute_kde_histogram_multi_channel(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        binning: MultiChannelBinning,
        selection_masks: Optional[Sequence[np.ndarray]] = None,
        bound_transformation: Optional[Union[str, List[str]]] = None,
        bw_method: Optional[Union[str, float, Sequence[Union[str, float]]]] = "silverman",
        **smooth_hist_kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[float], List[str]]]:
        """Smooth a multi-channel histogram given the data and weights.
        
        This function computes the central value for one instantiation of the data and weights.
        """

        return_kernel_width = smooth_hist_kwargs.pop("return_kernel_width", False)
        if isinstance(bw_method, str):
            bw_method_ = [bw_method] * len(binning)
        elif isinstance(bw_method, float):
            bw_method_ = [bw_method] * len(binning)
        elif isinstance(bw_method, list):
            bw_method_ = bw_method
            assert len(bw_method_) == len(binning)

        if bound_transformation is None:
            bound_transformation = ["auto"] * len(binning)
        elif isinstance(bound_transformation, str):
            bound_transformation = [bound_transformation] * len(binning)
        elif isinstance(bound_transformation, list):
            assert len(bound_transformation) == len(binning)

        if selection_masks is None:
            selection_masks = [np.ones(len(data), dtype=bool)] * len(binning)
        # To make a multi-channel histogram, we need to make a histogram for each channel
        # and then concatenate them together.
        central_value = []
        channel_bw = []
        for i, channel_binning in enumerate(binning):
            sample = data[:, i][selection_masks[i]]
            if bound_transformation[i] == "auto":
                bound_transformation[i] = self._autoselect_bound_transformation(
                    sample, channel_binning.bin_edges
                )
            channel_cv, _, kde_factor = self._compute_kde_histogram(
                data[:, i][selection_masks[i]],
                weights[selection_masks[i]],
                bins=channel_binning.bin_edges,
                bw_method=bw_method_[i],  # type: ignore
                bound_transformation=bound_transformation[i],
                **smooth_hist_kwargs,
            )
            central_value.append(channel_cv)
            channel_bw.append(kde_factor)
        central_value = np.concatenate(central_value)
        if return_kernel_width:
            return central_value, channel_bw, bound_transformation
        return central_value

    def _smoothed_histogram_multi_channel(
        self,
        dataframe: pd.DataFrame,
        weight_column: Optional[Union[str, List[str]]] = None,
        calculate_covariance: bool = True,
        **smooth_hist_kwargs,
    ) -> Union[Histogram, MultiChannelHistogram]:
        """This function is actually called by the HistogramGenerator class."""
        binning = self.binning
        return_single_channel = False
        if isinstance(binning, Binning):
            binning = MultiChannelBinning([binning])
            return_single_channel = True
        central_value, covariance_matrix, _ = self._compute_kde_histogram_bootstrap(
            dataframe, binning, weight_column=weight_column, calculate_covariance=calculate_covariance, **smooth_hist_kwargs
        )
        if return_single_channel:
            return Histogram(
                binning.binnings[0], central_value, covariance_matrix=covariance_matrix,
            )
        return MultiChannelHistogram(binning, central_value, covariance_matrix=covariance_matrix,)

    def _get_bootstrap_sample(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        method: str = "poisson",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Initialize a new random number generator
        if rng is None:
            rng = np.random.default_rng()
        bootstrap_data = None
        bootstrap_weights = None
        # Randomly sample the data with replacement
        if method == "choice":
            bootstrap_indices = rng.choice(range(len(data)), size=len(data))
            bootstrap_data = data[bootstrap_indices]
            bootstrap_weights = weights[bootstrap_indices]
        elif method == "poisson":
            bootstrap_weights = weights * np.random.poisson(1.0, size=len(weights))
            bootstrap_data = data
        else:
            raise ValueError(f"Unknown bootstrap method {method}.")
        return bootstrap_data, bootstrap_weights
