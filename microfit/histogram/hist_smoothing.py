from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

import pandas as pd
from scipy import integrate
from sklearn.neighbors import KernelDensity

from microfit.histogram import Binning, Histogram, MultiChannelBinning, MultiChannelHistogram
from microfit.statistics import covariance


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
        return dataframe[weight_column].to_numpy()

    def _integrate_kde(self, kde: KernelDensity, bin_edges: np.ndarray, points_per_bin: int = 10):
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

    def _get_smoothed_histogram(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        bins: Union[int, np.ndarray],
        density: bool = False,
        bw_method: str = "silverman",
        kernel_width_floor: float = 0.0,
        points_per_bin: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        hist, bin_edges = np.histogram(data, bins=bins, weights=weights, density=density)
        if sum(weights) == 0.0:
            return hist, bin_edges, 0.0
        nonzero_weights = weights[weights > 0]
        X = np.array(data)[weights > 0].reshape(-1, 1)
        # Because we are using old scikit-learn, we have to apply scott and silverman
        # rules by hand
        data_std = np.std(data)
        bin_width = bin_edges[1] - bin_edges[0]
        # If we just have one sample, the std is zero and the KDE calculation would fail.
        # To deal with this, we will assume that the data_std is never smaller than one
        # bin width.
        data_std = max(data_std, bin_width)
        if isinstance(bw_method, str):
            if bw_method == "scott":
                bw_method_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            elif bw_method == "silverman":
                bw_method_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))
            else:
                raise ValueError(f"Unknown bw_method {bw_method}.")
            # To deal with the case that the std is zero, add a floor
            bandwidth = max(bw_method_ * data_std, kernel_width_floor)
        else:
            bandwidth = bw_method
        kde = KernelDensity(bandwidth=bandwidth).fit(X, sample_weight=nonzero_weights)
        # To get smoothed bin counts, we want to integrate the PDF between bin edges.
        smoothed_hist = self._integrate_kde(kde, bin_edges, points_per_bin=points_per_bin)
        if not density:
            smoothed_hist *= np.sum(hist) / np.sum(smoothed_hist)
        # Ignore type of kde.bandwidth for now. We are using an older version of scikit-learn
        # that does have the attribute. In a newer version, the attribute should be kde.bandwidth_
        return smoothed_hist, bin_edges, kde.bandwidth  # type: ignore

    def _get_smoothed_histogram_bootstrap(
        self,
        dataframe: pd.DataFrame,
        binning: Union[Binning, MultiChannelBinning],
        weight_column: Optional[Union[str, List[str]]] = "weights",
        n_samples: int = 100,
        seed: int = 0,
        method: str = "poisson",
        **smooth_hist_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        return_kernel_width = smooth_hist_kwargs.pop("return_kernel_width", False)
        bw_method = smooth_hist_kwargs.pop("bw_method", "silverman")
        if isinstance(binning, Binning):
            binning = MultiChannelBinning([binning])
        selection_masks = []
        for i, query in enumerate(binning.selection_queries):
            selection_masks.append(self._get_query_mask(dataframe, query))
        data = dataframe[binning.variables].values
        weights = self.get_weights(dataframe, weight_column)
        central_value, channel_bw = self._smooth_hist_multi_channel(
            data,
            weights,
            binning,
            return_kernel_width=True,
            bw_method=bw_method,
            selection_masks=selection_masks,
            **smooth_hist_kwargs,
        )
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
                self._smooth_hist_multi_channel(
                    bootstrap_data[nonzero_mask],
                    bootstrap_weights[nonzero_mask],
                    binning=binning,
                    bw_method=channel_bw,
                    selection_masks=sample_selection_masks,
                    **smooth_hist_kwargs,
                )
            )
        bootstrap_samples = np.array(bootstrap_samples)

        covariance_matrix = covariance(
            bootstrap_samples, central_value, allow_approximation=True, tolerance=1e-10
        )
        return central_value, covariance_matrix, channel_bw

    def _smooth_hist_multi_channel(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        binning: MultiChannelBinning,
        selection_masks: Optional[Sequence[np.ndarray]] = None,
        **smooth_hist_kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
        return_kernel_width = smooth_hist_kwargs.pop("return_kernel_width", False)
        bw_method = smooth_hist_kwargs.pop("bw_method", "silverman")
        if isinstance(bw_method, str):
            bw_method = [bw_method] * len(binning)
        elif isinstance(bw_method, float):
            bw_method = [bw_method] * len(binning)
        elif isinstance(bw_method, list):
            assert len(bw_method) == len(binning)
        if selection_masks is None:
            selection_masks = [np.ones(len(data), dtype=bool)] * len(binning)
        # To make a multi-channel histogram, we need to make a histogram for each channel
        # and then concatenate them together.
        central_value = []
        channel_bw = []
        for i, channel_binning in enumerate(binning):
            channel_cv, _, kde_factor = self._get_smoothed_histogram(
                data[:, i][selection_masks[i]],
                weights[selection_masks[i]],
                bins=channel_binning.bin_edges,
                bw_method=bw_method[i],  # type: ignore
                **smooth_hist_kwargs,
            )
            central_value.append(channel_cv)
            channel_bw.append(kde_factor)
        central_value = np.concatenate(central_value)
        if return_kernel_width:
            return central_value, channel_bw
        return central_value

    def _smoothed_histogram_multi_channel(
        self,
        dataframe: pd.DataFrame,
        weight_column: Optional[Union[str, List[str]]] = None,
        **smooth_hist_kwargs,
    ) -> Union[Histogram, MultiChannelHistogram]:
        binning = self.binning
        return_single_channel = False
        if isinstance(binning, Binning):
            binning = MultiChannelBinning([binning])
            return_single_channel = True
        central_value, covariance_matrix, _ = self._get_smoothed_histogram_bootstrap(
            dataframe, binning, weight_column=weight_column, **smooth_hist_kwargs
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