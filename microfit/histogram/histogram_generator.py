from copy import deepcopy
import hashlib
import logging
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union, cast, overload

# Import Literal for Python <= 3.7
try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal
import warnings

import numpy as np
from microfit.fileio import from_json
from microfit.parameters import ParameterSet
from microfit.histogram import Binning, MultiChannelBinning

# Need to break potential import loops
from microfit.histogram.histogram import Histogram, MultiChannelHistogram
from microfit.histogram import SmoothHistogramMixin

import pandas as pd


from microfit.statistics import (
    fronebius_nearest_psd,
    covariance,
    is_psd,
    sideband_constraint_correction,
)

from data_loading import detector_variations


class HistogramGenerator(SmoothHistogramMixin):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        binning: Union[Binning, MultiChannelBinning],
        parameters: ParameterSet = ParameterSet([]),
        detvar_data: Optional[Union[Dict[str, AnyStr], str]] = None,
        enable_cache: bool = True,
        cache_total_covariance: bool = True,
        extra_mc_covariance: Optional[np.ndarray] = None,
        extra_background_fractional_error: Optional[Dict[str, float]] = None,
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
        detvar_data : dict or str, optional
            Dictionary containing the detector variations.
        extra_mc_covariance : np.ndarray, optional
            Extra covariance matrix to be added to the total covariance matrix.
        extra_background_fractional_error: dict, optional
            Dictionary where keys are the selection strings for the background and values are the
            fractional error to be applied to that background.
        """
        self.dataframe = dataframe
        self.parameters = parameters
        # Hardcode this as default because currently there is no situation where it would be
        # anything else. The only exceptions are when we calculate systematics, but even
        # then we also hard-coded which column must be used when.
        self.weight_column = "weights"
        self.parameters_last_evaluated = None  # used to invalidate the cache
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Creating histogram generator for with binning: {binning}")
        self.binning = binning
        self.enable_cache = enable_cache
        self.cache_total_covariance = cache_total_covariance
        self.detvar_data = detvar_data
        self.extra_mc_covariance = extra_mc_covariance
        if extra_background_fractional_error is not None:
            if not isinstance(extra_background_fractional_error, dict):
                raise ValueError("extra_background_fractional_error must be a dictionary.")
            for k, v in extra_background_fractional_error.items():
                if not isinstance(v, (float)):
                    raise ValueError(
                        f"Value for {k} in extra_background_fractional_error must be a number."
                    )
        self.extra_background_fractional_error = extra_background_fractional_error
        # in case a string was passed to detvar_data, we load it from the file
        if isinstance(self.detvar_data, str):
            self.detvar_data = from_json(self.detvar_data)
        # if isinstance(self.detvar_data):
        #    self.detvar_data = detvar_data

        # check that binning matches to detvar_data
        if self.detvar_data is not None:
            detvar_binning = self.detvar_data["binning"]
            assert isinstance(
                detvar_binning, (Binning, MultiChannelBinning)
            ), "Binning of detector variations must be a Binning or MultiChannelBinning object."
            # Just check the bin edges and variable rather than the entire binning object
            if not detvar_binning.is_compatible(self.binning):
                raise ValueError(
                    "Binning of detector variations does not match binning of main histogram."
                )

        self._invalidate_cache()

    def _generate_hash(self, *args, **kwargs) -> str:
        hash_obj = hashlib.md5()
        data = str(args) + str(kwargs)
        hash_obj.update(data.encode("utf-8"))
        return hash_obj.hexdigest()

    def _invalidate_cache(self):
        """Invalidate the cache."""
        self.hist_cache: Dict[str, Histogram] = dict()
        self.unisim_hist_cache = dict()
        self.multisim_hist_cache = dict()
        self.multisim_hist_cache["weightsReint"] = dict()
        self.multisim_hist_cache["weightsFlux"] = dict()
        self.multisim_hist_cache["weightsGenie"] = dict()
        self.parameters_last_evaluated = self.parameters.copy()

    def _return_empty_hist(self):
        """Return an empty histogram."""
        if isinstance(self.binning, MultiChannelBinning):
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
        mask = dataframe.eval(query, engine="python")
        assert len(mask) == len(dataframe)
        return mask

    def _histogram_multi_channel(
        self,
        dataframe: pd.DataFrame,
        weight_column: Optional[Union[str, List[str]]] = None,
    ) -> Union[Histogram, MultiChannelHistogram]:
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
        histogram : Histogram or MultiChannelHistogram
            Histogram of the data.
        """

        binning = self.binning.copy()
        return_single_channel = False
        # If we were to check for "Binning", this would also return True for MultiChannelBinning
        if not isinstance(binning, MultiChannelBinning):
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
                np.histogram(channel_sample, bins=bin_edges[i], weights=channel_weights**2)[0]
            )
        # now we build the covariance
        covariance_matrix = np.diag(np.concatenate(channel_bin_variances))
        # We are also filling a 2D array for the bin counts, where off-diagonal elements
        # represent the number of events that are shared between two bins.
        bin_counts_2d = np.diag(np.concatenate(channel_bin_counts))
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
                # To fill the 2D bin counts, we do the same thing but we do not square the weights
                hist, _, _ = np.histogram2d(
                    sample[mask, i],
                    sample[mask, j],
                    bins=[bin_edges[i], bin_edges[j]],
                    weights=weights[mask],
                )
                bin_counts_2d[
                    np.ix_(binning._channel_bin_idx(i), binning._channel_bin_idx(j))
                ] = hist
                bin_counts_2d[
                    np.ix_(binning._channel_bin_idx(j), binning._channel_bin_idx(i))
                ] = hist.T

        covariance_matrix, dist = fronebius_nearest_psd(covariance_matrix, return_distance=True)
        if dist > 1e-3:
            raise RuntimeError(f"Nearest PSD distance is {dist} away, which is too large.")
        if return_single_channel:
            return Histogram(
                binning.binnings[0],
                bin_counts_2d,
                covariance_matrix=covariance_matrix,
            )
        return MultiChannelHistogram(
            binning,
            bin_counts_2d,
            covariance_matrix=covariance_matrix,
        )

    def _multi_channel_universes(
        self,
        dataframe: pd.DataFrame,
        base_weight_column: str,
        multisim_weight_column: str,
        weight_rescale: float = 1 / 1000.0,
    ) -> np.ndarray:
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
        bin_counts : np.ndarray
            Array of shape (n_universes, n_bins) containing the bin counts for each universe.
        """

        binning = self.binning
        if not isinstance(binning, MultiChannelBinning):
            binning = MultiChannelBinning([binning])
        sample = dataframe[binning.variables].to_numpy()
        selection_masks = []
        for i, query in enumerate(binning.selection_queries):
            selection_masks.append(self._get_query_mask(dataframe, query))
        bin_edges = binning.consecutive_bin_edges
        base_weights = self.get_weights(dataframe, base_weight_column)
        multisim_weights: np.ndarray = dataframe[multisim_weight_column].to_numpy()
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
            universe_weights = self._limit_weights(df[column].to_numpy() * weight_rescale)
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
        include_multisim_errors: bool = False,
        use_sideband: bool = False,
        extra_query: Optional[str] = None,
        sideband_generator: Optional["HistogramGenerator"] = None,
        sideband_total_prediction: Optional[Histogram] = None,
        sideband_observed_hist: Optional[Histogram] = None,
        add_precomputed_detsys: bool = False,
        smooth_detsys_variations: bool = True,
        include_detsys_variations: Optional[List[str]] = None,
        detsys_only_diagonal: bool = False,
        use_kde_smoothing: bool = False,
        options: Dict[str, Any] = {},
    ) -> Union[Histogram, MultiChannelHistogram]:
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
        use_kde_smoothing : bool, optional
            Whether to use KDE smoothing to estimate the bin counts. This is useful for
            histograms with few events per bin.
        options : dict, optional
            Additional options that depend on the specific implementation of the histogram
            generator. If `use_kde_smoothing` is True, options are passed as keyword arguments
            to `_smoothed_histogram_multi_channel`.

        Returns
        -------
        histogram : Histogram
            Histogram object containing the binned data.
        """

        if use_kde_smoothing:
            self.logger.debug("Using KDE smoothing.")
            # The KDE smoothing option is incompatible with sidebands and multisim errors
            assert (
                not include_multisim_errors
            ), "KDE smoothing is incompatible with multisim errors."
            assert not use_sideband, "KDE smoothing is incompatible with sidebands."

        if use_sideband:
            # Issue deprecation warning: In the future, passing a sideband generator will no
            # longer be supported. Instead, one should use the "update_with_measurement"
            # method of the MultiChannelHistogram which is the output of this function.
            warnings.warn(
                "Passing a sideband generator will be deprecated in the future. Instead, use the "
                "update_with_measurement method of the MultiChannelHistogram which is the output of "
                "this function.",
                DeprecationWarning,
            )
            assert sideband_generator is not None
            assert sideband_total_prediction is not None
            assert sideband_observed_hist is not None

        if add_precomputed_detsys:
            assert self.detvar_data is not None, "No detector variations provided."

        calculate_hist = True
        hash = self._generate_hash(
            extra_query,
            add_precomputed_detsys,
            use_sideband,
            include_multisim_errors,
            use_kde_smoothing,
            options,
            smooth_detsys_variations,
            include_detsys_variations,
            detsys_only_diagonal,
        )
        hist = None
        if self.enable_cache:
            if self.parameters != self.parameters_last_evaluated:
                self.logger.debug("Parameters changed, invalidating cache.")
                self._invalidate_cache()
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
                self.logger.debug("Query returned no events, returning empty histogram.")
                hist = self._return_empty_hist()
                if self.enable_cache:
                    self.hist_cache[hash] = hist.copy()
                return hist
            if use_kde_smoothing:
                hist = self._smoothed_histogram_multi_channel(dataframe, **options)
            else:
                hist = self._histogram_multi_channel(dataframe)
            if self.enable_cache:
                self.hist_cache[hash] = hist.copy()
        # if we reach this point without having a histogram, something went wrong
        assert hist is not None
        self.logger.debug(f"Generated histogram: {hist}")
        if include_multisim_errors:
            self.logger.debug("Calculating multisim uncertainties")
            extended_cov = None
            if use_sideband:
                # initialize extended covariance matrix
                n_bins = hist.n_bins
                assert sideband_observed_hist is not None
                sb_n_bins = sideband_observed_hist.n_bins
                extended_cov = np.zeros((n_bins + sb_n_bins, n_bins + sb_n_bins))

            # calculate multisim histograms
            for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
                cov_mat = self.calculate_multisim_uncertainties(
                    ms_column,
                    extra_query=extra_query,
                    return_histograms=False,
                )
                hist.add_covariance(cov_mat)

                if use_sideband:
                    extended_cov += self.multiband_covariance(
                        [self, sideband_generator],
                        ms_column,
                        extra_queries=[extra_query, None],
                    )

            # calculate unisim histograms
            self.logger.debug("Calculating unisim uncertainties")
            cov_mat_unisim = self.calculate_unisim_uncertainties(
                central_value_hist=hist, extra_query=extra_query
            )
            hist.add_covariance(cov_mat_unisim)

            if use_sideband:
                # calculate constraint correction
                assert sideband_total_prediction is not None
                assert sideband_observed_hist is not None
                mu_offset, cov_corr = sideband_constraint_correction(
                    sideband_measurement=sideband_observed_hist.bin_counts,
                    sideband_central_value=sideband_total_prediction.bin_counts,
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
            if include_detsys_variations is None:
                include_detsys_variations = detector_variations
            det_cov = self.calculate_detector_covariance(
                smooth_variations=smooth_detsys_variations,
                extra_query=extra_query,
                include_variations=include_detsys_variations,
                only_diagonal=detsys_only_diagonal,
            )
            if det_cov is not None:
                hist.add_covariance(det_cov)

        if self.extra_mc_covariance is not None:
            hist.add_covariance(self.extra_mc_covariance)

        if self.enable_cache and self.cache_total_covariance:
            self.hist_cache[hash] = hist.copy()
        return hist

    @classmethod
    def generate_joint_histogram(
        cls,
        hist_generators,
        include_multisim_errors=True,
        ms_columns=["weightsGenie", "weightsFlux", "weightsReint"],
        include_unisim_errors=True,
        include_stat_errors=True,
        extra_query=None,
        add_precomputed_detsys=False,
        detsys_only_diagonal=False,
        smooth_detsys_variations=True,
        include_detsys_variations=detector_variations,
    ):
        """Generate a joint histogram from multiple histogram generators.

        The result is a MultiChannelHistogram object with covariance matrix that contains
        correlations between bins of different channels.
        """

        generate_kwargs = {
            "include_multisim_errors": False,
            "extra_query": extra_query,
        }
        # We want these histograms to only contain the statistical covariance matrix (which may include correlations
        # in case of overlapping selections between channels).
        # The resulting covariance is block-diagonal, with each block corresponding to the output of one histogram
        # generator.

        histogram = MultiChannelHistogram.from_histograms(
            [h.generate(**generate_kwargs) for h in hist_generators]
        )

        if not include_stat_errors:
            histogram.covariance_matrix = np.zeros_like(histogram.covariance_matrix)
        covariance_matrix = np.zeros((histogram.n_bins, histogram.n_bins))
        concatenated_cv = cls.multiband_cv(
            hist_generators, extra_queries=[extra_query] * len(hist_generators)
        )
        if include_multisim_errors:
            for ms_column in ms_columns:
                covariance_matrix += HistogramGenerator.multiband_covariance(
                    hist_generators,
                    ms_column,
                    concatenated_cv=concatenated_cv,
                    extra_queries=[extra_query] * len(hist_generators),
                )
        # We never include the unisim errors without also using the multisim errors
        if include_unisim_errors and include_multisim_errors:
            covariance_matrix += HistogramGenerator.multiband_unisim_covariance(
                hist_generators,
                concatenated_cv=concatenated_cv,
                extra_queries=[extra_query] * len(hist_generators),
            )

        if add_precomputed_detsys:
            covariance_matrix += HistogramGenerator.multiband_detector_covariance(
                hist_generators,
                smooth_variations=smooth_detsys_variations,
                include_variations=include_detsys_variations,
                extra_queries=[extra_query] * len(hist_generators),
                only_diagonal=detsys_only_diagonal,
            )

        histogram.add_covariance(covariance_matrix)

        return histogram

    @classmethod
    def multiband_covariance(
        cls, hist_generators, ms_column, extra_queries=None, concatenated_cv=None
    ):
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
        concatenated_cv : array_like, optional
            Concatenated central values. If not given, they are calculated from the histograms.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        # types = [type(hg) for hg in hist_generators]
        # assert all(isinstance(hg, cls) for hg in hist_generators), f"Must provide a list of HistogramGenerator objects. Types are {types}."

        universe_hists = []
        if extra_queries is None or isinstance(extra_queries, str):
            extra_queries = [extra_queries] * len(hist_generators)
        assert isinstance(extra_queries, list)
        for hg, extra_query in zip(hist_generators, extra_queries):
            cov_mat, universe_hist = hg.calculate_multisim_uncertainties(
                ms_column, return_histograms=True, extra_query=extra_query
            )
            universe_hists.append(universe_hist)
        if concatenated_cv is None:
            concatenated_cv = cls.multiband_cv(hist_generators, extra_queries=extra_queries)
        concatenated_universes = np.concatenate(universe_hists, axis=1)
        cov_mat = covariance(
            concatenated_universes,
            concatenated_cv,
            allow_approximation=True,
            tolerance=1e-8,
        )
        return cov_mat

    @classmethod
    def multiband_cv(cls, hist_generators, extra_queries=None) -> np.ndarray:
        """Calculate the central values for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the central values of the
        multisim universes. The underlying assumption
        is that the weights listed in the multisim column are from the same universes
        in the same order for all histograms.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        extra_queries : list of str, optional
            List of additional queries to apply to the dataframe.

        Returns
        -------
        concatenated_cv : array_like
            Concatenated central values.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        # types = [type(hg) for hg in hist_generators]
        # assert all(isinstance(hg, cls) for hg in hist_generators), f"Must provide a list of HistogramGenerator objects. Types are {types}."

        central_values = []
        if extra_queries is None or isinstance(extra_queries, str):
            extra_queries = [extra_queries] * len(hist_generators)
        assert isinstance(extra_queries, list)
        for hg, extra_query in zip(hist_generators, extra_queries):
            central_values.append(
                hg.generate(extra_query=extra_query, include_multisim_errors=False).bin_counts
            )

        concatenated_cv = np.concatenate(central_values)
        return concatenated_cv

    @classmethod
    def multiband_unisim_covariance(
        cls,
        hist_generators: List["HistogramGenerator"],
        extra_queries=None,
        concatenated_cv: Optional[np.ndarray] = None,
    ):
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
        concatenated_cv : array_like, optional
            Concatenated central values. If not given, they are calculated from the histograms.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        universe_hist_dicts = []
        if extra_queries is None or isinstance(extra_queries, str):
            extra_queries = [extra_queries] * len(hist_generators)
        assert isinstance(extra_queries, list)
        for hg, extra_query in zip(hist_generators, extra_queries):
            universe_hist_dicts.append(
                hg.calculate_unisim_uncertainties(
                    return_histograms=True, extra_query=extra_query, skip_covariance=True
                )[1]
            )
        if concatenated_cv is None:
            concatenated_cv = cls.multiband_cv(hist_generators, extra_queries=extra_queries)
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
                concatenated_universes,
                concatenated_cv,
                allow_approximation=True,
                tolerance=1e-8,
            )
            summed_cov_mat += cov_mat
        return summed_cov_mat

    @classmethod
    def multiband_detector_covariance(
        cls,
        hist_generators: List["HistogramGenerator"],
        smooth_variations=True,
        include_variations=detector_variations,
        extra_queries=None,
        only_diagonal=False,
    ):
        """Calculate the covariance matrix for multiple histograms.

        Given a list of HistogramGenerator objects, calculate the covariance matrix of the
        detector variations.

        Parameters
        ----------
        hist_generators : list of HistogramGenerator
            List of HistogramGenerator objects.
        """

        assert len(hist_generators) > 0, "Must provide at least one histogram generator."
        assert "cv" in include_variations, "Must include the central value variation."

        total_bins = sum([hg.binning.n_bins for hg in hist_generators])
        summed_cov_mat = np.zeros((total_bins, total_bins))
        if extra_queries is None or isinstance(extra_queries, str):
            extra_queries = [extra_queries] * len(hist_generators)
        assert isinstance(extra_queries, list)
        variation_diffs_dict = {variation: np.array([]) for variation in include_variations}
        extra_background_variances = np.array([])
        for hg, extra_query in zip(hist_generators, extra_queries):
            detvar_output = hg.calculate_detector_covariance(
                return_histograms=True,
                smooth_variations=smooth_variations,
                include_variations=include_variations,
                extra_query=extra_query,
                only_diagonal=only_diagonal,
            )
            if detvar_output is None:
                cov_mat = np.zeros((hg.binning.n_bins, hg.binning.n_bins))
                variation_diffs = {
                    variation: np.zeros(hg.binning.n_bins) for variation in include_variations
                }
            else:
                cov_mat, variation_diffs = detvar_output

            for variation in include_variations:
                variation_diffs_dict[variation] = np.concatenate(
                    [variation_diffs_dict[variation], variation_diffs[variation]]
                )

            extra_background_variances = np.concatenate(
                [extra_background_variances, hg.extra_background_variance(extra_query=extra_query)]
            )

        for variation in include_variations:
            if variation == "cv":
                continue
            h = variation_diffs_dict[variation]
            summed_cov_mat += covariance(
                h.reshape(1, -1),
                np.zeros(total_bins),
                allow_approximation=True,
                tolerance=1e-10,
            )

        summed_cov_mat += np.diag(extra_background_variances)

        # We have to do this here again, because the original covariance matrix from the 
        # output of the function call earlier is ignored and instead this function uses only
        # the variations.
        if only_diagonal:
            summed_cov_mat = np.diag(np.diag(summed_cov_mat))

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

    @overload
    def calculate_multisim_uncertainties(
        self,
        multisim_weight_column: str,
        weight_rescale: float = ...,
        weight_column: Optional[str] = ...,
        central_value_hist: Optional[Union[Histogram, MultiChannelHistogram]] = ...,
        extra_query: Optional[str] = ...,
        return_histograms: Literal[False] = ...,
    ) -> np.ndarray:
        ...

    @overload
    def calculate_multisim_uncertainties(
        self,
        multisim_weight_column: str,
        weight_rescale: float = ...,
        weight_column: Optional[str] = ...,
        central_value_hist: Optional[Union[Histogram, MultiChannelHistogram]] = ...,
        extra_query: Optional[str] = ...,
        return_histograms: Literal[True] = ...,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ...

    def calculate_multisim_uncertainties(
        self,
        multisim_weight_column: str,
        weight_rescale: float = 1 / 1000,
        weight_column: Optional[str] = None,
        central_value_hist: Optional[Union[Histogram, MultiChannelHistogram]] = None,
        extra_query: Optional[str] = None,
        return_histograms: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
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
        hash = None
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
        if len(dataframe) == 0:
            cov = np.zeros((self.binning.n_bins, self.binning.n_bins))
            if return_histograms:
                return cov, None
            else:
                return cov
        universe_histograms = self._multi_channel_universes(
            dataframe,
            weight_column,
            multisim_weight_column,
            weight_rescale=weight_rescale,
        )
        if central_value_hist is None:
            central_value_hist = self._histogram_multi_channel(dataframe)
        # calculate the covariance matrix from the histograms
        cov = covariance(
            universe_histograms,
            central_value_hist.bin_counts,
            allow_approximation=True,
            tolerance=1e-8,
        )
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
        self,
        central_value_hist: Optional[Histogram] = None,
        extra_query: Optional[str] = None,
        return_histograms: bool = False,
        skip_covariance: bool = False,
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
        skip_covariance : bool, optional
            If True, only return the histograms, not the covariance matrix. Can be used within
            a function that calculates joint unisim and multisim uncertainties to speed it up.

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
                    ).bin_counts
                    observations.append(bincounts)
                observations = np.array(observations)
                if self.enable_cache:
                    self.unisim_hist_cache[hash][knob] = observations
                    assert central_value_hist is not None
                    self.unisim_hist_cache[hash]["central_value"] = central_value_hist
            observation_dict[knob] = observations
            # If we get to this point without having either calculated a central value hist
            # or taken one from the cache, something is wrong
            assert central_value_hist is not None
            if skip_covariance:
                continue
            # calculate the covariance matrix from the histograms
            cov = covariance(
                observations,
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
            return total_cov, observation_dict
        return total_cov

    @overload
    def calculate_detector_covariance(
        self,
        only_diagonal: bool = ...,
        fractional: bool = ...,
        smooth_variations: bool = True,
        include_variations: Optional[List[str]] = None,
        extra_query: Optional[str] = None,
        return_histograms: Literal[False] = ...,
    ) -> Optional[np.ndarray]:
        ...

    @overload
    def calculate_detector_covariance(
        self,
        only_diagonal: bool = ...,
        fractional: bool = ...,
        smooth_variations: bool = True,
        include_variations: Optional[List[str]] = None,
        extra_query: Optional[str] = None,
        return_histograms: Literal[True] = ...,
    ) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        ...

    def calculate_detector_covariance(
        self,
        only_diagonal: bool = False,
        fractional: bool = True,
        smooth_variations: bool = True,
        include_variations: Optional[List[str]] = None,
        extra_query: Optional[str] = None,
        return_histograms: bool = False,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]]:
        """Get the covariance matrix for detector uncertainties.

        Parameters
        ----------
        only_diagonal : bool, optional
            If True, only return the diagonal of the covariance matrix.
        fractional : bool, optional
            If True, rescale the detector variations to the central value of this histogram generator.
        smooth_variations : bool, optional
            If True, apply a smoothing filter to the variations.
        include_variations : list of str, optional
            List of variations to include in the covariance matrix. The default is to include all
            variations.
        extra_query : str, optional
            Extra query to apply to the dataframe before calculating the covariance matrix. This is only
            relevant when `fractional` is set to `True`, as the extra query is used to calculate the
            central value for the re-scaling.

        Returns
        -------
        covariance_matrix : array_like
            Covariance matrix of the bin counts.

        Notes
        -----
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
        # If we are at this point and somehow didn't get a dict, we must have forgotten to load it
        # from json in the constructor.
        assert isinstance(self.detvar_data, dict)

        variation_hist_data = cast(
            Dict[str, Dict[str, Histogram]], deepcopy(self.detvar_data["variation_hist_data"])
        )
        cov_mat = np.zeros((self.binning.n_bins, self.binning.n_bins), dtype=float)

        filter = np.array([0.1, 0.3, 1.0, 0.3, 0.1])
        filter /= np.sum(filter)

        def smooth_bin_counts(arr, filter):
            """Filter the array with the given filter, reflecting the array at the edges."""
            # pad the array with zeros
            arr_padded = np.pad(arr, (len(filter) // 2, len(filter) // 2), mode="edge")
            # apply the filter
            arr_filtered = np.convolve(arr_padded, filter, mode="valid")
            return arr_filtered
        
        def extract_channels(hist):
            # Extract the channels that are relevant for this histogram generator.
            # This step also ensures that the order of the channels is correct.
            if isinstance(hist, MultiChannelHistogram):
                return hist[self.binning.channels]
            return hist
        
        def filter_hist(hist):
            if not smooth_variations:
                return extract_channels(hist)

            if isinstance(hist, MultiChannelHistogram):
                # We have to apply the filter separately for each channel, otherwise we would
                # have unphysical correlations between the bins of adjacent channels.
                return extract_channels(MultiChannelHistogram.from_histograms(
                    [filter_hist(h) for h in  hist]
                ))
            else:
                hist.bin_counts = smooth_bin_counts(hist.bin_counts, filter)
            return extract_channels(hist)

        # Get the CV variation hist
        variation_cv_hist = np.zeros(self.binning.n_bins)
        if include_variations is None:
            include_variations = detector_variations
        variation_hists = {v: np.zeros(self.binning.n_bins) for v in include_variations}
        for dataset in cast(str, self.detvar_data["mc_sets"]):
            variation_cv_hist = np.add(
                variation_cv_hist, filter_hist(variation_hist_data[dataset]["cv"]).bin_counts
            )
            for v in include_variations:
                variation_hists[v] = np.add(
                    variation_hists[v], filter_hist(variation_hist_data[dataset][v]).bin_counts
                )

        variation_diffs: Dict[str, np.ndarray] = {
            v: (h - variation_cv_hist) for v, h in variation_hists.items()
        }

        if fractional:
            # The detector systematics might have been calculated for a specific selection
            # of events.
            detvar_selection_query = self.detvar_data["extra_selection_query"]
            if detvar_selection_query is not None:
                assert isinstance(detvar_selection_query, str)
                if extra_query is not None:
                    combined_extra_query = f"({extra_query}) and ({detvar_selection_query})"
                else:
                    combined_extra_query = detvar_selection_query
            else:
                combined_extra_query = extra_query
            this_histgen_cv = self.generate(extra_query=combined_extra_query)
            for v in variation_diffs.keys():
                with np.errstate(divide="ignore", invalid="ignore"):
                    variation_diffs[v] *= this_histgen_cv.bin_counts / variation_cv_hist

        for v, h in variation_diffs.items():
            if v == "cv":
                continue
            h[~np.isfinite(h)] = 0.0

            # We have just one observation and the central value is zero since it was already subtracted
            cov_mat += covariance(
                h.reshape(1, -1),
                central_value=np.zeros_like(h),
                # As with all unisim variations, small deviations from the PSD case are expected
                allow_approximation=True,
                tolerance=1e-10,
                # tolerance=1e-8,
                debug_name=f"detector_{v}",
            )

        if only_diagonal:
            cov_mat = np.diag(np.diag(cov_mat))

        cov_mat += np.diag(self.extra_background_variance(extra_query=extra_query))

        if return_histograms:
            return cov_mat, variation_diffs
        return cov_mat

    def extra_background_variance(self, extra_query: Optional[str] = None) -> np.ndarray:
        """Calculate the variance (squared error) vector for extra background uncertainties.

        These are a flat fractional error that we add to low-stats background.
        """
        output = np.zeros(self.binning.n_bins)
        if self.extra_background_fractional_error is not None:
            for k, v in self.extra_background_fractional_error.items():
                if k is None:
                    continue
                if v == 0.0:
                    continue
                assert isinstance(k, str)
                # Here, each key is the selection query for the background
                # and the value is the fractional error to be applied.

                # If the extra_query to this function is not None, we add
                # the background query to it
                backgr_query = extra_query
                if backgr_query is not None:
                    backgr_query += f" and {k}"
                else:
                    backgr_query = k
                backgr_hist = self.generate(extra_query=backgr_query)
                output += (backgr_hist.bin_counts * v) ** 2
        return output

    def _resync_parameters(self):
        """Not needed since there are no parameters in this class."""
        pass

    def _check_shared_parameters(self, parameters):
        return True
