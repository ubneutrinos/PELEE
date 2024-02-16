"""Module containing different signal generators. Every class defined in this file is available 
as a signal generator when defining an analysis in a toml file.
"""


from typing import Optional, Tuple, Union
import numpy as np
from copy import deepcopy

import pandas as pd
from microfit.histogram import (
    Binning,
    HistogramGenerator,
    RunHistGenerator,
    MultiChannelBinning,
    MultiChannelHistogram,
)
from microfit.parameters import ParameterSet


class SignalOverBackgroundGenerator(HistogramGenerator):
    hist_gen_cls = HistogramGenerator

    def __init__(
        self,
        dataframe: pd.DataFrame,
        binning: Union[Binning, MultiChannelBinning],
        signal_query="category == 111",
        background_query="category != 111",
        parameters: ParameterSet = ParameterSet([]),
        **kwargs,
    ):
        self.signal_query = signal_query
        self.background_query = background_query
        self.parameters = parameters
        assert "signal_strength" in parameters.names, "signal_strength must be in parameters"
        # We want to forward all parameters except the signal strength to the histogram generator
        pnames = parameters.names
        pnames.remove("signal_strength")
        forward_parameters = parameters[pnames]
        # We split all channels into signal and background components. This method ensures
        # that the covariance between signal and background is correctly calculated.
        self.channels = binning.channels
        assert len(self.channels) > 0, "Binning must contain at least one channel"
        self.binning = binning
        sob_binning = self.split_binning(binning).copy()

        self.extra_mc_covariance = kwargs.pop('extra_mc_covariance',None)
        #print(self.extra_mc_covariance)

        self.add_precomputed_detsys = kwargs.pop('add_precomputed_detsys',False) 
        self.detvar_data = kwargs.pop('detvar_data',None)

        self.hist_generator = self.hist_gen_cls(
            dataframe, sob_binning, parameters=forward_parameters, **kwargs
        )

    def split_binning(self, binning):
        """Split a binning into signal and background channels."""

        def split_channel(channel: Binning):
            signal_channel = channel.copy()
            background_channel = channel.copy()
            signal_channel.selection_query = self.append_query(
                channel.selection_query, self.signal_query
            )
            signal_channel.label = f"{channel.label}__SIGNAL"
            background_channel.selection_query = self.append_query(
                channel.selection_query, self.background_query
            )
            background_channel.label = f"{channel.label}__BACKGROUND"
            return signal_channel, background_channel

        if isinstance(binning, Binning):
            return MultiChannelBinning(list(split_channel(binning)))
        binnings = []
        for channel in binning:
            binnings.extend(split_channel(channel))
        return MultiChannelBinning(binnings)

    def append_query(self, query, extra_query):
        if query is None:
            return extra_query
        return query + " and " + extra_query

    def generate(self, **kwargs):

        add_precomputed_detsys = kwargs.pop('add_precomputed_detsys',False)
        hist = self.hist_generator.generate(**kwargs)
        # It will have to be a MultiChannelHistogram because we are using a split-channel binning.
        assert isinstance(hist, MultiChannelHistogram), "Histogram must be a MultiChannelHistogram"
        for ch in self.channels:
            # Scaling a channel will also scale the covariance matrix by the factor squared,
            # and update the correlations of that channel with all other channels by the
            # factor.
            hist.scale_channel(f"{ch}__SIGNAL", self.parameters["signal_strength"].m)
            hist.sum_channels(
                [f"{ch}__SIGNAL", f"{ch}__BACKGROUND"], ch, replace=True, inplace=True
            )
        if not isinstance(self.binning, MultiChannelBinning):
            assert self.binning.label is not None
            hist = hist[self.binning.label]
            hist.binning.selection_query = self.binning.selection_query
        else:
            for ch in self.channels:
                hist[ch].binning.selection_query = self.binning[ch].selection_query
        # After summing signal and background, we should now have a histogram with the same binning
        # as was originally requested when this generator was initialized.
        assert (
            hist.binning == self.binning
        ), "Binning of the generated histogram does not match the binning of the generator"

        # Add the covariance for the detector systematics down here
        if self.detvar_data is not None and add_precomputed_detsys: 
            hist.add_covariance(self.calculate_detector_covariance())

        return hist

    def _resync_parameters(self):
        self.parameters.synchronize(self.hist_generator.parameters)

    def calculate_multisim_uncertainties(
        self, *args, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        return_histograms = kwargs.get("return_histograms", False)
        result = self.hist_generator.calculate_multisim_uncertainties(*args, **kwargs)
        summed_cov = None
        if not return_histograms:
            assert isinstance(result, np.ndarray), "Result must be a numpy array"
            cov_mat = result
        else:
            cov_mat = result[0]
        # Use the mechanics of the MultiChannelHistogram to correctly scale and add the channels.
        assert isinstance(
            self.hist_generator.binning, MultiChannelBinning
        ), "Binning must be a MultiChannelBinning"
        ms_hist = MultiChannelHistogram(
            self.hist_generator.binning.copy(),
            # The bin counts don't matter for the covariance matrix, so we just fill it with ones.
            bin_counts=np.ones(self.hist_generator.binning.n_bins),
            covariance_matrix=cov_mat,
        )
        for ch in self.channels:
            ms_hist.scale_channel(f"{ch}__SIGNAL", self.parameters["signal_strength"].m)
            ms_hist.sum_channels(
                [f"{ch}__SIGNAL", f"{ch}__BACKGROUND"], ch, replace=True, inplace=True
            )
        if not return_histograms:
            return ms_hist.covariance_matrix
        else:
            summed_cov = ms_hist.covariance_matrix
        # If we get here, then the result should be a tuple where the second element contains
        # the bin counts of the universes.
        universes = result[1]
        # If we get here, both signal and background have universes.
        assert isinstance(universes, np.ndarray), "Universes must be numpy arrays"
        # The universes will be concatenated bin counts of the signal and background channels.
        # We have to extract the universes for each channel and sum them.
        # The shape of the array should be (n_universes, n_bins)
        assert (
            universes.shape[1] == self.hist_generator.binning.n_bins
        ), "Universes must have the same number of bins as the histogram generator binning."
        # The binning that is self.binning is the original binning before we split signal and background
        summed_universes = np.zeros((universes.shape[0], self.binning.n_bins))
        for ch in self.channels:
            signal_bin_idx = self.hist_generator.binning._channel_bin_idx(f"{ch}__SIGNAL")
            background_bin_idx = self.hist_generator.binning._channel_bin_idx(f"{ch}__BACKGROUND")
            if isinstance(self.binning, MultiChannelBinning):
                channel_bin_idx = self.binning._channel_bin_idx(ch)
            else:
                # in the case that the output binning is just a single Binning, the bin indices
                # are just the range of the number of bins.
                channel_bin_idx = range(self.binning.n_bins)
            summed_universes[:, channel_bin_idx] = universes[:, background_bin_idx]
            summed_universes[:, channel_bin_idx] += (
                universes[:, signal_bin_idx] * self.parameters["signal_strength"].m
            )
        return summed_cov, summed_universes

    def calculate_unisim_uncertainties(self, *args, **kwargs):
        return_histograms = kwargs.get("return_histograms", False)
        result = self.hist_generator.calculate_unisim_uncertainties(*args, **kwargs)
        summed_cov = None
        if not return_histograms:
            assert isinstance(result, np.ndarray), "Result must be a numpy array"
            cov_mat = result
        else:
            cov_mat = result[0]
        # Use the mechanics of the MultiChannelHistogram to correctly scale and add the channels.
        assert isinstance(
            self.hist_generator.binning, MultiChannelBinning
        ), "Binning must be a MultiChannelBinning"
        us_hist = MultiChannelHistogram(
            self.hist_generator.binning.copy(),
            # The bin counts don't matter for the covariance matrix, so we just fill it with ones.
            bin_counts=np.ones(self.hist_generator.binning.n_bins),
            covariance_matrix=cov_mat,
        )
        for ch in self.channels:
            us_hist.scale_channel(f"{ch}__SIGNAL", self.parameters["signal_strength"].m)
            us_hist.sum_channels(
                [f"{ch}__SIGNAL", f"{ch}__BACKGROUND"], ch, replace=True, inplace=True
            )
        if not return_histograms:
            return us_hist.covariance_matrix
        else:
            summed_cov = us_hist.covariance_matrix

        # For unisim variations, the universes are stored in a dict where keys are the names of the variations.
        # The values are still 2D numpy arrays, because there are some knobs for which there are two
        # variations. That means that the shape of the values of the dictionary is either (1, n_bins)
        # or (2, n_bins).
        summed_obs = {}
        for key in result[1].keys():
            summed_obs[key] = np.zeros((result[1][key].shape[0], self.binning.n_bins))
            for ch in self.channels:
                signal_bin_idx = self.hist_generator.binning._channel_bin_idx(f"{ch}__SIGNAL")
                background_bin_idx = self.hist_generator.binning._channel_bin_idx(
                    f"{ch}__BACKGROUND"
                )
                if isinstance(self.binning, MultiChannelBinning):
                    channel_bin_idx = self.binning._channel_bin_idx(ch)
                else:
                    # in the case that the output binning is just a single Binning, the bin indices
                    # are just the range of the number of bins.
                    channel_bin_idx = range(self.binning.n_bins)
                summed_obs[key][:, channel_bin_idx] = result[1][key][:, background_bin_idx]
                summed_obs[key][:, channel_bin_idx] += (
                    result[1][key][:, signal_bin_idx] * self.parameters["signal_strength"].m
                )
        return summed_cov, summed_obs

    # for all attributes that are not the "generate" function we want to forward the call to the full generator
    def __getattr__(self, name):
        return getattr(self.hist_generator, name)


class SpectralIndexGenerator(HistogramGenerator):
    def adjust_weights(self, dataframe: pd.DataFrame, base_weights):
        assert self.parameters is not None
        delta_gamma = self.parameters["delta_gamma"].m
        assert isinstance(delta_gamma, float), "delta_gamma must be a float"
        spectral_weight = np.power(dataframe["reco_e"], delta_gamma)
        spectral_weight[~np.isfinite(spectral_weight)] = 0
        return base_weights * spectral_weight


class SpectralSoBGenerator(SignalOverBackgroundGenerator):
    hist_gen_cls = SpectralIndexGenerator
