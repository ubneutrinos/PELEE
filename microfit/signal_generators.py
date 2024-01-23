"""Module containing different signal generators. Every class defined in this file is available 
as a signal generator when defining an analysis in a toml file.
"""


import numpy as np
from copy import deepcopy
from microfit.histogram import Binning, HistogramGenerator, RunHistGenerator
from microfit.parameters import ParameterSet


class SignalOverBackgroundGenerator(HistogramGenerator):
    hist_gen_cls = HistogramGenerator

    def __init__(
        self,
        *args,
        signal_query="category == 111",
        background_query="category != 111",
        parameters=None,
        **kwargs
    ):
        assert parameters is not None, "parameters must be given"
        self.signal_query = signal_query
        self.background_query = background_query
        self.parameters = parameters
        assert "signal_strength" in parameters.names, "signal_strength must be in parameters"
        # We want to forward all parameters except the signal strength to the histogram generator
        pnames = parameters.names
        pnames.remove("signal_strength")
        forward_parameters = parameters[pnames]
        self.hist_generator = self.hist_gen_cls(*args, parameters=forward_parameters, **kwargs)

    def append_query(self, query, extra_query):
        if query is None:
            return extra_query
        return query + " and " + extra_query

    def signal_kwargs(self, kwargs):
        output = deepcopy(kwargs)
        extra_query = output.pop("extra_query", None)
        extra_query = self.append_query(extra_query, self.signal_query)
        output["extra_query"] = extra_query
        return output

    def background_kwargs(self, kwargs):
        output = deepcopy(kwargs)
        extra_query = output.pop("extra_query", None)
        extra_query = self.append_query(extra_query, self.background_query)
        output["extra_query"] = extra_query
        return output

    def generate(self, **kwargs):
        signal_hist = self.hist_generator.generate(**self.signal_kwargs(kwargs))
        background_hist = self.hist_generator.generate(**self.background_kwargs(kwargs))
        return float(self.parameters["signal_strength"].m) * signal_hist + background_hist

    def _resync_parameters(self):
        self.parameters.synchronize(self.hist_generator.parameters)

    def calculate_multisim_uncertainties(self, *args, **kwargs):
        return_histograms = kwargs.get("return_histograms", False)
        signal_result = self.hist_generator.calculate_multisim_uncertainties(
            *args, **self.signal_kwargs(kwargs)
        )
        background_result = self.hist_generator.calculate_multisim_uncertainties(
            *args, **self.background_kwargs(kwargs)
        )
        summed_cov = None
        if not return_histograms:
            # In this case the result is just the covariance matrix, so we can just add them.
            # Note that we need to multiply the signal result by the signal strength SQUARED
            summed_cov = (
                self.parameters["signal_strength"].m ** 2 * signal_result + background_result
            )
            return summed_cov
        # If the histograms are to be returned, the result is a tuple of the covariance matrix and the histograms.
        summed_cov = (
            self.parameters["signal_strength"].m ** 2 * signal_result[0] + background_result[0]
        )
        signal_universes = signal_result[1]
        background_universes = background_result[1]
        if signal_universes is None:
            return summed_cov, background_universes
        if background_universes is None:
            return summed_cov, signal_universes
        # If we get here, both signal and background have universes.
        assert len(signal_result[1]) == len(
            background_result[1]
        ), "Number of universes must be the same for signal and background"
        summed_universes = (
            self.parameters["signal_strength"].m * signal_result[1] + background_result[1]
        )
        # If these are lists, then summing then will cause a nasty bug where the universes are not summed,
        # but instead appended. This will be happily processed by the rest of the code, but the result will be wrong.
        assert len(summed_universes) == len(
            signal_result[1]
        ), "Number of universes must not change when summing. Ensure that the universes are numpy arrays, not lists."
        return summed_cov, summed_universes

    def calculate_unisim_uncertainties(self, *args, **kwargs):
        return_histograms = kwargs.get("return_histograms", False)
        signal_result = self.hist_generator.calculate_unisim_uncertainties(
            *args, **self.signal_kwargs(kwargs)
        )
        background_result = self.hist_generator.calculate_unisim_uncertainties(
            *args, **self.background_kwargs(kwargs)
        )
        summed_cov = None
        if not return_histograms:
            # In this case the result is just the covariance matrix, so we can just add them.
            # Note that we need to multiply the signal result by the signal strength SQUARED
            summed_cov = (
                self.parameters["signal_strength"].m ** 2 * signal_result + background_result
            )
            return summed_cov
        # If the histograms are to be returned, the result is a tuple of the covariance matrix and the histograms.
        summed_cov = (
            self.parameters["signal_strength"].m ** 2 * signal_result[0] + background_result[0]
        )
        # For unisim variations, the universes are stored in a dict where keys are the names of the variations.
        # We iterate over all the keys and sum the histograms for each variation.
        summed_obs = {}
        for key in signal_result[1].keys():
            assert isinstance(signal_result[1][key], np.ndarray), "Universes must be numpy arrays"
            assert isinstance(
                background_result[1][key], np.ndarray
            ), "Universes must be numpy arrays"
            summed_obs[key] = (
                self.parameters["signal_strength"].m * signal_result[1][key]
                + background_result[1][key]
            )
        return summed_cov, summed_obs

    # for all attributes that are not the "generate" function we want to forward the call to the full generator
    def __getattr__(self, name):
        return getattr(self.hist_generator, name)


class SpectralIndexGenerator(HistogramGenerator):
    def adjust_weights(self, dataframe, base_weights):
        delta_gamma = self.parameters["delta_gamma"].m
        spectral_weight = np.power(dataframe["reco_e"], delta_gamma)
        spectral_weight[~np.isfinite(spectral_weight)] = 0
        return base_weights * spectral_weight


class SpectralSoBGenerator(SignalOverBackgroundGenerator):
    hist_gen_cls = SpectralIndexGenerator
