"""Module containing different signal generators. Every class defined in this file is available 
as a signal generator when defining an analysis in a toml file.
"""


import numpy as np
from microfit.histogram import Binning, HistogramGenerator, RunHistGenerator
from microfit.parameters import ParameterSet

class SignalOverBackgroundGenerator(HistogramGenerator):

    hist_gen_cls = HistogramGenerator

    def __init__(self, *args, signal_query="category == 111", background_query="category != 111", parameters=None, **kwargs):
        assert parameters is not None, "parameters must be given"
        query = kwargs.pop("query", None)
        if query is not None:
            assert not "category" in query, "category should not be in query"
            signal_query = query + f" and ({signal_query})"
            background_query = query + f" and ({background_query})"
        self.parameters = parameters
        assert "signal_strength" in parameters.names, "signal_strength must be in parameters"
        # We want to forward all parameters except the signal strength to the histogram generator
        pnames = parameters.names
        pnames.remove("signal_strength")
        forward_parameters = parameters[pnames]
        self.signal_generator = self.hist_gen_cls(*args, query=signal_query, parameters=forward_parameters, **kwargs)
        self.background_generator = self.hist_gen_cls(*args, query=background_query, parameters=forward_parameters, **kwargs)
        # the dataframe is needed to find the categories when plotting stacked histograms
        # hope we can get rid of this in the future
        self.dataframe = self.signal_generator.dataframe
        self._binning = self.signal_generator._binning
        self.full_generator = self.hist_gen_cls(*args, query=query, parameters=forward_parameters, **kwargs)
    
    def generate(self, *args, **kwargs):
        signal_hist = self.signal_generator.generate(*args, **kwargs)
        background_hist = self.background_generator.generate(*args, **kwargs)
        return self.parameters["signal_strength"].m * signal_hist + background_hist
    
    def _resync_parameters(self):
        self.parameters.synchronize(self.signal_generator.parameters)
        self.parameters.synchronize(self.background_generator.parameters)

    def calculate_multisim_uncertainties(self, *args, **kwargs):
        return_histograms = kwargs.get("return_histograms", False)
        signal_result = self.signal_generator.calculate_multisim_uncertainties(*args, **kwargs)
        background_result = self.background_generator.calculate_multisim_uncertainties(*args, **kwargs)
        summed_cov = None
        if not return_histograms:
            # In this case the result is just the covariance matrix, so we can just add them.
            # Note that we need to multiply the signal result by the signal strength SQUARED
            summed_cov = self.parameters["signal_strength"].m**2 * signal_result + background_result
            return summed_cov
        # If the histograms are to be returned, the result is a tuple of the covariance matrix and the histograms.
        summed_cov = self.parameters["signal_strength"].m**2 * signal_result[0] + background_result[0]
        # In this case, we also need to add together the histograms of the observations for every universe.
        summed_obs = self.parameters["signal_strength"].m * signal_result[1] + background_result[1]
        return summed_cov, summed_obs
    
    def calculate_unisim_uncertainties(self, *args, **kwargs):
        return_histograms = kwargs.get("return_histograms", False)
        signal_result = self.signal_generator.calculate_unisim_uncertainties(*args, **kwargs)
        background_result = self.background_generator.calculate_unisim_uncertainties(*args, **kwargs)
        summed_cov = None
        if not return_histograms:
            # In this case the result is just the covariance matrix, so we can just add them.
            # Note that we need to multiply the signal result by the signal strength SQUARED
            summed_cov = self.parameters["signal_strength"].m**2 * signal_result + background_result
            return summed_cov
        # If the histograms are to be returned, the result is a tuple of the covariance matrix and the histograms.
        summed_cov = self.parameters["signal_strength"].m**2 * signal_result[0] + background_result[0]
        # For unisim variations, the universes are stored in a dict where keys are the names of the variations.
        # We iterate over all the keys and sum the histograms for each variation.
        summed_obs = {}
        for key in signal_result[1].keys():
            summed_obs[key] = self.parameters["signal_strength"].m * signal_result[1][key] + background_result[1][key]
        return summed_cov, summed_obs
    
    # for all attributes that are not the "generate" function we want to forward the call to the full generator
    def __getattr__(self, name):
        return getattr(self.full_generator, name)

class SpectralIndexGenerator(HistogramGenerator):
    def adjust_weights(self, dataframe, base_weights):
        delta_gamma = self.parameters["delta_gamma"].m
        spectral_weight = np.power(dataframe["reco_e"], delta_gamma)
        spectral_weight[~np.isfinite(spectral_weight)] = 0
        return base_weights * spectral_weight

class SpectralSoBGenerator(SignalOverBackgroundGenerator):
    hist_gen_cls = SpectralIndexGenerator