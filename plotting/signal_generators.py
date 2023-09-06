"""Module containing different signal generators. Every class defined in this file is available 
as a signal generator when defining an analysis in a toml file.
"""


import numpy as np
from plotting.histogram import Binning, HistogramGenerator, RunHistGenerator
from plotting.parameters import ParameterSet

class SignalOverBackgroundGenerator(HistogramGenerator):

    hist_gen_cls = HistogramGenerator

    def __init__(self, *args, signal_query="category == 111", background_query="category != 111", parameters=None, **kwargs):
        assert parameters is not None, "parameters must be given"
        query = kwargs.pop("query", None)
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
        return self.full_generator.calculate_multisim_uncertainties(*args, **kwargs)

class SpectralIndexGenerator(HistogramGenerator):
    def adjust_weights(self, dataframe, base_weights):
        delta_gamma = self.parameters["delta_gamma"].m
        spectral_weight = np.power(dataframe["reco_e"], delta_gamma)
        spectral_weight[~np.isfinite(spectral_weight)] = 0
        return base_weights * spectral_weight

class SpectralSoBGenerator(SignalOverBackgroundGenerator):
    hist_gen_cls = SpectralIndexGenerator