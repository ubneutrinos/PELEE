"""Classes to run the LEE (and possibly other) analyses."""


from typing import List
from matplotlib import pyplot as plt

import toml
from data_loading import load_runs
from plotting.histogram import Binning, HistogramGenerator, RunHistGenerator
from plotting.parameters import ParameterSet
from plotting.run_plotter import RunHistPlotter
from plotting import signal_generators


class MultibandAnalysis(object):
    def __init__(self, configuration):
        # The analysis may use several signal bands, but only one sideband.
        # For every signal band and the sideband, we are going to create a
        # RunHistGenerator object that can create histograms from the data.
        signal_configurations = configuration["signal"]
        self._sideband_generator = None
        if "sideband" in configuration:
            sideband_configuration = configuration["sideband"]
            sideband_data, sideband_weights, sideband_pot = load_runs(**configuration["sideband_data"])
            self._sideband_generator = self.run_hist_generator_from_config(
                sideband_data, sideband_weights, sideband_pot, sideband_configuration, is_signal=False
            )
        else:
            sideband_configuration = None
            self._sideband_generator = None

        for config in signal_configurations + [sideband_configuration]:
            self._check_config(config)

        self._signal_configurations = signal_configurations
        rundata, weights, data_pot = load_runs(**configuration["signal_data"])
        self._signal_generators = [
            self.run_hist_generator_from_config(rundata, weights, data_pot, config) for config in signal_configurations
        ]
        self.parameters = sum([g.parameters for g in self._signal_generators])
        for gen in self._signal_generators:
            gen.mc_hist_generator._resync_parameters()
        self._check_shared_params([g.parameters for g in self._signal_generators])

    def _check_shared_params(self, param_sets: List[ParameterSet]):
            shared_names = list(set(param_sets[0].names).intersection(*[set(ps.names) for ps in param_sets[1:]]))
            # Reference ParameterSet
            ref_set = param_sets[0]
            for name in shared_names:
                ref_param = ref_set[name]
                for ps in param_sets[1:]:
                    assert ref_param is ps[name], f"Parameter {name} is not the same object in all ParameterSets"

    def _check_config(self, config):
        return True  # TODO: implement

    def run_hist_generator_from_config(self, rundata, weights, data_pot, config, is_signal=True):
        binning = Binning.from_config(**config["binning"])
        print(f"Making generator for selection {config['selection']} and preselection {config['preselection']}")
        parameters = ParameterSet.from_dict(config["parameter"]) if "parameter" in config else None
        if "mc_hist_generator_cls" in config:
            try:
                mc_hist_generator_cls = getattr(signal_generators, config["mc_hist_generator_cls"])
            except AttributeError:
                # try globals instead 
                mc_hist_generator_cls = globals()[config["mc_hist_generator_cls"]]
            mc_hist_generator_kwargs = config.get("mc_hist_generator_kwargs", {})
        else:
            mc_hist_generator_cls = None
            mc_hist_generator_kwargs = {}
        run_generator = RunHistGenerator(
            rundata,
            binning,
            data_pot=data_pot,
            selection=config["selection"],
            preselection=config["preselection"],
            sideband_generator=self._sideband_generator,
            uncertainty_defaults=config["uncertainties"],
            mc_hist_generator_cls=mc_hist_generator_cls if is_signal else None,
            parameters=parameters,
            **mc_hist_generator_kwargs,
        )
        return run_generator

    def plot_signals(self, category_column="paper_category", **kwargs):
        # make sub-plot for each signal in a horizontal arrangement
        n_signals = len(self._signal_generators)
        fig, axes = plt.subplots(1, n_signals, figsize=(n_signals * 5, 5), squeeze=False)
        for i, generator in enumerate(self._signal_generators):
            plotter = RunHistPlotter(generator)
            plotter.plot(category_column=category_column, ax=axes[0, i], **kwargs)
        
    def plot_sideband(self, category_column="category", **kwargs):
        plotter = RunHistPlotter(self._sideband_generator)
        plotter.plot(category_column=category_column, **kwargs)

    @classmethod
    def from_toml(cls, file_path):
        # Try loading from file
        with open(file_path, "r") as file:
            configuration = toml.load(file)
        return cls(configuration)