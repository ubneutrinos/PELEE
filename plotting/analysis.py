"""Classes to run the LEE (and possibly other) analyses."""


from typing import List
from matplotlib import pyplot as plt
import numpy as np

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
        self.sideband_name = None
        if "sideband" in configuration:
            sideband_configuration = configuration["sideband"]
            if "name" in sideband_configuration:
                self.sideband_name = sideband_configuration["name"]
            else:
                self.sideband_name = sideband_configuration["selection"]
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
        self.signal_names = [config.get("name", config["selection"]) for config in signal_configurations]
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

    def _get_total_multiband_covariance(self):
        hist_generators = []
        if self._sideband_generator is not None:
            hist_generators.append(self._sideband_generator.mc_hist_generator)
        hist_generators.extend([g.mc_hist_generator for g in self._signal_generators])
        total_nbins = sum([g.binning.n_bins for g in hist_generators])
        combined_covar = np.zeros((total_nbins, total_nbins))
        for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
            multiband_covariance = HistogramGenerator.multiband_covariance(hist_generators, ms_column=ms_column)
            combined_covar += multiband_covariance
        # combined_covar += HistogramGenerator.multiband_unisim_covariance(hist_generators)
        return combined_covar

    def plot_correlation(self, ms_column=None, ax=None):
        hist_generators = []
        hist_gen_labels = []
        if self._sideband_generator is not None:
            hist_generators.append(self._sideband_generator.mc_hist_generator)
            hist_gen_labels.append(self.sideband_name)
        hist_generators.extend([g.mc_hist_generator for g in self._signal_generators])
        hist_gen_labels.extend(self.signal_names)

        if ms_column is None:
            multiband_covariance = self._get_total_multiband_covariance()
        else:
            multiband_covariance = HistogramGenerator.multiband_covariance(hist_generators, ms_column=ms_column)
        # convert to correlation matrix
        multiband_covariance = multiband_covariance / np.sqrt(
            np.outer(np.diag(multiband_covariance), np.diag(multiband_covariance))
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        else:
            fig = ax.figure
        # show the covariance matrix as a heatmap
        X, Y = np.meshgrid(np.arange(multiband_covariance.shape[0] + 1), np.arange(multiband_covariance.shape[1] + 1))
        p = ax.pcolormesh(X, Y, multiband_covariance.T, cmap="Spectral_r", shading="flat")
        # colorbar
        cbar = fig.colorbar(p, ax=ax)
        cbar.set_label("correlation")
        if ms_column is None:
            ax.set_title("Total Multiband Correlation")
        else:
            ax.set_title(f"Multiband Correlation: {ms_column}")
        # turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # set tick marks at every bin
        ax.set_xticks(np.arange(multiband_covariance.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(multiband_covariance.shape[1]) + 0.5, minor=False)
        ax.tick_params(axis="both", which="both", direction="in")
        # draw vertical and horizontal lines splitting the different histograms that went
        # into the covariance matrix
        pos = 0
        for hist_gen, label in zip(hist_generators, hist_gen_labels):
            pos += hist_gen.binning.n_bins
            ax.axvline(pos, color="k", linestyle="--")
            ax.axhline(pos, color="k", linestyle="--")
            ax.text(pos - hist_gen.binning.n_bins / 2, -1, label, ha="center", va="top", fontsize=12)
            ax.text(-1, pos - hist_gen.binning.n_bins / 2, label, ha="right", va="center", fontsize=12)
        return fig, ax

    @classmethod
    def from_toml(cls, file_path):
        # Try loading from file
        with open(file_path, "r") as file:
            configuration = toml.load(file)
        return cls(configuration)
