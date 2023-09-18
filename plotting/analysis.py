"""Classes to run the LEE (and possibly other) analyses."""


import logging
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from scipy.linalg import block_diag
import toml
from data_loading import load_runs
from plotting.histogram import Binning, HistogramGenerator, RunHistGenerator, Histogram
from plotting.parameters import ParameterSet
from plotting.run_plotter import RunHistPlotter
from plotting import signal_generators
from plotting.statistics import sideband_constraint_correction


class MultibandAnalysis(object):
    def __init__(self, configuration):
        self.logger = logging.getLogger(__name__)
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

    def run_hist_generator_from_config(self, rundata, weights, data_pot, config, is_signal=True) -> RunHistGenerator:
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

    def generate_multiband_histogram(self, include_multisim_errors=False, use_sideband=False, strict_covar_checking=False):
        """Generate a combined histogram from all signal bands."""

        total_nbins = sum([g.binning.n_bins for g in self._signal_generators])
        # make a Binning of the global bin number
        bin_edges = np.arange(total_nbins + 1)
        global_binning = Binning("None", bin_edges, label="Global bin number")
        stat_only_signal_hists = [
            g.mc_hist_generator.generate(include_multisim_errors=False, use_sideband=False)
            for g in self._signal_generators
        ]
        if not include_multisim_errors:
            # If we don't include the multisim errors, we are done
            combined_nominal_values = np.concatenate([h.nominal_values for h in stat_only_signal_hists])
            combined_uncertainty = np.concatenate([h.std_devs for h in stat_only_signal_hists])
            return Histogram(global_binning, combined_nominal_values, uncertainties=combined_uncertainty)
        # combine the histograms
        combined_nominal_values = np.concatenate([h.nominal_values for h in stat_only_signal_hists])
        combined_covariance = self._get_total_multiband_covariance(with_stat_only=True, with_unisim=True, include_sideband=False)
        # sanity check: the diagonal blocks of the combined covariance should be the same
        # as if we had calculated the histograms with multisim separately
        pos = 0
        for i, g in enumerate(self._signal_generators):
            n_bins = g.binning.n_bins
            signal_hist = g.mc_hist_generator.generate(include_multisim_errors=True, use_sideband=False)
            covar = signal_hist.cov_matrix
            # for testing purposes, we divide the covariance by the square of the nominal values to get
            # the relative covariance
            covar = covar / np.outer(signal_hist.nominal_values, signal_hist.nominal_values)
            reference_covar = combined_covariance[pos : pos + n_bins, pos : pos + n_bins] / np.outer(
                combined_nominal_values[pos : pos + n_bins], combined_nominal_values[pos : pos + n_bins]
            )
            # When we are using non-standard histogram generators such as the signal-over-background generator,
            # it is possible that the block matrices are not equivalent. In that case, we only check for 
            # loose agreement. However, when the standard HistogramGenerator class is used, this really should
            # work correctly.
            np.testing.assert_allclose(reference_covar, covar, atol=1e-6 if strict_covar_checking else 1e-1)
            pos += n_bins
        if not use_sideband:
            return Histogram(global_binning, combined_nominal_values, covariance_matrix=combined_covariance)
        sideband_prediction = self._sideband_generator.get_total_prediction()
        sideband_observation = self._sideband_generator.data_hist_generator.generate()
        # To make the covariance matrix, we need to first generate the full multiband covariance
        # and then apply the sideband constraint.
        multiband_covariance_multisim = self._get_total_multiband_covariance(with_stat_only=False, with_unisim=False, include_sideband=True)
        # Now we can calculate the sideband correction. This is done only for multisim errors.
        delta_mu, delta_covar = sideband_constraint_correction(
            sideband_measurement=sideband_observation.nominal_values,
            sideband_central_value=sideband_prediction.nominal_values,
            concat_covariance=multiband_covariance_multisim,
            sideband_covariance=sideband_prediction.cov_matrix,
        )
        self.logger.debug(f"Sideband constraint correction: {delta_mu}")
        self.logger.debug(f"Sideband constraint correction covariance: {delta_covar}")
        combined_nominal_values += delta_mu
        combined_covariance += delta_covar

        return Histogram(global_binning, combined_nominal_values, covariance_matrix=combined_covariance)
    
    def generate_multiband_data_histogram(self):
        """Generate a combined histogram from all signal bands."""

        total_nbins = sum([g.binning.n_bins for g in self._signal_generators])
        # make a Binning of the global bin number
        bin_edges = np.arange(total_nbins + 1)
        global_binning = Binning("None", bin_edges, label="Global bin number")
        data_hists = [
            g.data_hist_generator.generate()
            for g in self._signal_generators
        ]
        # combine the histograms
        combined_nominal_values = np.concatenate([h.nominal_values for h in data_hists])
        combined_uncertainty = np.concatenate([h.std_devs for h in data_hists])
        return Histogram(global_binning, combined_nominal_values, uncertainties=combined_uncertainty)

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

    def _get_total_multiband_covariance(self, with_unisim=False, with_stat_only=False, include_sideband=False):
        hist_generators = []
        hist_generators.extend([g.mc_hist_generator for g in self._signal_generators])
        if self._sideband_generator is not None and include_sideband:
            hist_generators.append(self._sideband_generator.mc_hist_generator)
        total_nbins = sum([g.binning.n_bins for g in hist_generators])
        combined_covar = np.zeros((total_nbins, total_nbins))
        for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
            multiband_covariance = HistogramGenerator.multiband_covariance(hist_generators, ms_column=ms_column)
            combined_covar += multiband_covariance
        if with_unisim:
            combined_covar += HistogramGenerator.multiband_unisim_covariance(hist_generators)
        if with_stat_only:
            combined_covar += self._get_multiband_stat_only_covariance(include_sideband=include_sideband)
        return combined_covar

    def _get_multiband_stat_only_covariance(self, include_sideband=False):
        hist_generators = []
        hist_generators.extend([g.mc_hist_generator for g in self._signal_generators])
        if self._sideband_generator is not None and include_sideband:
            hist_generators.append(self._sideband_generator.mc_hist_generator)
        stat_only_histograms = [g.generate(include_multisim_errors=False, use_sideband=False) for g in hist_generators]
        stat_only_covariance = np.diag(np.concatenate([h.std_devs**2 for h in stat_only_histograms]))
        return stat_only_covariance

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

    def two_hypothesis_test(self, h0_params, h1_params, sensitivity_only=False):
        """Perform a two hypothesis test between two parameter sets.

        For each hypothesis, pseudo-experiments are generated by sampling
        first from the covariance matrix of the histograms, and then using
        the result as the expectation value for Poisson-fluctuations. The
        chi-square of the best fit of every pseudo-experiment is recorded.
        Then, the p-value of the observed chi-square can be calculated under
        both hypotheses. If we are only generating a sensitivity, the result
        is the p-value of the median chi-square of the pseudo-experiments
        sampled with H1 (the alternative hypothesis) calculated under H0 (the
        null hypothesis).

        The resulting dictionary contains the following keys:
        - "chi2": The chi-square of the observed data.
        - "pval": The p-value of the observed chi-square under H0.
        - "pval_h1": The p-value of the observed chi-square under H1.
        - "chi2_median_h1": The median chi-square of the pseudo-experiments
          sampled under H1.
        - "median_pval": The p-value of the median chi-square under H0.
        - "samples_h0": The chi-square of the pseudo-experiments sampled
          under H0.
        - "samples_h1": The chi-square of the pseudo-experiments sampled
          under H1.

        Args:
            h0_params (ParameterSet): The parameters of the first hypothesis.
            h1_params (ParameterSet): The parameters of the second hypothesis.

        Returns:
            dict : Dictionary containing the results of the hypothesis test.
        """

        assert set(h0_params.names) == set(h1_params.names), "Parameter sets must have the same parameters"
        assert self.parameters.names == set(
            h0_params.names
        ), "Parameter sets must have the same parameters as the analysis"
        # set parameters to H0, be sure to set the *values* and not the objects in order
        # not to break the linking between parameters of the signal bands
        for name in h0_params.names:
            self.parameters[name].value = h0_params[name].value

    @classmethod
    def from_toml(cls, file_path):
        # Try loading from file
        with open(file_path, "r") as file:
            configuration = toml.load(file)
        return cls(configuration)
