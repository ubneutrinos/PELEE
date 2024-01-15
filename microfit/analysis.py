"""Classes to run the LEE (and possibly other) analyses."""


import logging
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from scipy.linalg import block_diag
import toml
from data_loading import load_runs
from microfit.histogram import (
    Binning,
    HistogramGenerator,
    RunHistGenerator,
    Histogram,
    MultiChannelBinning,
)
from microfit.parameters import ParameterSet
from microfit.run_plotter import RunHistPlotter
from microfit import signal_generators
from microfit.statistics import sideband_constraint_correction, chi_square


class MultibandAnalysis(object):
    def __init__(self, configuration=None, run_hist_generators=None, constraint_channels=[], signal_channels=[]):
        self.logger = logging.getLogger(__name__)
        # The analysis may use several signal bands, but only one sideband.
        # For every signal band and the sideband, we are going to create a
        # RunHistGenerator object that can create histograms from the data.
        if configuration is None:
            self._init_from_generators(run_hist_generators, constraint_channels, signal_channels)
        else:
            self._init_from_config(configuration)
        self.channels = []
        for gen in self._run_hist_generators:
            if isinstance(gen.binning, MultiChannelBinning):
                self.channels.extend(gen.binning.labels)
            else:
                self.channels.append(gen.binning.label)
        for ch in constraint_channels:
            assert ch in self.channels, f"Constraint channel {ch} not found in analysis channels"

    def _init_from_generators(
        self, run_hist_generators: List[RunHistGenerator], constraint_channels: List[str], signal_channels: List[str]
    ):
        self._run_hist_generators = run_hist_generators
        self.constraint_channels = constraint_channels
        self.signal_channels = signal_channels
        self.parameters = sum([g.parameters for g in self._run_hist_generators])
        for gen in self._run_hist_generators:
            gen.mc_hist_generator._resync_parameters()
        self._check_shared_params([g.parameters for g in self._run_hist_generators])

    def _init_from_config(self, configuration):
        # The analysis may use several generators to produce a multi-channel histogram
        raise NotImplementedError("TODO: update implementation for configuration loading")
        generator_configurations = configuration["generator"]
        for config in generator_configurations:
            self._check_config(config)
        rundata, weights, data_pot = load_runs(**configuration["data_loading"])
        self.data_pot = data_pot
        self._run_hist_generators = [
            self.run_hist_generator_from_config(rundata, weights, data_pot, config)
            for config in generator_configurations
        ]
        self.parameters = sum([g.parameters for g in self._run_hist_generators])
        for gen in self._run_hist_generators:
            gen.mc_hist_generator._resync_parameters()
        self._check_shared_params([g.parameters for g in self._run_hist_generators])

    def _check_shared_params(self, param_sets: List[ParameterSet]):
        shared_names = list(
            set(param_sets[0].names).intersection(*[set(ps.names) for ps in param_sets[1:]])
        )
        # Reference ParameterSet
        ref_set = param_sets[0]
        for name in shared_names:
            ref_param = ref_set[name]
            for ps in param_sets[1:]:
                assert (
                    ref_param is ps[name]
                ), f"Parameter {name} is not the same object in all ParameterSets"

    def _check_config(self, config):
        return True  # TODO: implement

    def run_hist_generator_from_config(
        self, rundata, weights, data_pot, config, is_signal=True
    ) -> RunHistGenerator:
        raise NotImplementedError("TODO: update implementation for configuration loading")
        # binning = Binning.from_config(**config["binning"])
        # print(f"Making generator for selection {config['selection']} and preselection {config['preselection']}")
        # parameters = ParameterSet.from_dict(config["parameter"]) if "parameter" in config else None
        # if "mc_hist_generator_cls" in config:
        #     try:
        #         mc_hist_generator_cls = getattr(signal_generators, config["mc_hist_generator_cls"])
        #     except AttributeError:
        #         # try globals instead
        #         mc_hist_generator_cls = globals()[config["mc_hist_generator_cls"]]
        #     mc_hist_generator_kwargs = config.get("mc_hist_generator_kwargs", {})
        # else:
        #     mc_hist_generator_cls = None
        #     mc_hist_generator_kwargs = {}
        # run_generator = RunHistGenerator(
        #     rundata,
        #     binning,
        #     data_pot=data_pot,
        #     selection=config["selection"],
        #     preselection=config["preselection"],
        #     sideband_generator=self._sideband_generator,
        #     uncertainty_defaults=config["uncertainties"],
        #     mc_hist_generator_cls=mc_hist_generator_cls if is_signal else None,
        #     parameters=parameters,
        #     **mc_hist_generator_kwargs,
        # )
        # return run_generator

    def _apply_constraints(self, hist, constraint_channels=None):
        """Apply the sideband constraint to the given histogram."""

        data_hist = self.generate_multiband_data_histogram()
        if data_hist is None:
            return hist
        constraint_channels = constraint_channels or self.constraint_channels
        for constraint_channel in constraint_channels:
            constraint_hist = data_hist[constraint_channel]
            hist = hist.update_with_measurement(constraint_channel, constraint_hist.nominal_values)
        return hist

    def generate_multiband_histogram(
        self,
        include_multisim_errors=False,
        use_sideband=False,
        scale_to_pot=None,
        constraint_channels=None,
        signal_channels=None,
        include_non_signal_channels=False,
    ):
        """Generate the combined MC histogram from all channels."""

        mc_hist_generators = [g.mc_hist_generator for g in self._run_hist_generators]
        output_hist = HistogramGenerator.generate_joint_histogram(
            mc_hist_generators, include_multisim_errors=include_multisim_errors
        )
        ext_hist_generators = [g.ext_hist_generator for g in self._run_hist_generators]
        joint_ext_hist = HistogramGenerator.generate_joint_histogram(
            ext_hist_generators, include_multisim_errors=False
        )
        output_hist += joint_ext_hist
        constraint_channels = constraint_channels or self.constraint_channels
        signal_channels = signal_channels or self.signal_channels
        all_channels = signal_channels + constraint_channels
        output_hist = output_hist[all_channels]
        if use_sideband:
            output_hist = self._apply_constraints(output_hist, constraint_channels=constraint_channels)
        if not include_non_signal_channels:
            output_hist = output_hist[signal_channels]
        if scale_to_pot is not None:
            output_hist *= scale_to_pot / self.data_pot
        return output_hist

    def generate_multiband_data_histogram(self):
        """Generate a combined histogram from all unblinded data channels."""

        unblinded_generators = [
            g.data_hist_generator for g in self._run_hist_generators if not g.is_blinded
        ]
        if len(unblinded_generators) == 0:
            return None
        return HistogramGenerator.generate_joint_histogram(
            unblinded_generators, include_multisim_errors=False
        )

    def plot_signals(self, category_column="paper_category", **kwargs):
        # make sub-plot for each channel in a horizontal arrangement
        n_signals = len(self.channels)
        fig, axes = plt.subplots(1, n_signals, figsize=(n_signals * 8, 5), squeeze=False)
        idx = 0
        for generator in self._run_hist_generators:
            plotter = RunHistPlotter(generator)
            if isinstance(generator.binning, MultiChannelBinning):
                for ch in generator.binning.labels:
                    plotter.plot(
                        category_column=category_column, ax=axes[0, idx], channel=ch, **kwargs
                    )
                    idx += 1
            else:
                plotter.plot(category_column=category_column, ax=axes[0, idx], **kwargs)
                idx += 1
        return fig, axes

    def plot_sideband(self, category_column="category", **kwargs):
        raise NotImplementedError("TODO: implement")

    def plot_correlation(
        self,
        ms_columns=["weightsGenie", "weightsFlux", "weightsReint"],
        ax=None,
        include_unisim_errors=True,
        channels=None,
        **draw_kwargs,
    ):
        hist_generators = []
        hist_generators.extend([g.mc_hist_generator for g in self._run_hist_generators])
        histogram = HistogramGenerator.generate_joint_histogram(
            hist_generators,
            include_multisim_errors=True,
            ms_columns=ms_columns,
            include_unisim_errors=include_unisim_errors,
        )
        all_channels = self.signal_channels + self.constraint_channels
        channels = channels or all_channels
        histogram = histogram[channels]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        else:
            fig = ax.figure
        histogram.draw_covariance_matrix(ax=ax, **draw_kwargs)
        return fig, ax

    def two_hypothesis_test(
        self, h0_params, h1_params, sensitivity_only=False, n_trials=1000, scale_to_pot=None
    ):
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
        - "chi2_h0": The chi-square of the observed data with respect to H0.
        - "pval_h0": The p-value of the observed chi-square under H0.
        - "chi2_h1": The chi-square of the observed data with respect to H1.
        - "pval_h1": The p-value of the observed chi-square under H1.
        - "ts_median_h1": The median test statistic of the pseudo-experiments
          sampled under H1.
        - "median_pval": The p-value of the median test statistic under H0.
        - "samples_h0": The test statistic of the pseudo-experiments sampled
          under H0.
        - "samples_h1": The test statistic of the pseudo-experiments sampled
          under H1.

        Parameters
        ----------
        h0_params : ParameterSet
            The parameters of the null hypothesis.
        h1_params : ParameterSet
            The parameters of the alternative hypothesis.
        sensitivity_only : bool
            If True, only the p-value of the median test statistic under H0
            is calculated. This is useful for calculating the sensitivity of
            an analysis.
        n_trials : int
            The number of pseudo-experiments to generate.
        scale_to_pot : float
            If given, scale the histograms to this POT value before generating
            the pseudo-experiments. Only a sensitivity can be produced in this
            case.

        Returns
        -------
        dict
            A dictionary containing the results of the test.
        """

        if scale_to_pot is not None:
            assert sensitivity_only, "Can only scale to POT when calculating sensitivity"
        assert set(h0_params.names) == set(
            h1_params.names
        ), "Parameter sets must have the same parameters"
        self.set_parameters(h0_params, check_matching=True)
        # generate the multiband histogram
        h0_hist = self.generate_multiband_histogram(
            include_multisim_errors=True, use_sideband=True, scale_to_pot=scale_to_pot
        )

        self.set_parameters(h1_params, check_matching=True)
        h1_hist = self.generate_multiband_histogram(
            include_multisim_errors=True, use_sideband=True, scale_to_pot=scale_to_pot
        )

        test_stat_h0 = []
        test_stat_h1 = []

        def test_statistic(observation):
            chi2_h0 = chi_square(
                observation.nominal_values, h0_hist.nominal_values, h0_hist.covariance_matrix
            )
            chi2_h1 = chi_square(
                observation.nominal_values, h1_hist.nominal_values, h1_hist.covariance_matrix
            )
            return chi2_h0 - chi2_h1

        for i in range(n_trials):
            pseudo_data_h0 = h0_hist.fluctuate(seed=4 * i).fluctuate_poisson(seed=4 * i + 1)
            pseudo_data_h1 = h1_hist.fluctuate(seed=4 * i + 2).fluctuate_poisson(seed=4 * i + 3)

            test_stat_h0.append(test_statistic(pseudo_data_h0))
            test_stat_h1.append(test_statistic(pseudo_data_h1))

        test_stat_h0 = np.array(test_stat_h0)
        test_stat_h1 = np.array(test_stat_h1)

        results = dict()
        results["ts_median_h1"] = np.median(test_stat_h1)
        results["median_pval"] = np.sum(test_stat_h0 > results["ts_median_h1"]) / n_trials
        results["samples_h0"] = test_stat_h0
        results["samples_h1"] = test_stat_h1

        data_hist = self.generate_multiband_data_histogram()
        if data_hist is None or sensitivity_only:
            # If the analysis has been run in blind mode, then the data histogram is None
            # and we are done. Fill None for the remaining results keys.
            results["chi2_h0"] = None
            results["pval_h0"] = None
            results["chi2_h1"] = None
            results["pval_h1"] = None

            return results
        # calculate the p-value of the observed data under H0
        real_data_ts = test_statistic(data_hist)
        results["chi2_h0"] = chi_square(
            data_hist.nominal_values, h0_hist.nominal_values, h0_hist.covariance_matrix
        )
        results["pval_h0"] = np.sum(test_stat_h0 > real_data_ts) / n_trials
        # calculate the p-value of the observed data under H1
        results["chi2_h1"] = chi_square(
            data_hist.nominal_values, h1_hist.nominal_values, h1_hist.covariance_matrix
        )
        results["pval_h1"] = np.sum(test_stat_h1 > real_data_ts) / n_trials

        return results

    def _get_minuit(self, observed_hist, scale_to_pot=None):
        """Prepare the Minuit object that can run a fit and more.

        Parameters
        ----------
        observed_hist : Histogram
            The data histogram to be fitted.

        Returns
        -------
        Minuit
            The Minuit object.
        """

        # make this an optional dependency only in case one wants to run a fit
        from iminuit import Minuit

        # if scale_to_pot is not None:
        # Warn the user that this should only be used when calculating a sensitivity
        # self.logger.warning("Scaling MC to non-default POT. This should only be used when calculating a sensitivity.")

        def loss(*args):
            # set the parameters
            for i, name in enumerate(self.parameters.names):
                self.parameters[name].value = args[i]
            # generate the histogram
            generated_hist = self.generate_multiband_histogram(
                include_multisim_errors=True,
                use_sideband=True,
                check_covar=False,
                scale_to_pot=scale_to_pot,
            )
            # calculate the chi-square
            return chi_square(
                observed_hist.nominal_values,
                generated_hist.nominal_values,
                generated_hist.covariance_matrix,
            )

        # TODO: This is the syntax for the old Minuit 1.5.4 version. We should upgrade to the new version
        # at some point.
        initial_value_kwargs = {name: self.parameters[name].m for name in self.parameters.names}
        limit_kwargs = {
            f"limit_{name}": self.parameters[name].magnitude_bounds
            for name in self.parameters.names
        }
        minuit_kwargs = {**initial_value_kwargs, **limit_kwargs}
        m = Minuit(loss, name=self.parameters.names, errordef=Minuit.LEAST_SQUARES, **minuit_kwargs)
        return m

    def fit_to_data(self, return_migrad=True):
        m = self._get_minuit(self.generate_multiband_data_histogram())
        m.migrad()
        best_fit_parameters = self.parameters.copy()
        if return_migrad:
            return best_fit_parameters, m
        return best_fit_parameters

    def fc_scan(self, parameter_name, scan_points, n_trials=100, scale_to_pot=None):
        """Perform a Feldman-Cousins scan of the given parameter."""
        from tqdm import tqdm

        # For every point in the scan, we assume that it is the truth and
        # then create pseudo-experiments by sampling from the covariance
        # matrix. Then, we calculate the best fit for every pseudo-experiment
        # and record the difference in chi-square between the best fit and
        # a fit where the parameter in question is fixed to the assumed truth.
        results = []
        print(f"Running FC scan over {len(scan_points)} points in {parameter_name}...")
        number_of_samples = len(scan_points) * n_trials
        for i, scan_point in enumerate(tqdm(scan_points)):
            self.parameters[parameter_name].value = scan_point
            expectation = self.generate_multiband_histogram(
                scale_to_pot=scale_to_pot, include_multisim_errors=True, use_sideband=True
            )
            delta_chi2 = []
            for j in range(n_trials):
                global_trial_index = n_trials * i + j
                # the seeds are chosen such that no seed is used twice
                pseudo_data = expectation.fluctuate(seed=global_trial_index).fluctuate_poisson(
                    seed=global_trial_index + number_of_samples
                )
                self.parameters[parameter_name].value = scan_point
                chi2_at_truth = chi_square(
                    pseudo_data.nominal_values,
                    expectation.nominal_values,
                    expectation.covariance_matrix,
                )
                m = self._get_minuit(pseudo_data, scale_to_pot=scale_to_pot)
                m.migrad()
                chi2_at_best_fit = m.fval
                delta_chi2.append(chi2_at_truth - chi2_at_best_fit)
            delta_chi2 = np.array(delta_chi2)
            results.append({"delta_chi2": delta_chi2, "scan_point": scan_point})
        return results

    def set_parameters(self, parameter_set, check_matching=True):
        """Set the parameters of the analysis.

        Parameters
        ----------
        parameter_set : ParameterSet
        check_matching : bool
            If True, check that the parameter set contains all the parameters
            of the analysis. If False, only the parameters that are present
            in the parameter set are set.
        """
        if check_matching:
            assert set(parameter_set.names) == set(
                self.parameters.names
            ), "Parameter sets must have the same parameters"
        for name in parameter_set.names:
            self.parameters[name].value = parameter_set[name].value

    @classmethod
    def from_toml(cls, file_path):
        # Try loading from file
        with open(file_path, "r") as file:
            configuration = toml.load(file)
        return cls(configuration)
