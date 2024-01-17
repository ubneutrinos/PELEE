"""Classes to run the LEE (and possibly other) analyses."""

import os
import logging
from typing import List, Optional, Union
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
    MultiChannelHistogram,
)
from microfit.parameters import ParameterSet
from microfit.run_plotter import RunHistPlotter
from microfit import signal_generators
from microfit.statistics import sideband_constraint_correction, chi_square
from microfit import category_definitions


class MultibandAnalysis(object):
    def __init__(
        self,
        configuration=None,
        run_hist_generators=None,
        constraint_channels=[],
        signal_channels=[],
        uncertainty_defaults=None,
    ):
        self.logger = logging.getLogger(__name__)
        # The analysis may use several signal bands, but only one sideband.
        # For every signal band and the sideband, we are going to create a
        # RunHistGenerator object that can create histograms from the data.
        if configuration is None:
            self._init_from_generators(
                run_hist_generators, constraint_channels, signal_channels, uncertainty_defaults
            )
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
        # Adding the data_pot property just for compatibility with the RunHistPlotter
        self.data_pot = None
        self.plot_sideband = False

    def _init_from_generators(
        self,
        run_hist_generators: List[RunHistGenerator],
        constraint_channels: List[str],
        signal_channels: List[str],
        uncertainty_defaults=None,
    ):
        self._run_hist_generators = run_hist_generators
        self.constraint_channels = constraint_channels
        self.uncertainty_defaults = uncertainty_defaults or {}
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

    def _apply_constraints(
        self, hist: MultiChannelHistogram, constraint_channels=None, total_prediction_hist=None
    ):
        """Apply the sideband constraint to the given histogram."""

        data_hist = self.generate_multiband_data_histogram()
        if data_hist is None:
            return hist
        constraint_channels = constraint_channels or self.constraint_channels
        total_prediction_hist = total_prediction_hist or hist
        for constraint_channel in constraint_channels:
            constraint_hist = data_hist[constraint_channel]
            prediction_hist = total_prediction_hist[constraint_channel]
            hist = hist.update_with_measurement(
                constraint_channel,
                measurement=constraint_hist.nominal_values,
                central_value=prediction_hist.nominal_values,
            )
        return hist

    def generate_multiband_histogram(
        self,
        include_multisim_errors=False,
        use_sideband=False,
        scale_to_pot=None,
        constraint_channels=None,
        signal_channels=None,
        include_non_signal_channels=False,
        include_ext=True,
    ):
        """Generate the combined MC histogram from all channels."""

        mc_hist_generators = [g.mc_hist_generator for g in self._run_hist_generators]
        mc_hist = HistogramGenerator.generate_joint_histogram(
            mc_hist_generators, include_multisim_errors=include_multisim_errors
        )
        ext_hist_generators = [g.ext_hist_generator for g in self._run_hist_generators]
        joint_ext_hist = HistogramGenerator.generate_joint_histogram(
            ext_hist_generators, include_multisim_errors=False
        )

        constraint_channels = constraint_channels or self.constraint_channels
        signal_channels = signal_channels or self.signal_channels
        all_channels = signal_channels + constraint_channels
        mc_hist = mc_hist[all_channels]
        joint_ext_hist = joint_ext_hist[all_channels]

        total_prediction = mc_hist + joint_ext_hist
        if use_sideband:
            # We have to be careful here to use the *full* prediction as the central
            # value when applying the constraint, not just the MC prediction.
            mc_hist = self._apply_constraints(
                mc_hist,
                constraint_channels=constraint_channels,
                total_prediction_hist=total_prediction,
            )
        if include_ext:
            output_hist = mc_hist + joint_ext_hist[mc_hist.channels]
        else:
            output_hist = mc_hist
        if not include_non_signal_channels:
            output_hist = output_hist[signal_channels]
        if scale_to_pot is not None:
            raise NotImplementedError("Scaling to POT not implemented in the Analysis class.")
        return output_hist

    def get_mc_hist(
        self,
        include_multisim_errors: Optional[bool] = None,
        extra_query: Optional[str] = None,
        scale_to_pot: Optional[float] = None,
        use_sideband: Optional[bool] = None,
        add_precomputed_detsys: bool = False,
    ) -> Union[Histogram, MultiChannelHistogram]:
        """Get the MC histogram. This function is solely for plotting purposes.

        The function is built to be compatible with the function of the same name in
        the RunHistGenerator class, so that the analysis can be dropped into the
        RunHistPlotter.
        """
        if extra_query is not None:
            raise NotImplementedError("extra_query not implemented in the Analysis class.")
        if add_precomputed_detsys:
            raise NotImplementedError(
                "add_precomputed_detsys not implemented in the Analysis class."
            )
        output = self.generate_multiband_histogram(
            include_multisim_errors=include_multisim_errors,
            use_sideband=use_sideband,
            scale_to_pot=scale_to_pot,
            include_ext=False,
            include_non_signal_channels=True,
        )
        channels = self.constraint_channels if self.plot_sideband else self.signal_channels
        return output[channels]

    def get_data_hist(
        self, type="data", add_error_floor=False, scale_to_pot=None, smooth_ext_histogram=False
    ):
        """Get the data histogram. This function is solely for plotting purposes.

        The function is built to be compatible with the function of the same name in
        the RunHistGenerator class, so that the analysis can be dropped into the
        RunHistPlotter.
        """
        if smooth_ext_histogram:
            raise NotImplementedError("smooth_ext_histogram not implemented in the Analysis class.")
        if type == "data":
            data_hist = self.generate_multiband_data_histogram(impute_blinded_channels=True)
        elif type == "ext":
            data_hist = self.generate_multiband_ext_histogram()
        else:
            raise ValueError(f"Invalid data type {type}")
        if data_hist is None:
            return None
        if add_error_floor:
            prior_errors = np.ones(data_hist.n_bins) * 1.4**2
            prior_errors[data_hist.nominal_values > 0] = 0
            data_hist.add_covariance(np.diag(prior_errors))
        if scale_to_pot is not None:
            raise NotImplementedError("Scaling to POT not implemented in the Analysis class.")
        channels = self.constraint_channels if self.plot_sideband else self.signal_channels
        return data_hist[channels]

    def get_mc_hists(
        self, category_column="dataset_name", include_multisim_errors=False, scale_to_pot=None
    ):
        """Get the MC histograms, split by category. This function is solely for plotting purposes.

        The function is built to be compatible with the function of the same name in
        the RunHistGenerator class, so that the analysis can be dropped into the
        RunHistPlotter.
        """

        if scale_to_pot is not None:
            raise NotImplementedError("Scaling to POT not implemented in the Analysis class.")
        # I almost want to apologize for how complicated this function is. One issue is that
        # not all channels will have every category of events, so we have to carefully impute
        # the missing histograms. We also can't directly go in and grab the dataframes from
        # inside the histogram generators, because they might be sub-classes of the default
        # ones that work differently.
        histograms = [
            g.get_mc_hists(
                category_column=category_column, include_multisim_errors=include_multisim_errors
            )
            for g in self._run_hist_generators
        ]
        # We should now have a list of dicts of histograms, where each dict is keyed by category
        # and each histogram is a MultiChannelHistogram. The issue now is that not every category
        # might be present in every dict. What we need to do is to impute empty histograms for
        # missing categories.
        # First we need the superset of keys of all the dictionaries.
        all_keys = set()
        for hist_dict in histograms:
            all_keys.update(hist_dict.keys())
        reference_states = dict()
        for key in all_keys:
            for hist_dict in histograms:
                if key in hist_dict:
                    reference_states[key] = hist_dict[key].to_dict()
                    break
        for hist_dict in histograms:
            empty_hist_state = hist_dict[list(hist_dict.keys())[0]].to_dict()
            empty_hist_state["bin_counts"] = np.zeros_like(empty_hist_state["bin_counts"])
            empty_hist_state["covariance_matrix"] = np.zeros_like(
                empty_hist_state["covariance_matrix"]
            )
            for key in all_keys:
                if key not in hist_dict:
                    # This is a bit of a Frankenstein monster: We use the binning of this channel, but
                    # apply the label, color, etc. of the reference channel.
                    empty_hist_state["label"] = reference_states[key]["label"]
                    empty_hist_state["plot_color"] = reference_states[key]["plot_color"]
                    empty_hist_state["plot_hatch"] = reference_states[key]["plot_hatch"]
                    empty_hist_state["tex_string"] = reference_states[key]["tex_string"]
                    try:
                        hist_dict[key] = MultiChannelHistogram.from_dict(empty_hist_state)
                    except KeyError:
                        # An error indicates that the state was not a MultiChannelHistogram,
                        # but a Histogram.
                        hist_dict[key] = Histogram.from_dict(empty_hist_state)
        # Now, all the dicts should have the same keys, assert this
        for hist_dict in histograms:
            assert set(hist_dict.keys()) == set(all_keys)
        combined_channels = dict()
        if category_column != "dataset_name":
            all_categories = category_definitions.get_categories(category_column)
        else:
            all_categories = all_keys
        channels = self.constraint_channels if self.plot_sideband else self.signal_channels
        for key in all_categories:
            if key not in all_keys:
                continue
            combined_channels[key] = MultiChannelHistogram.from_histograms(
                [hist_dict[key] for hist_dict in histograms]
            )
            combined_channels[key] = combined_channels[key][channels]
            combined_channels[key].color = category_definitions.get_category_color(
                category_column, key
            )
            combined_channels[key].tex_string = category_definitions.get_category_label(
                category_column, key
            )
        return combined_channels

    def generate_multiband_ext_histogram(self):
        """Generate a combined histogram from all ext channels."""

        ext_hist_generators = [g.ext_hist_generator for g in self._run_hist_generators]
        if len(ext_hist_generators) == 0:
            return None
        ext_hist = HistogramGenerator.generate_joint_histogram(
            ext_hist_generators, include_multisim_errors=False
        )
        ext_hist.color = "k"
        ext_hist.tex_string = "EXT"
        ext_hist.hatch = "///"
        return ext_hist

    def generate_multiband_data_histogram(self, impute_blinded_channels=False):
        """Generate a combined histogram from all unblinded data channels."""

        unblinded_generators = [
            g.data_hist_generator for g in self._run_hist_generators if not g.is_blinded
        ]
        if len(unblinded_generators) == 0:
            return None
        data_hist = HistogramGenerator.generate_joint_histogram(
            unblinded_generators, include_multisim_errors=False
        )
        if not impute_blinded_channels:
            return data_hist
        # If we need to impute the blinded channels, we generate a histogram with all channels,
        # set everything to zero and then update the channels that are not blinded.
        full_hist = self.generate_multiband_histogram()
        for channel in full_hist.channels:
            if channel not in data_hist.channels:
                data_hist.append_empty_channel(full_hist[channel].binning)
        return data_hist

    def _get_pot_for_channel(self, channel):
        """Get the POT for the given channel."""

        # Iterate through run hist generators. When the channel is found in a
        # generator, return the POT of that generator.
        for gen in self._run_hist_generators:
            if channel in gen.channels:
                return gen.data_pot
        raise ValueError(f"Channel {channel} not found in analysis")

    def _plot_bands(
        self,
        category_column,
        plot_signals=True,
        separate_figures=False,
        save_path=None,
        filename_format="analysis_{}.pdf",
        **kwargs,
    ):
        if plot_signals:
            channels = self.signal_channels
            self.plot_sideband = False
        else:
            channels = self.constraint_channels
            self.plot_sideband = True

        n_channels = len(channels)
        if n_channels == 0:
            return None
        show_data_mc_ratio = kwargs.get("show_data_mc_ratio", False)
        if separate_figures:
            for channel in channels:
                ax, _ = RunHistPlotter(self).plot(
                    category_column=category_column,
                    channel=channel,
                    data_pot=self._get_pot_for_channel(channel),
                    **kwargs,
                )
                if save_path is not None:
                    fig = ax.figure
                    fig.savefig(os.path.join(save_path, filename_format.format(channel)))
                    plt.close(fig)
            return
        n_rows = 2 if show_data_mc_ratio else 1
        n_cols = n_channels
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 8, 5 if n_rows == 1 else 8),
            squeeze=False,
            gridspec_kw={"height_ratios": [3, 1] if show_data_mc_ratio else [1]},
            constrained_layout=True,
        )
        plotter = RunHistPlotter(self)
        self.plot_sideband = not plot_signals
        for channel in channels:
            plotter.plot(
                category_column=category_column,
                ax=axes[0, channels.index(channel)],
                ax_ratio=axes[1, channels.index(channel)] if show_data_mc_ratio else None,
                channel=channel,
                **kwargs,
            )
        return fig, axes

    def plot_signals(self, category_column="paper_category", signals=None, **kwargs):
        return self._plot_bands(category_column, plot_signals=True, **kwargs)

    def plot_sidebands(self, category_column="category", **kwargs):
        return self._plot_bands(category_column, plot_signals=False, **kwargs)

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
