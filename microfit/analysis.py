"""Classes to run the LEE (and possibly other) analyses."""

import os
import logging
import sys
from typing import List, Optional, Tuple, Union, Dict, overload
from matplotlib import pyplot as plt, ticker
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
from functools import lru_cache


class MultibandAnalysis(object):
    def __init__(
        self,
        configuration=None,
        run_hist_generators: Optional[List[RunHistGenerator]] = None,
        constraint_channels=[],
        signal_channels=[],
        uncertainty_defaults=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.parameters = ParameterSet([])
        self.constraint_channels = []  # type: List[str]
        self.signal_channels = []  # type: List[str]
        # This attribute is a dictionary holding additional keyword arguments that are forwarded to the
        # `.plot` functino of the RunHistPlotter. It's a good idea to set the `run_title` property here
        # so that the runs can be shown next to the selection (e.g. "Runs 1-5").
        self.plotting_config = {}
        # The analysis may use several signal bands, but only one sideband.
        # For every signal band and the sideband, we are going to create a
        # RunHistGenerator object that can create histograms from the data.
        if configuration is None:
            assert run_hist_generators is not None
            self._init_from_generators(
                run_hist_generators, constraint_channels, signal_channels, uncertainty_defaults
            )
        else:
            self._init_from_config(configuration)
        self.channels = []
        for gen in self._run_hist_generators:
            self.channels.extend(gen.channels)
        for ch in self.signal_channels:
            assert ch in self.channels, f"Signal channel {ch} not found in analysis channels"
        for ch in self.constraint_channels:
            assert ch in self.channels, f"Constraint channel {ch} not found in analysis channels"
        # Adding the data_pot property just for compatibility with the RunHistPlotter
        self.data_pot = None
        self.plot_sideband = False
        assert isinstance(self.parameters, ParameterSet)
        assert len(self.signal_channels) > 0, "No signal channels given"

    def print_configuration(self):
        """Print a summary of the configuration of the analysis to console.

        This function should be used after completing the configuration
        to give the analyzer a confirmation that the analysis is configured
        as expected.
        """

        print("Analysis configuration:")
        print(f"  Signal channels: {self.signal_channels}")
        if len(self.constraint_channels) > 0:
            print(f"  Constraint channels: {self.constraint_channels}")
        else:
            print("  No constraint channels")
        print(self.parameters)  # relying on __repr__ of the ParameterSet class
        # also print plotting config if not empty
        if self.plotting_config:
            print("  Plotting configuration:")
            for key, value in self.plotting_config.items():
                print(f"    {key}: {value}")

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
        self.parameters = sum([g.parameters for g in self._run_hist_generators], ParameterSet([]))
        for gen in self._run_hist_generators:
            gen.mc_hist_generator._resync_parameters()
        self._check_shared_params([g.parameters for g in self._run_hist_generators])

    def _init_from_config(self, configuration):
        # The analysis may use several generators to produce a multi-channel histogram
        self._check_config(configuration)
        self._run_hist_generators: List[RunHistGenerator] = []
        generator_configurations = configuration["generator"]
        for config in generator_configurations:
            self._run_hist_generators.append(self.run_hist_generator_from_config(config))
        self.parameters = sum([g.parameters for g in self._run_hist_generators], ParameterSet([]))
        for gen in self._run_hist_generators:
            gen.mc_hist_generator._resync_parameters()
        self._check_shared_params([g.parameters for g in self._run_hist_generators])
        self.signal_channels = configuration["channels"]["signal_channels"]
        if "constraint_channels" in configuration["channels"]:
            self.constraint_channels = configuration["channels"]["constraint_channels"]
        else:
            self.constraint_channels = []
        self.uncertainty_defaults = configuration.get("uncertainty_defaults", {})
        self.plotting_config = configuration.get("plotting", {})

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

    def _check_channel_config(self, config):
        required_keys = ["variable", "n_bins", "limits", "selection", "preselection"]
        assert all(
            key in config for key in required_keys
        ), f"Configuration for channel must contain the keys {required_keys}"

    def _check_run_hist_config(self, config):
        required_keys = ["channel", "load_runs"]
        assert all(
            key in config for key in required_keys
        ), f"Configuration for RunHistGenerator must contain the keys {required_keys}"
        for channel_config in config["channel"]:
            self._check_channel_config(channel_config)

    def _check_config(self, config):
        assert isinstance(config, dict), "Configuration must be a dictionary"
        required_keys = ["generator", "channels"]
        assert all(
            key in config for key in required_keys
        ), f"Configuration must contain the keys {required_keys}"
        for generator_config in config["generator"]:
            self._check_run_hist_config(generator_config)

    def run_hist_generator_from_config(self, config: dict) -> RunHistGenerator:
        channel_configs = config["channel"]
        channel_binnings = []
        for channel_config in channel_configs:
            binning_cfg = {
                "variable": channel_config["variable"],
                "n_bins": channel_config["n_bins"],
                "limits": channel_config["limits"],
                "variable_tex": channel_config.get("variable_tex", None),
            }
            binning = Binning.from_config(**binning_cfg)
            binning.set_selection(
                selection=channel_config["selection"], preselection=channel_config["preselection"]
            )
            label = channel_config.get("label", None)
            binning.label = label or channel_config["selection"]
            channel_binnings.append(binning)
        binning = MultiChannelBinning(channel_binnings)

        rundata, weights, data_pot = load_runs(**config["load_runs"])
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
        parameters = None
        if "parameter" in config:
            parameters = ParameterSet.from_dict(config["parameter"])
        return RunHistGenerator(
            rundata,
            binning,
            data_pot=data_pot,
            parameters=parameters,
            mc_hist_generator_cls=mc_hist_generator_cls,
            **mc_hist_generator_kwargs,
        )

    def _apply_constraints(
        self, hist: MultiChannelHistogram, constraint_channels=None, total_prediction_hist=None
    ):
        """Apply the sideband constraint to the given histogram."""

        data_hist = self.generate_multiband_data_histogram()
        if data_hist is None:
            return hist
        constraint_channels = constraint_channels or self.constraint_channels
        total_prediction_hist = total_prediction_hist or hist
        assert len(constraint_channels) > 0, "No constraint channels given"
        constraint_data = data_hist[constraint_channels]
        hist = hist.update_with_measurement(
            measurement=constraint_data, central_value=total_prediction_hist
        )
        return hist

    def generate_multiband_histogram(
        self,
        include_multisim_errors: bool = False,
        use_sideband: bool = False,
        scale_to_pot: Optional[float] = None,
        constraint_channels: Optional[List[str]] = None,
        signal_channels: Optional[List[str]] = None,
        include_non_signal_channels: bool = False,
        include_ext: bool = True,
        extra_query=None,
        ms_columns=["weightsGenie", "weightsFlux", "weightsReint"],
        include_unisim_errors=True,
        include_stat_errors=True,
    ) -> MultiChannelHistogram:
        """Generate the combined MC histogram from all channels."""

        mc_hist_generators = [g.mc_hist_generator for g in self._run_hist_generators]

        # Try adding detvars in here

        mc_hist = HistogramGenerator.generate_joint_histogram(
            mc_hist_generators,
            include_multisim_errors=include_multisim_errors,
            extra_query=extra_query,
            ms_columns=ms_columns,
            include_unisim_errors=include_unisim_errors,
            include_stat_errors=include_stat_errors,
        )
        ext_hist_generators = [g.ext_hist_generator for g in self._run_hist_generators]
        joint_ext_hist = HistogramGenerator.generate_joint_histogram(
            ext_hist_generators,
            include_multisim_errors=False,
            extra_query=extra_query,
            include_stat_errors=include_stat_errors,
        )

        constraint_channels = constraint_channels or self.constraint_channels
        signal_channels = signal_channels or self.signal_channels
        all_channels = signal_channels + constraint_channels
        mc_hist = mc_hist[all_channels]
        assert isinstance(mc_hist, MultiChannelHistogram)
        joint_ext_hist = joint_ext_hist[all_channels]
        assert isinstance(joint_ext_hist, MultiChannelHistogram)


        total_prediction = mc_hist + joint_ext_hist
        if use_sideband and len(constraint_channels) > 0:
            # We have to be careful here to use the *full* prediction as the central
            # value when applying the constraint, not just the MC prediction.
            print("mc_hist=",type(mc_hist))
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
            assert isinstance(output_hist, MultiChannelHistogram)
        if scale_to_pot is not None:
            raise NotImplementedError("Scaling to POT not implemented in the Analysis class.")
        return output_hist

    def get_mc_hist(
        self,
        include_multisim_errors: bool = True,
        extra_query: Optional[str] = None,
        scale_to_pot: Optional[float] = None,
        use_sideband: bool = False,
        add_precomputed_detsys: bool = False,
    ) -> Union[Histogram, MultiChannelHistogram]:
        """Get the MC histogram. This function is solely for plotting purposes.

        The function is built to be compatible with the function of the same name in
        the RunHistGenerator class, so that the analysis can be dropped into the
        RunHistPlotter.
        """

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
            extra_query=extra_query,
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
            prior_errors[data_hist.bin_counts > 0] = 0
            data_hist.add_covariance(np.diag(prior_errors))
        if scale_to_pot is not None:
            raise NotImplementedError("Scaling to POT not implemented in the Analysis class.")
        channels = self.constraint_channels if self.plot_sideband else self.signal_channels
        return data_hist[channels]

    def get_mc_hists(
        self, category_column="dataset_name", include_multisim_errors=False,add_precomputed_detsys=False, scale_to_pot=None
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
                category_column=category_column, include_multisim_errors=include_multisim_errors,add_precomputed_detsys=add_precomputed_detsys
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
        for i, key in enumerate(all_categories):
            if key not in all_keys:
                continue
            combined_channels[key] = MultiChannelHistogram.from_histograms(
                [hist_dict[key] for hist_dict in histograms]
            )
            combined_channels[key] = combined_channels[key][channels]
            try:
                combined_channels[key].color = category_definitions.get_category_color(
                    category_column, key
                )
                combined_channels[key].tex_string = category_definitions.get_category_label(
                    category_column, key
                )
            except KeyError:
                combined_channels[key].color = f"C{i}"
                combined_channels[key].tex_string = key
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

    @lru_cache(maxsize=1)
    def generate_multiband_data_histogram(
        self, impute_blinded_channels=False
    ) -> Optional[MultiChannelHistogram]:
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
                # extracting one channel is known to return a Histogram
                data_hist.append_empty_channel(full_hist[channel].binning)  # type: ignore
        return data_hist

    def _get_pot_for_channel(self, channel):
        """Get the POT for the given channel."""

        # Iterate through run hist generators. When the channel is found in a
        # generator, return the POT of that generator.
        for gen in self._run_hist_generators:
            if channel in gen.channels:
                return gen.data_pot
        raise ValueError(f"Channel {channel} not found in analysis")

    def _get_channel_is_blinded(self, channel):
        """Get whether the given channel is blinded."""

        # Iterate through run hist generators. When the channel is found in a
        # generator, return the POT of that generator.
        for gen in self._run_hist_generators:
            if channel in gen.channels:
                return gen.is_blinded
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

        if separate_figures:
            for channel in channels:
                if channel in self.plotting_config:
                    default_kwargs = self.plotting_config[channel].copy()
                    default_kwargs.update(kwargs)
                else:
                    default_kwargs = kwargs
                ax, _ = RunHistPlotter(self).plot(
                    category_column=category_column,
                    channel=channel,
                    data_pot=self._get_pot_for_channel(channel),
                    show_data=not self._get_channel_is_blinded(channel),
                    **default_kwargs,
                )
                if save_path is not None:
                    assert ax is not None, "Cannot save figure when ax is None"
                    fig = ax.figure
                    fig.savefig(os.path.join(save_path, filename_format.format(channel)))
                    plt.close(fig)
            return
        show_data_mc_ratio = kwargs.get("show_data_mc_ratio", False)
        n_rows = 2 if show_data_mc_ratio else 1
        n_cols = n_channels
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 6, 5 if n_rows == 1 else 8),
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
                data_pot=self._get_pot_for_channel(channel),
                show_data=not self._get_channel_is_blinded(channel),
                **kwargs,
            )
        return fig, axes

    def plot_signals(self, category_column="paper_category", **kwargs):
        """
        Plot the signals for each category.

        Parameters
        ----------
        category_column : str
            The column in the dataset that contains the categories.
        separate_figures : bool
            If True, each channel is plotted in a separate figure.
        save_path : str
            If given, the figures are saved to this path.
        filename_format : str
            The format string for the filename. The channel name is inserted into the
            format string by calling `format` on the string with the channel name as
            the argument. For example, the default format string is "analysis_{}.pdf",
            which will result in filenames like "analysis_nue.pdf".
        """
        return self._plot_bands(category_column, plot_signals=True, **kwargs)

    def plot_sidebands(self, category_column="category", **kwargs):
        """
        Plot the sidebands for each category.

        Parameters
        ----------
        category_column : str
            The column in the dataset that contains the categories.
        separate_figures : bool
            If True, each channel is plotted in a separate figure.
        save_path : str
            If given, the figures are saved to this path.
        filename_format : str
            The format string for the filename. The channel name is inserted into the
            format string by calling `format` on the string with the channel name as
            the argument. For example, the default format string is "analysis_{}.pdf",
            which will result in filenames like "analysis_nue.pdf".
        """
        # When plotting sidebands, we of course never want to use the sideband as a constraint
        kwargs.pop("use_sideband", None)
        return self._plot_bands(category_column, plot_signals=False, use_sideband=False, **kwargs)

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
            add_precomputed_detsys=add_precomputed_detsys
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
        self,
        h0_params: ParameterSet,
        h1_params: ParameterSet,
        sensitivity_only: bool = False,
        n_trials: int = 1000,
        scale_to_pot: Optional[float] = None,
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
        from tqdm import tqdm

        if scale_to_pot is not None:
            assert sensitivity_only, "Can only scale to POT when calculating sensitivity"
        assert set(h0_params.names) == set(
            h1_params.names
        ), "Parameter sets must have the same parameters"
        self.set_parameters(h0_params, check_matching=True)
        print("Generating H0 histogram")
        # generate the multiband histogram
        h0_hist = self.generate_multiband_histogram(
            include_multisim_errors=True, use_sideband=True, scale_to_pot=scale_to_pot
        )
        print("Generating H1 histogram")
        self.set_parameters(h1_params, check_matching=True)
        h1_hist = self.generate_multiband_histogram(
            include_multisim_errors=True, use_sideband=True, scale_to_pot=scale_to_pot
        )

        test_stat_h0 = []
        test_stat_h1 = []

        def test_statistic(observation):
            chi2_h0 = chi_square(
                observation.bin_counts, h0_hist.bin_counts, h0_hist.covariance_matrix
            )
            chi2_h1 = chi_square(
                observation.bin_counts, h1_hist.bin_counts, h1_hist.covariance_matrix
            )
            return chi2_h0 - chi2_h1

        for i in tqdm(range(n_trials), desc="Generating pseudo-experiments"):
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
            data_hist.bin_counts, h0_hist.bin_counts, h0_hist.covariance_matrix
        )
        results["pval_h0"] = np.sum(test_stat_h0 > real_data_ts) / n_trials
        # calculate the p-value of the observed data under H1
        results["chi2_h1"] = chi_square(
            data_hist.bin_counts, h1_hist.bin_counts, h1_hist.covariance_matrix
        )
        results["pval_h1"] = np.sum(test_stat_h1 > real_data_ts) / n_trials

        return results

    @lru_cache(maxsize=1000)
    def _get_hist_at_parameter_values(self, **kwargs):
        """Get the histogram at the given parameter values.

        This function is cached, so that the histogram is only generated once.
        Be sure to clear the cache if any settings of the analysis change.
        """

        assert set(kwargs.keys()) == set(
            self.parameters.names
        ), "Must give values for all parameters"
        for name in kwargs:
            self.parameters[name].value = kwargs[name]
        return self.generate_multiband_histogram(include_multisim_errors=True, use_sideband=True)

    def _get_minuit(self, observed_hist, scale_to_pot=None, reset_cache=True):
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

        # clear the cache just to be safe
        if reset_cache:
            self._get_hist_at_parameter_values.cache_clear()

        def loss(*args):
            # set the parameters
            parameter_kwargs = {name: value for name, value in zip(self.parameters.names, args)}
            # generate the histogram
            generated_hist = self._get_hist_at_parameter_values(**parameter_kwargs)
            # calculate the chi-square
            return chi_square(
                observed_hist.bin_counts,
                generated_hist.bin_counts,
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

    def scan_chi2(self, parameter_name, scan_points, data=None):
        """Perform a scan of the chi-square as a function of the given parameter."""

        data = data or self.generate_multiband_data_histogram()
        assert data is not None, "Cannot scan chi-square when data is None"
        for channel in self.signal_channels:
            if channel not in data.channels:
                raise ValueError(f"Channel {channel} not found in data histogram")
        data = data[self.signal_channels]
        results = []
        for scan_point in scan_points:
            self.parameters[parameter_name].value = scan_point
            expectation = self.generate_multiband_histogram(
                include_multisim_errors=True, use_sideband=True
            )
            chi2 = chi_square(
                data.bin_counts, expectation.bin_counts, expectation.covariance_matrix
            )
            results.append(chi2)
        results = np.array(results)
        return results

    def scan_asimov_sensitivity(self, parameter_name, injection_point, scan_points):
        """Perform a sensitivity scan using the expectation value as the data.

        The parameter is set to the injection point and then the expectation
        value is calculated. Then, the parameter is set to the scan points and
        the chi-square is calculated for each point.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to scan.
        injection_point : float
            The value of the parameter to use as the expectation value.
        scan_points : array-like
            The values of the parameter to scan.

        Returns
        -------
        np.ndarray
            The chi-square values for each scan point.
        """

        self.parameters[parameter_name].value = injection_point
        expectation = self.generate_multiband_histogram(
            include_multisim_errors=True, use_sideband=True
        )
        return self.scan_chi2(parameter_name, scan_points, data=expectation)

    def scan_asimov_fc_sensitivity(
        self,
        scan_points=None,
        parameter_name=None,
        n_trials=100,
        fc_scan_results: Optional[dict] = None,
        **fc_scan_kwargs,
    ):
        if fc_scan_results is None:
            assert scan_points is not None, "Either fc_scan_results or scan_points must be given"
            assert (
                parameter_name is not None
            ), "Either fc_scan_results or parameter_name must be given"
            fc_scan_results = self.fc_scan(
                parameter_name, scan_points, n_trials=n_trials, **fc_scan_kwargs
            )
        elif scan_points is not None:
            if "scan_points" not in fc_scan_results:
                assert np.allclose(
                    scan_points,
                    np.array([result["scan_point"] for result in fc_scan_results["results"]]),
                ), "incomptible scan points"
                fc_scan_results["scan_points"] = scan_points
        elif parameter_name is not None:
            assert parameter_name == fc_scan_results["parameter_name"]

        scan_points = np.array([result["scan_point"] for result in fc_scan_results["results"]])
        parameter_name = fc_scan_results["parameter_name"]
        assert set({"parameter_name", "scan_points", "results"}).issubset(
            set(fc_scan_results.keys())
        )

        def inverse_quantile(x, q):
            return sum(x < q) / len(x)

        print(f"Calculating Asimov sensitivity for {len(scan_points)} points...")
        for result in fc_scan_results["results"]:
            result["asimov_chi2"] = self.scan_asimov_sensitivity(
                parameter_name,
                injection_point=result["scan_point"],
                scan_points=scan_points,
            )
            # For every value in the asimov chi-square, we want to know the p-value.
            # To get it we have to loop over the fc scan we did earlier for all the points.
            result["pval"] = [
                inverse_quantile(r["delta_chi2"], asim_chi2)
                for (r, asim_chi2) in zip(fc_scan_results["results"], result["asimov_chi2"])
            ]
        # For convenience, we also already compute the 2D map of p-values and return them as well.
        # These can be used for plotting.
        fc_scan_results["pval_map"] = np.array(
            [result["pval"] for result in fc_scan_results["results"]]
        ).T
        X, Y = np.meshgrid(scan_points, scan_points)
        fc_scan_results["measured_map"] = X
        fc_scan_results["truth_map"] = Y
        return fc_scan_results

    @classmethod
    def plot_fc_scan_results(
        cls,
        fc_scan_results,
        ax=None,
        parameter_tex=None,
        levels=[0.0, 0.68, 0.9, 0.95, 1.0],
        **kwargs,
    ):
        """Plot the results of an FC scan."""

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.figure
        pval_map = fc_scan_results["pval_map"]
        X, Y = fc_scan_results["measured_map"], fc_scan_results["truth_map"]
        contour = ax.contourf(X, Y, pval_map, levels=levels, cmap="Blues_r")
        cbar = fig.colorbar(
            contour, format=ticker.FuncFormatter(lambda x, pos: f"{x * 100:.0f}%"), drawedges=True
        )
        cbar.ax.set_ylabel("Confidence Level")
        ax.contour(X, Y, pval_map, levels=levels, colors="k", linewidths=0.5)
        # add line where x=y
        ax.plot(
            [min(X.min(), Y.min()), max(X.max(), Y.max())],
            [min(X.min(), Y.min()), max(X.max(), Y.max())],
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
        if parameter_tex is None:
            parameter_tex = fc_scan_results["parameter_name"]
        ax.set_xlabel(rf"Measured {parameter_tex}")
        ax.set_ylabel(rf"True {parameter_tex}")

        return fig, ax

    def _fit_to_data_migrad(
        self,
        data=None,
        return_migrad=False,
        reset_cache=True,
    ):
        data = data or self.generate_multiband_data_histogram()
        assert data is not None, "Cannot fit to data when data is None"
        for channel in self.signal_channels:
            if channel not in data.channels:
                raise ValueError(f"Channel {channel} not found in data histogram")
        data = data[self.signal_channels]
        m = self._get_minuit(data, reset_cache=reset_cache)
        m.migrad()
        best_fit_parameters = self.parameters.copy()
        if return_migrad:
            return m.fval, best_fit_parameters, m
        return m.fval, best_fit_parameters

    def _fit_to_data_grid_scan(
        self,
        data: Optional[Histogram] = None,
        fit_grid: Dict[str, np.ndarray] = {},
        reset_cache=True,
    ):
        """Fit the data to a grid of parameter values.

        The grid should be given as a dictionary where keys are the parameter
        names and values are the scan points for that parameter.
        The chi-square will be calculated for each combination of parameter values. The function then
        returns the chi-square grid and the best fit parameters.

        This function can be potentially fast for repeated fits since it uses caching.
        """
        data = data or self.generate_multiband_data_histogram()
        assert data is not None, "Cannot fit to data when data is None"
        assert isinstance(data, MultiChannelHistogram)
        for channel in self.signal_channels:
            if channel not in data.channels:
                raise ValueError(f"Channel {channel} not found in data histogram")
        data = data[self.signal_channels]
        assert fit_grid is not None, "Must give fit grid"
        assert set(fit_grid.keys()) == set(
            self.parameters.names
        ), "Must give fit grid for all parameters"

        if reset_cache:
            self._get_hist_at_parameter_values.cache_clear()

        def loss(*args):
            # set the parameters, make sure to use the same order as in the fit grid
            parameter_kwargs = {name: value for name, value in zip(fit_grid.keys(), args)}
            # generate the histogram
            generated_hist = self._get_hist_at_parameter_values(**parameter_kwargs)
            # calculate the chi-square
            return chi_square(
                data.bin_counts,
                generated_hist.bin_counts,
                generated_hist.covariance_matrix,
            )

        loss = np.vectorize(loss)  # simplify broadcasting
        # Evaluate the loss over a meshgrid of parameter values
        meshgrid = np.meshgrid(*fit_grid.values())
        chi2_grid = loss(*meshgrid)
        # Find the minimum
        min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
        best_fit_parameter_dict = {
            name: meshgrid[i][min_index] for i, name in enumerate(fit_grid.keys())
        }
        best_chi2 = chi2_grid[min_index]
        for name in best_fit_parameter_dict:
            self.parameters[name].value = best_fit_parameter_dict[name]
        best_fit_parameters = self.parameters.copy()
        # return the chi-square of the best fit and the best fit parameters
        return best_chi2, best_fit_parameters

    def fit_to_data(self, data, method="migrad", **kwargs):
        assert method in ["migrad", "grid_scan"], "Invalid fitting method"
        if method == "migrad":
            return self._fit_to_data_migrad(data, **kwargs)
        elif method == "grid_scan":
            fit_grid = kwargs.pop("fit_grid")
            return self._fit_to_data_grid_scan(data, fit_grid=fit_grid, **kwargs)

    def fc_scan(
        self,
        parameter_name,
        scan_points,
        n_trials=100,
        scale_to_pot=None,
        fit_method="migrad",
        **fit_kwargs,
    ):
        """Perform a Feldman-Cousins scan of the given parameter."""
        from tqdm import tqdm

        # For every point in the scan, we assume that it is the truth and
        # then create pseudo-experiments by sampling from the covariance
        # matrix. Then, we calculate the best fit for every pseudo-experiment
        # and record the difference in chi-square between the best fit and
        # a fit where the parameter in question is fixed to the assumed truth.
        results = []
        print(f"Running FC scan over {len(scan_points)} points in {parameter_name}...")
        self._get_hist_at_parameter_values.cache_clear()
        number_of_samples = len(scan_points) * n_trials
        with tqdm(total=len(scan_points), desc="Scanning") as pbar1:
            for i, scan_point in enumerate(scan_points):
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
                        pseudo_data.bin_counts,
                        expectation.bin_counts,
                        expectation.covariance_matrix,
                    )
                    # ignore typing here, this requires overloading that is not yet
                    # available in Python 3.7 because it lacks the "Literal" type
                    chi2_at_best_fit = self.fit_to_data(
                        pseudo_data, reset_cache=False, method=fit_method, **fit_kwargs
                    )[  # type: ignore
                        0
                    ]  # type: ignore
                    delta_chi2.append(chi2_at_truth - chi2_at_best_fit)
                delta_chi2 = np.array(delta_chi2)
                results.append({"delta_chi2": delta_chi2, "scan_point": scan_point})
                pbar1.update()
        return {"parameter_name": parameter_name, "scan_points": scan_points, "results": results}

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
