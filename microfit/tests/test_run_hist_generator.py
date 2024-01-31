from typing import Union
import unittest
import pandas as pd
import numpy as np
from microfit.histogram import Binning
from microfit.histogram import RunHistGenerator
from microfit.histogram import MultiChannelBinning
from microfit.parameters import ParameterSet, Parameter
from microfit.signal_generators import SignalOverBackgroundGenerator


class TestRunHistGenerator(unittest.TestCase):
    def make_test_binning(
        self, multichannel: bool = False, with_query: bool = False, second_query: str = "matching"
    ) -> Union[Binning, MultiChannelBinning]:
        first_channel_binning = Binning("energy", np.linspace(0, 3, 12), "energy")
        if with_query:
            first_channel_binning.selection_query = "bdt > 0.5"
        if not multichannel:
            return first_channel_binning

        second_channel_binning = Binning("angle", np.linspace(0, 3.14, 13), "angle")
        second_query_string = {
            "matching": "bdt > 0.5",
            "non_matching": "bdt < 0.5",
            "overlapping": "bdt < 0.8",
        }[second_query]
        if with_query:
            second_channel_binning.selection_query = second_query_string

        binning = MultiChannelBinning([first_channel_binning, second_channel_binning])
        return binning

    def make_dataframe(
        self,
        n_samples=1000,
        data_like=False,
        with_multisim=False,
        weights_scale=1.0,
        signal_flag=False,
    ):
        df = pd.DataFrame()
        np.random.seed(0)
        # sampling energy from a slightly more realistic distribution
        df["energy"] = np.random.lognormal(0, 0.5, n_samples)
        df["angle"] = np.random.uniform(0, 3.14, n_samples)
        df["bdt"] = np.random.uniform(0, 1, n_samples)
        if data_like:
            df["weights"] = np.ones(n_samples)
            # data never contains multisim weights
            return df
        else:
            df["weights"] = np.random.uniform(0, 1, n_samples)
            df["weights"] *= weights_scale / df["weights"].mean()
        df["is_signal"] = signal_flag
        if not with_multisim:
            return df
        # The 'weights_no_tune' column is used to calculate multisim uncertainties for GENIE
        # variables. For testing purposes, we just set it to the same as 'weights'.
        df["weights_no_tune"] = df["weights"]
        n_universes = 11
        for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
            df[ms_column] = [
                1000 * np.random.normal(loc=1, size=n_universes, scale=0.1) for _ in range(len(df))
            ]
        # Also add unisim "knob" weights
        knob_v = [
            "knobRPA",
            "knobCCMEC",
            "knobAxFFCCQE",
            "knobVecFFCCQE",
            "knobDecayAngMEC",
            "knobThetaDelta2Npi",
        ]
        for knob in knob_v:
            df[f"{knob}up"] = 1.01
            df[f"{knob}dn"] = 0.99
        return df

    def test_get_data_hist(self):
        # Create some mock data
        rundata_dict = {
            "data": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]}),
            "mc": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]}),
            "ext": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]}),
        }
        binning = Binning("x", np.array([0, 1, 2, 3, 4]), "x")
        generator = RunHistGenerator(rundata_dict, binning, data_pot=1.0)

        # Test getting the data histogram
        data_hist = generator.get_data_hist()
        assert data_hist is not None
        np.testing.assert_array_equal(data_hist.bin_counts, [0, 1, 1, 1])

        # Test getting the EXT histogram
        ext_hist = generator.get_data_hist(type="ext")
        assert ext_hist is not None
        np.testing.assert_array_equal(ext_hist.bin_counts, [0, 1, 1, 1])

        # Test scaling the EXT histogram
        ext_hist_scaled = generator.get_data_hist(type="ext", scale_to_pot=2)
        assert ext_hist_scaled is not None
        np.testing.assert_array_equal(ext_hist_scaled.bin_counts, [0, 2, 2, 2])

    def test_multiband_histogram_equivalence(self):
        def run_test_with_mc_gen_class(
            mc_hist_generator_cls=None,
            parameters=None,
            mc_hist_generator_kwargs={},
            enable_cache=True,
            strict_covar_checking=True,
        ):
            mock_rundata = {
                "mc": self.make_dataframe(n_samples=900, weights_scale=0.1, with_multisim=True),
                "data": self.make_dataframe(n_samples=100, data_like=True),
                "ext": self.make_dataframe(n_samples=10, data_like=True),
                "signal": self.make_dataframe(
                    n_samples=100, weights_scale=0.1, signal_flag=True, with_multisim=True
                ),
            }
            # If and only if the selections between channels are disjoint, generating a multichannel
            # histogram should be equivalent to joining the histograms of the individual channels.
            # This will carry over those bin-to-bin correlations that are induced by the multisim weights.
            multi_binning = self.make_test_binning(
                multichannel=True, with_query=True, second_query="non_matching"
            )
            assert isinstance(multi_binning, MultiChannelBinning)

            run_hist_generator = RunHistGenerator(
                mock_rundata,
                multi_binning,
                data_pot=1.0,
                sideband_generator=None,
                uncertainty_defaults=None,
                mc_hist_generator_cls=mc_hist_generator_cls,
                parameters=parameters,
                enable_cache=enable_cache,
                **mc_hist_generator_kwargs,
            )
            run_hist_generator_energy = RunHistGenerator(
                mock_rundata,
                multi_binning["energy"],
                data_pot=1.0,
                sideband_generator=None,
                uncertainty_defaults=None,
                mc_hist_generator_cls=mc_hist_generator_cls,
                parameters=parameters,
                enable_cache=enable_cache,
                **mc_hist_generator_kwargs,
            )
            run_hist_generator_angle = RunHistGenerator(
                mock_rundata,
                multi_binning["angle"],
                data_pot=1.0,
                sideband_generator=None,
                uncertainty_defaults=None,
                mc_hist_generator_cls=mc_hist_generator_cls,
                parameters=parameters,
                enable_cache=enable_cache,
                **mc_hist_generator_kwargs,
            )

            from microfit.histogram import HistogramGenerator

            mc_hist_gen_multichannel = run_hist_generator.mc_hist_generator
            multichannel_hist = mc_hist_gen_multichannel.generate(include_multisim_errors=True)

            mc_hist_gen_energy = run_hist_generator_energy.mc_hist_generator
            mc_hist_gen_angle = run_hist_generator_angle.mc_hist_generator
            joined_hist = HistogramGenerator.generate_joint_histogram(
                [mc_hist_gen_energy, mc_hist_gen_angle], include_multisim_errors=True
            )
            # There are very small numerical differences in the covariance matrices, but the equality check
            # in the Histogram class only checks for approximate equality, so this should be fine.
            np.testing.assert_array_almost_equal(
                multichannel_hist.bin_counts, joined_hist.bin_counts
            )
            np.testing.assert_array_almost_equal(
                multichannel_hist.covariance_matrix,
                joined_hist.covariance_matrix,
                decimal=6 if strict_covar_checking else 1,
            )

            # Another sanity check: The diagonal blocks of the covariance matrix should be the same
            # as if we had generated the histograms separately.
            energy_hist = mc_hist_gen_energy.generate(include_multisim_errors=True)
            angle_hist = mc_hist_gen_angle.generate(include_multisim_errors=True)
            np.testing.assert_array_almost_equal(
                energy_hist.covariance_matrix,
                joined_hist.covariance_matrix[: len(energy_hist), : len(energy_hist)],
                decimal=6 if strict_covar_checking else 1,
            )
            np.testing.assert_array_almost_equal(
                angle_hist.covariance_matrix,
                joined_hist.covariance_matrix[len(energy_hist) :, len(energy_hist) :],
                decimal=6 if strict_covar_checking else 1,
            )

        run_test_with_mc_gen_class(mc_hist_generator_cls=None)
        run_test_with_mc_gen_class(enable_cache=False)

        signal_parameters = ParameterSet(
            [
                Parameter("signal_strength", 1.0, bounds=(0, 10)),  # type: ignore
            ]
        )
        sob_kwargs = {
            "signal_query": "is_signal",
            "background_query": "not is_signal",
        }
        run_test_with_mc_gen_class(
            mc_hist_generator_cls=SignalOverBackgroundGenerator,
            parameters=signal_parameters,
            mc_hist_generator_kwargs=sob_kwargs,
            # The signal-over-background generator does not produce exactly the same results as the
            # default generator.
            strict_covar_checking=False,
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
