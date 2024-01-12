from typing import Union
import unittest
import pandas as pd
import numpy as np
from microfit.histogram import Binning
from microfit.histogram import RunHistGenerator
from microfit.histogram import MultiChannelBinning

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

    def make_dataframe(self, n_samples=1000, data_like=False, with_multisim=False):
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
        if not with_multisim:
            return df
        # The 'weights_no_tune' column is used to calculate multisim uncertainties for GENIE
        # variables. For testing purposes, we just set it to the same as 'weights'.
        df["weights_no_tune"] = df["weights"]
        n_universes = 100
        for ms_column in ["weightsGenie", "weightsFlux", "weightsReint"]:
            df[ms_column] = [
                n_samples * np.random.normal(loc=1, size=n_universes, scale=0.1)
                for _ in range(len(df))
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
            df[f"{knob}up"] = 1.1
            df[f"{knob}dn"] = 0.9
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
        np.testing.assert_array_equal(data_hist.nominal_values, [0, 1, 1, 1])

        # Test getting the EXT histogram
        ext_hist = generator.get_data_hist(type="ext")
        assert ext_hist is not None
        np.testing.assert_array_equal(ext_hist.nominal_values, [0, 1, 1, 1])

        # Test scaling the EXT histogram
        ext_hist_scaled = generator.get_data_hist(type="ext", scale_to_pot=2)
        assert ext_hist_scaled is not None
        np.testing.assert_array_equal(ext_hist_scaled.nominal_values, [0, 2, 2, 2])

if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
