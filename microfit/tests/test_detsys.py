import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from microfit.histogram import (
    Binning,
    MultiChannelBinning,
    HistogramGenerator,
    MultiChannelHistogram,
)
from microfit.detsys import make_variation_histograms, make_variations


class MockLoadRunsDetvar:
    def __init__(self, mock_df):
        self.mock_df = mock_df

    def __call__(self, run, dataset, variation, **kwargs):
        if variation == "cv":
            return {"mc": self.mock_df}, {}, 1.0
        elif variation == "up":
            mock_df_up = self.mock_df.copy()
            mock_df_up["weights"] *= 1.1
            return {"mc": mock_df_up}, {}, 1.0
        elif variation == "down":
            mock_df_down = self.mock_df.copy()
            mock_df_down["weights"] *= 0.9
            return {"mc": mock_df_down}, {}, 1.0
        else:
            raise ValueError(f"Unknown variation: {variation}")


class TestDetSys(unittest.TestCase):
    @patch("microfit.detsys.dl.load_runs_detvar")
    def test_make_variation_histograms(self, mock_load_runs_detvar):
        # Create a mock dataframe
        mock_df = pd.DataFrame(
            {
                "energy": np.random.lognormal(0, 0.5, 1000),
                "angle": np.random.uniform(0, 3.14, 1000),
                "bdt": np.random.uniform(0, 1, 1000),
                "weights": np.random.uniform(0, 1, 1000),
            }
        )
        mock_df["weights"] *= 0.1 / mock_df["weights"].mean()  # scale weights to have a mean of 0.1

        # Create a mock object that behaves like load_runs_detvar
        mock_load_runs_detvar.side_effect = MockLoadRunsDetvar(mock_df)

        # Create a binning object
        binning = Binning("energy", np.linspace(0, 3, 12), "energy")

        # Test the make_variation_histograms function
        hist_dict = make_variation_histograms(["run1"], "dataset", "cv", binning)
        cv_hist = hist_dict["mc"]
        cv_bin_counts = cv_hist.bin_counts

        hist_dict_up = make_variation_histograms(["run1"], "dataset", "up", binning)
        up_hist = hist_dict_up["mc"]
        up_bin_counts = up_hist.bin_counts

        hist_dict_down = make_variation_histograms(["run1"], "dataset", "down", binning)
        down_hist = hist_dict_down["mc"]
        down_bin_counts = down_hist.bin_counts

        # Check that the bin counts are scaled as expected
        np.testing.assert_allclose(up_bin_counts, cv_bin_counts * 1.1, rtol=1e-5)
        np.testing.assert_allclose(down_bin_counts, cv_bin_counts * 0.9, rtol=1e-5)

    def make_test_binning(self, multichannel=False, with_query=False, second_query="matching"):
        first_channel_binning = Binning("energy", np.linspace(0, 3, 12), "energy")
        if with_query:
            first_channel_binning.selection_query = "bdt > 0.5"
        if not multichannel:
            return first_channel_binning

        second_channel_binning = Binning.from_config("angle", 10, (0, 3.14), "angle")
        second_query_string = {
            "matching": "bdt > 0.5",
            "non_matching": "bdt < 0.5",
            "overlapping": "bdt < 0.8",
        }[second_query]
        if with_query:
            second_channel_binning.selection_query = second_query_string

        binning = MultiChannelBinning([first_channel_binning, second_channel_binning])
        return binning

    @patch("microfit.detsys.dl.load_runs_detvar")
    def test_make_variations(self, mock_load_runs_detvar):
        # Create a mock dataframe
        mock_df = pd.DataFrame(
            {
                "energy": np.random.lognormal(0, 0.5, 1000),
                "angle": np.random.uniform(0, 3.14, 1000),
                "bdt": np.random.uniform(0, 1, 1000),
                "weights": np.random.uniform(0, 1, 1000),
            }
        )
        mock_df["weights"] *= 0.1 / mock_df["weights"].mean()  # scale weights to have a mean of 0.1

        # Create a mock object that behaves like load_runs_detvar
        mock_load_runs_detvar.side_effect = MockLoadRunsDetvar(mock_df)

        # Create a temporary directory for the detvar_cache
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test with a Binning
            binning = self.make_test_binning(with_query=True)
            detvar_data_binning = make_variations(
                ["1", "2", "3"],
                "dataset",
                binning,
                detvar_cache_dir=tmp_dir,
                make_plots=False,
                variations=["cv", "up", "down"],
            )
            self.assertIsInstance(detvar_data_binning, dict)
            self.assertIn("variation_hist_data", detvar_data_binning)

            # Test with a MultiChannelBinning
            multi_binning = self.make_test_binning(
                multichannel=True, with_query=True, second_query="non_matching"
            )
            detvar_data_multi_binning = make_variations(
                ["1", "2", "3"],
                "dataset",
                multi_binning,
                detvar_cache_dir=tmp_dir,
                make_plots=False,
                variations=["cv", "up", "down"],
            )
            self.assertIsInstance(detvar_data_multi_binning, dict)
            self.assertIn("variation_hist_data", detvar_data_multi_binning)

    @patch("microfit.detsys.dl.load_runs_detvar")
    def test_with_hist_generator(self, mock_load_runs_detvar):
        """Test detvar_data when passed to a HistogramGenerator"""

        # Make mock data, mock detvar_data and instantiate a HistogramGenerator
        mock_df = pd.DataFrame(
            {
                "energy": np.random.lognormal(0, 0.5, 1000),
                "angle": np.random.uniform(0, 3.14, 1000),
                "bdt": np.random.uniform(0, 1, 1000),
                "weights": np.random.uniform(0, 1, 1000),
            }
        )

        mock_df["weights"] *= 0.1 / mock_df["weights"].mean()  # scale weights to have a mean of 0.1
        mock_load_runs_detvar.side_effect = MockLoadRunsDetvar(mock_df)

        binning = self.make_test_binning(
            multichannel=True, with_query=True, second_query="non_matching"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            detvar_data = make_variations(
                ["1", "2", "3"],
                "dataset",
                binning,
                detvar_cache_dir=tmp_dir,
                make_plots=False,
                variations=["cv", "up", "down"],
            )

        hist_gen = HistogramGenerator(mock_df, binning, detvar_data=detvar_data)

        hist_no_detvar = hist_gen.generate()
        hist_with_detvar = hist_gen.generate(
            add_precomputed_detsys=True, include_detsys_variations=["cv", "up", "down"]
        )

        # Assert valid histograms were produced
        self.assertIsInstance(hist_no_detvar, MultiChannelHistogram)
        self.assertIsInstance(hist_with_detvar, MultiChannelHistogram)

        # Assert that the covariance matrix is diagonal for the histogram without detector variations,
        # and non-diagonal for the histogram with detector variations
        np.testing.assert_array_equal(
            hist_no_detvar.covariance_matrix, np.diag(np.diag(hist_no_detvar.covariance_matrix))
        )
        # There should be non-zero off-diagonal elements in the covariance matrix with detvars
        self.assertTrue(
            np.any(
                hist_with_detvar.covariance_matrix
                - np.diag(np.diag(hist_with_detvar.covariance_matrix))
            )
        )


if __name__ == "__main__":
    unittest.main()
