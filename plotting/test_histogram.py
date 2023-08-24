"""Unit Tests for plotting.histogram module"""
import unittest
from .histogram import Histogram, HistogramGenerator, Binning, RunHistGenerator
import numpy as np
import pandas as pd
import uncertainties.unumpy as unumpy

class TestHistogram(unittest.TestCase):
    def test_uncorrelated(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x")
        bin_counts1 = np.array([1, 2, 3])
        uncertainties1 = np.array([0.1, 0.2, 0.3])
        bin_counts2 = np.array([4, 5, 6])
        uncertainties2 = np.array([0.4, 0.5, 0.6])

        hist1 = Histogram(binning, bin_counts1, uncertainties1, label="hist1", tex_string="hist1")
        hist2 = Histogram(binning, bin_counts2, uncertainties2, label="hist2", tex_string="hist2")

        hist_sum = hist1 + hist2
        hist_diff = hist1 - hist2

        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_sum.bin_counts), np.array([5, 7, 9]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_sum.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )
        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_diff.bin_counts), np.array([-3, -3, -3]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_diff.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )

    def test_correlated(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x")
        bin_counts1 = np.array([1, 2, 3])
        covariance_matrix1 = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        bin_counts2 = np.array([4, 5, 6])
        covariance_matrix2 = np.array([[0.16, 0.20, 0.24], [0.20, 0.25, 0.30], [0.24, 0.30, 0.36]])

        hist1 = Histogram(
            binning, bin_counts1, covariance_matrix=covariance_matrix1, label="hist1", tex_string="hist1"
        )
        hist2 = Histogram(
            binning, bin_counts2, covariance_matrix=covariance_matrix2, label="hist2", tex_string="hist2"
        )

        hist_sum = hist1 + hist2
        hist_diff = hist1 - hist2

        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_sum.bin_counts), np.array([5, 7, 9]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_sum.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )
        np.testing.assert_array_almost_equal(unumpy.nominal_values(hist_diff.bin_counts), np.array([-3, -3, -3]))
        np.testing.assert_array_almost_equal(
            unumpy.std_devs(hist_diff.bin_counts), np.sqrt(np.array([0.17, 0.29, 0.45]))
        )

    def test_fluctuation(self):
        # Generate a histogram with a covariance matrix
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x")
        bin_counts = np.array([1, 2, 3])
        covariance_matrix = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        hist = Histogram(binning, bin_counts, covariance_matrix=covariance_matrix, label="hist", tex_string="hist")

        # fluctuate the histogram and check that the fluctuated bin counts are distributed according to the covariance matrix
        fluctuated_counts = []
        for i in range(10000):
            fluctuated_hist = hist.fluctuate(seed=i)
            fluctuated_counts.append(fluctuated_hist.nominal_values)
        fluctuated_counts = np.array(fluctuated_counts)

        # calculate covariance matrix of fluctuated counts with numpy
        cov_matrix = np.cov(fluctuated_counts, rowvar=False)

        # this should be close to the expectation value
        np.testing.assert_array_almost_equal(cov_matrix, covariance_matrix, decimal=2)

    def test_division(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x")
        # We want to test the division away from zero to avoid division by zero errors
        bin_counts1 = np.array([1, 2, 3]) + 10
        covariance_matrix1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]])
        # covariance_matrix1 = np.array([[0.1, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.9]])
        bin_counts2 = np.array([4, 5, 6]) + 10
        covariance_matrix2 = np.array([[0.16, 0.20, 0.24], [0.20, 0.25, 0.30], [0.24, 0.30, 0.36]])
        # covariance_matrix2 = np.array([[0.16, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.36]])

        hist1 = Histogram(
            binning, bin_counts1, covariance_matrix=covariance_matrix1, label="hist1", tex_string="hist1"
        )
        hist2 = Histogram(
            binning, bin_counts2, covariance_matrix=covariance_matrix2, label="hist2", tex_string="hist2"
        )

        fluctuated_divisions = []
        # To test error propagation, we fluctuate hist1 and hist2 and divide them. The covariance matrix
        # of the fluctuated divisions should be close to the expected covariance matrix that we get
        # from the division function.
        for i in range(10000):
            fluctuated_hist1 = hist1.fluctuate(seed=i)
            # It's important not to repeat seeds here, otherwise the values will be correlated
            # when they should not be.
            fluctuated_hist2 = hist2.fluctuate(seed=i + 10000)
            fluctuated_division = fluctuated_hist1 / fluctuated_hist2
            fluctuated_divisions.append(fluctuated_division.nominal_values)
        fluctuated_divisions = np.array(fluctuated_divisions)

        # calculate covariance matrix of fluctuated divisions with numpy
        cov_matrix = np.cov(fluctuated_divisions, rowvar=False)

        # get expectation histogram
        expected_div_hist = hist1 / hist2
        # check nominal values
        np.testing.assert_array_almost_equal(
            fluctuated_divisions.mean(axis=0), expected_div_hist.nominal_values, decimal=3
        )
        # check covariance matrix
        np.testing.assert_array_almost_equal(cov_matrix, expected_div_hist.cov_matrix, decimal=4)

    def test_scalar_multiplication(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x")
        bin_counts = np.array([1, 2, 3])
        covariance_matrix = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        hist = Histogram(binning, bin_counts, covariance_matrix=covariance_matrix, label="hist", tex_string="hist")

        # Test multiplication of a Histogram by a scalar from the left
        hist_scaled_left = 2 * hist
        np.testing.assert_array_almost_equal(hist_scaled_left.nominal_values, np.array([2, 4, 6]))
        np.testing.assert_array_almost_equal(hist_scaled_left.std_devs, np.sqrt(np.array([0.04, 0.16, 0.36])))

        # Test multiplication of a Histogram by a scalar from the right
        hist_scaled_right = hist * 2
        np.testing.assert_array_almost_equal(hist_scaled_right.nominal_values, np.array([2, 4, 6]))
        np.testing.assert_array_almost_equal(hist_scaled_right.std_devs, np.sqrt(np.array([0.04, 0.16, 0.36])))

    # Test conversion to and from dict
    def test_dict_conversion(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x-axis label")
        bin_counts = np.array([1, 2, 3])
        covariance_matrix = np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.06], [0.03, 0.06, 0.09]])
        hist = Histogram(binning, bin_counts, covariance_matrix=covariance_matrix, label="hist", tex_string="hist")
        hist_dict = hist.to_dict()
        hist_from_dict = Histogram.from_dict(hist_dict)
        self.assertEqual(hist, hist_from_dict)


class TestHistogramGenerator(unittest.TestCase):
    def test_generate(self):
        # Create an example DataFrame
        df = pd.DataFrame({"x": np.random.rand(100), "weights": np.random.rand(100)})

        # Create a binning
        binning = Binning("x", np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), "X")

        # Initialize the HistogramGenerator
        generator = HistogramGenerator(df, binning, weight_column="weights")

        # Generate the histogram
        histogram = generator.generate()

        # Verify that the histogram is created correctly
        self.assertEqual(len(histogram.bin_counts), 5)
        self.assertEqual(len(histogram.std_devs), 5)

    def test_generate_with_query(self):
        # Create an example DataFrame with an additional column 'y'
        df = pd.DataFrame(
            {
                "x": np.random.rand(100),
                "y": np.random.randint(0, 2, 100),  # Additional column for querying
                "weights": np.random.rand(100),
            }
        )

        # Create a binning
        binning = Binning("x", np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), "X")

        # Initialize the HistogramGenerator
        generator = HistogramGenerator(df, binning, weight_column="weights")

        # Define a query to select rows where 'y' is equal to 1
        query = "y == 1"

        # Generate the histogram using the query
        histogram = generator.generate(query=query)

        # Verify that the histogram is created correctly
        self.assertEqual(len(histogram.bin_counts), 5)
        self.assertEqual(len(histogram.std_devs), 5)

class TestRunHistGenerator(unittest.TestCase):

    def test_get_data_hist(self):
        # Create some mock data
        rundata_dict = {
            "data": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]}),
            "mc": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]}), 
            "ext": pd.DataFrame({"x": [1, 2, 3], "weights": [1, 1, 1]})
        }
        binning = Binning("x", [0, 1, 2, 3, 4], "x")
        generator = RunHistGenerator(rundata_dict, binning, data_pot=1.0)
        
        # Test getting the data histogram
        data_hist = generator.get_data_hist()
        np.testing.assert_array_equal(data_hist.nominal_values, [0, 1, 1, 1])
        
        # Test getting the EXT histogram
        ext_hist = generator.get_data_hist(type="ext")
        np.testing.assert_array_equal(ext_hist.nominal_values, [0, 1, 1, 1])

        # Test scaling the EXT histogram
        ext_hist_scaled = generator.get_data_hist(type="ext", scale_to_pot=2)
        np.testing.assert_array_equal(ext_hist_scaled.nominal_values, [0, 2, 2, 2])

if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)