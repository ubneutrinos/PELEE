import unittest
import numpy as np
from ..statistics import (
    covariance,
    sideband_constraint_correction,
    get_cnp_covariance,
    check_frob_psd,
)


class TestMatrix(unittest.TestCase):
    def test_matrix_random(self):
        """Test if the approximated matrix is indeed PSD."""
        m_test = np.array([[1, -1], [2, 4]])
        check_frob_psd(m_test)
        for i in range(100):
            m_test = np.random.randn(3, 3)
            check_frob_psd(m_test)


class TestCovariance(unittest.TestCase):
    # here we test if the naive loop implementation of the covariance matrix
    # calculation is equivalent to the numpy implementation
    def loop_covariance(self, observations, central_value):
        cov = np.zeros((observations.shape[1], observations.shape[1]))
        for i in range(observations.shape[1]):
            for j in range(observations.shape[1]):
                cov[i, j] = np.sum(
                    (observations[:, i] - central_value[i])
                    * (observations[:, j] - central_value[j])
                )
        cov = cov / observations.shape[0]
        return cov

    # This is the optimized implementation that we actually use in the code
    def numpy_covariance(self, observations, central_value):
        centered_data = observations - central_value
        cov_matrix = centered_data.T @ centered_data / observations.shape[0]
        return cov_matrix

    def test_covariance(self):
        """Unit test for the covariance matrix calculation."""
        np.random.seed(42)
        # generate some random data
        central_value = np.random.rand(5)
        observations = np.random.rand(1000, 5)
        # calculate the covariance matrix
        cov = covariance(observations, central_value)
        # calculate the covariance matrix with the naive loop implementation
        cov_loop = self.loop_covariance(observations, central_value)
        # calculate the covariance matrix with the numpy implementation
        cov_numpy = self.numpy_covariance(observations, central_value)
        # check that the results are the same
        np.testing.assert_almost_equal(cov, cov_loop)
        np.testing.assert_almost_equal(cov, cov_numpy)


class TestConstrainedCovariance(unittest.TestCase):
    def test_conditional_covariance(self):
        # Generate synthetic data
        np.random.seed(42)  # Setting a seed for reproducibility

        obs_central_value = np.array([4, 5, 7])  # shape N
        sideband_central_value = np.array([1, 2, 3, 4, 5]) + 5  # shape M

        # generate a true covariance matrix for the concatenated observations
        # of shape (N + M, N + M)
        N = len(obs_central_value)
        M = len(sideband_central_value)
        true_cov = np.random.rand(N + M, N + M)
        # we need to make sure that this is symmetric and positive definite
        true_cov = true_cov + true_cov.T
        true_cov = true_cov + 2 * np.eye(N + M)
        # extract the block matrices
        true_cov_xx = true_cov[:N, :N]
        true_cov_xy = true_cov[:N, N:]
        true_cov_yx = true_cov[N:, :N]
        true_cov_yy = true_cov[N:, N:]

        concat_observations = np.random.multivariate_normal(
            np.concatenate((obs_central_value, sideband_central_value)), true_cov, 1000000
        )
        # generate one observation of the sideband measurement
        sideband_measurement = np.random.multivariate_normal(
            sideband_central_value, true_cov_yy, 1
        )[0]
        # add the CNP covariance to the sideband
        true_cov_yy += get_cnp_covariance(sideband_central_value, sideband_measurement)
        # calculate the expected conditional mu and covariance
        expected_cond_cov = true_cov_xx - true_cov_xy @ np.linalg.inv(true_cov_yy) @ true_cov_yx
        expected_mu = obs_central_value + true_cov_xy @ np.linalg.inv(true_cov_yy) @ (
            sideband_measurement - sideband_central_value
        )

        observations = concat_observations[:, :N]
        sideband_observations = concat_observations[:, N:]
        # calculate the conditional covariance and mu
        mu_offset, cov_corr = sideband_constraint_correction(
            sideband_measurement,
            sideband_central_value,
            obs_central_value,
            observations=observations,
            sideband_observations=sideband_observations,
        )
        cond_cov = true_cov_xx + cov_corr
        mu = obs_central_value + mu_offset

        # np.testing.assert_almost_equal(cond_cov, expected_cond_cov, decimal=2)
        np.testing.assert_almost_equal(mu, expected_mu, decimal=2)


if __name__ == "__main__":
    unittest.main()
