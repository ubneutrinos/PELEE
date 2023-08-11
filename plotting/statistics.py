"""Module containing various statistical functions."""

import unittest
import numpy as np



def covariance(observations, central_value=None):
    """Calculate the covariance matrix of the given observations.

    Optionally, a central value can be given that will be used instead of the mean of the
    observations. This is useful for calculating the covariance matrix of the multisim
    uncertainties where the central value is the nominal MC prediction. Note that the
    calculation of the covariance matrix has to be normalized by (N - 1) if there
    is no central value given, which is done internally by numpy.cov.

    Parameters
    ----------
    observations : array_like
        Array of observations.
    central_value : array_like, optional
        Central value of the observations.

    Returns
    -------
    covariance_matrix : array_like
        Covariance matrix of the observations.
    """

    # Make sure that the observations and the central value are both ndarray
    observations = np.asarray(observations)
    if central_value is not None:
        central_value = np.asarray(central_value)
    # make sure the central value, if given, has the right length
    if central_value is not None:
        if central_value.shape[0] != observations.shape[1]:
            raise ValueError("Central value has wrong length.")
    # calculate covariance matrix
    if central_value is None:
        return np.cov(observations, rowvar=False)
    else:
        cov = np.zeros((observations.shape[1], observations.shape[1]))
        for i in range(observations.shape[1]):
            for j in range(observations.shape[1]):
                cov[i, j] = np.sum((observations[:, i] - central_value[i]) * (observations[:, j] - central_value[j]))
        # Here, we normalize by 1 / N, rather than 1 / (N - 1) as done by numpy.cov, because we
        # used the given central value rather than calculating it from the observations.
        return cov / observations.shape[0]


def constrained_covariance(
    sideband_measurement, obs_central_value, sideband_central_value, observations=None, sideband_observations=None, concat_cov=None
):
    """Calculate the covariance on the observations given the sideband measurement.

    This follows the Block-matrix prescription that can be found in
    https://en.wikipedia.org/wiki/Covariance_matrix#Block_matrices

    Parameters
    ----------
    sideband_measurement : array_like
        Measurement of the sideband.
    obs_central_value : array_like
        Central value of the observations.
    sideband_central_value : array_like
        Central value of the sideband observations.
    observations : array_like or None
        Array of observations. If None, the concat_cov must be given.
    sideband_observations : array_like or None
        Array of sideband observations. If None, the concat_cov must be given.
    concat_cov : array_like, optional
        Covariance matrix of the concatenated observations. If not given, it will be calculated
        from the observations and the central value.
    """

    if concat_cov is None:
        assert observations is not None
        assert sideband_observations is not None
        # First, we concatenate the observations and the sideband observations.
        # These must come from the same "universes" and therefore have the same length.
        assert observations.shape[0] == sideband_observations.shape[0]
        concat_observations = np.concatenate((observations, sideband_observations), axis=1)
        concat_central_value = np.concatenate((obs_central_value, sideband_central_value))
        # calculate the covariance matrix of the concatenated observations
        cov = covariance(concat_observations, concat_central_value)
    else:
        cov = concat_cov
    n = len(obs_central_value)
    # extract the blocks of the covariance matrix
    Kxx = cov[: n, : n]
    Kxy = cov[: n, n :]
    Kyx = cov[n :, : n]
    Kyy = cov[n :, n :]

    # now we can calculate the conditional mean
    mu = obs_central_value + Kxy @ np.linalg.inv(Kyy) @ (sideband_measurement - sideband_central_value)
    # and the conditional covariance
    cond_cov = Kxx - Kxy @ np.linalg.inv(Kyy) @ Kyx

    return cond_cov, mu

def error_propagation_division(x1, x2, C1, C2):
    """
    Compute the result of element-wise division of x1 by x2 and the associated covariance matrix.

    Parameters
    ----------
    x1 : array_like
        First array to be divided.
    x2 : array_like
        Second array to divide by.
    C1 : array_like
        Covariance matrix of x1.
    C2 : array_like
        Covariance matrix of x2.

    Returns
    -------
    y : array_like
        Result of element-wise division of x1 by x2.
    Cy : array_like
        Covariance matrix of y.
    """

    # Element-wise division to get y
    y = x1 / x2

    n = len(x1)
    Cy = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements (variance)
                Cy[i, i] = y[i] ** 2 * (C1[i, i] / x1[i] ** 2 + C2[i, i] / x2[i] ** 2)
            else:
                # Off-diagonal elements (covariance)
                Cy[i, j] = y[i] * y[j] * (C1[i, j] / (x1[i] * x1[j]) + C2[i, j] / (x2[i] * x2[j]))

    return y, Cy


class TestConstrainedCovariance(unittest.TestCase):
    def test_conditional_covariance(self):
        # Generate synthetic data
        np.random.seed(42)  # Setting a seed for reproducibility

        obs_central_value = np.array([4, 5, 7])  # shape N
        sideband_central_value = np.array([1, 2, 3, 4, 5])  # shape M

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
            np.concatenate((obs_central_value, sideband_central_value)), true_cov, 100000
        )
        # generate one observation of the sideband measurement
        sideband_measurement = np.random.multivariate_normal(sideband_central_value, true_cov_yy, 1)[0]
        # calculate the expected conditional mu and covariance
        expected_cond_cov = true_cov_xx - true_cov_xy @ np.linalg.inv(true_cov_yy) @ true_cov_yx
        expected_mu = obs_central_value + true_cov_xy @ np.linalg.inv(true_cov_yy) @ (
            sideband_measurement - sideband_central_value
        )

        observations = concat_observations[:, :N]
        sideband_observations = concat_observations[:, N:]
        # calculate the conditional covariance and mu
        cond_cov, mu = constrained_covariance(
            sideband_measurement,
            obs_central_value,
            sideband_central_value,
            observations=observations,
            sideband_observations=sideband_observations,
        )

        np.testing.assert_almost_equal(cond_cov, expected_cond_cov, decimal=2)
        np.testing.assert_almost_equal(mu, expected_mu, decimal=2)


if __name__ == "__main__":
    unittest.main()
