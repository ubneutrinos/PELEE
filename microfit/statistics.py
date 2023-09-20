"""Module containing various statistical functions."""

import unittest
import numpy as np
import logging
import scipy.linalg as lin

logger = logging.getLogger(__name__)


def is_psd(A, ignore_zeros=False):
    """Test whether a matrix is positive semi-definite.

    Test is done via attempted Cholesky decomposition as suggested in [1]_.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    ignore_zeros : bool, optional
        Ignore rows and columns that are all zero. This is useful for covariance matrices
        where the zero rows and columns correspond to bins that are empty.

    Returns
    -------
    bool
        True if `A` is positive semi-definite, else False

    References
    ----------
    ..  [1] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    if ignore_zeros:
        # Delete all rows and columns from A
        # where all entries are zero.
        ixgrid = np.ix_(np.any(A != 0, axis=1), np.any(A != 0, axis=1))
        A = A[ixgrid]
    # pylint: disable=invalid-name
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def fronebius_nearest_psd(A, return_distance=False):
    """Find the positive semi-definite matrix closest to `A`.

    The closeness to `A` is measured by the Fronebius norm. The matrix closest to `A`
    by that measure is uniquely defined in [3]_.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    return_distance : bool, optional
        Return distance of the input matrix to the approximation as given in
        theorem 2.1 in [3]_.
        This can be compared to the actual Frobenius norm between the
        input and output to verify the calculation.

    Returns
    -------
    X : numpy.ndarray
        Positive semi-definite matrix approximating `A`.

    Notes
    -----
    This function is a modification of [1]_, which is a Python adaption of [2]_, which
    credits [3]_.

    References
    ----------
    ..  [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    ..  [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    ..  [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    # pylint: disable=invalid-name
    assert A.ndim == 2, "input is not a 2D matrix"
    B = (A + A.T) / 2.0
    _, H = lin.polar(B)
    X = (B + H) / 2.0
    # small numerical errors can make matrices that are not exactly
    # symmetric, fix that
    X = (X + X.T) / 2.0
    # due to numerics, it's possible that the matrix is _still_ not psd.
    # We can fix that iteratively by adding small increments of the identity matrix.
    # This part comes from [1].
    if not is_psd(X):
        spacing = np.spacing(lin.norm(X))
        I = np.eye(X.shape[0])
        k = 1
        while not is_psd(X):
            mineig = np.min(np.real(lin.eigvals(X)))
            X += I * (-mineig * k**2 + spacing)
            k += 1
    if return_distance:
        C = (A - A.T) / 2.0
        lam = lin.eigvalsh(B)
        # pylint doesn't know that numpy.sum takes the "where" argument
        # pylint: disable=unexpected-keyword-arg
        dist = np.sqrt(np.sum(lam**2, where=lam < 0.0) + lin.norm(C, ord="fro") ** 2)
        return X, dist
    return X


def check_frob_psd(A):
    """Check approximation of Frobenius-closest PSD on given matrix.

    This is not a unit test.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    """
    # pylint: disable=invalid-name
    X, xdist = fronebius_nearest_psd(A, return_distance=True)
    is_psd_after = is_psd(X)
    actual_dist = lin.norm(A - X, ord="fro")
    assert is_psd_after, "did not produce PSD matrix"
    assert np.isclose(xdist, actual_dist), "actual distance differs from expectation"


def covariance(observations, central_value=None, allow_approximation=False, debug_name=None, tolerance=0.0):
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
        cov = np.cov(observations, rowvar=False)
    else:
        centered_data = observations - central_value
        cov = centered_data.T @ centered_data / observations.shape[0]

    # check that all is finite
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains NaN or inf.")

    # For the following checks, we want to ignore all entries that are zero.
    # Delete all rows and columns from cov
    # where all entries are zero.
    ixgrid = np.ix_(np.any(cov != 0, axis=1), np.any(cov != 0, axis=1))
    cov_non_zero = cov[ixgrid]

    # check if the covariance matrix is positive definite
    if not is_psd(cov_non_zero):
        if allow_approximation:
            # if not, we try to find the nearest positive semi-definite matrix
            cov_non_zero, dist = fronebius_nearest_psd(cov_non_zero, return_distance=True)
            name_str = f"for {debug_name}" if debug_name is not None else ""
            logger.debug(f"Covariance matrix{name_str} is not positive semi-definite. ")
            logger.debug(
                "Using nearest positive semi-definite matrix instead. " "Distance to original matrix: %s",
                dist,
            )
            if dist > tolerance:
                raise ValueError(
                    f"Nearest positive semi-definite matrix{name_str} is not close enough to original matrix. "
                    f"Distance: {dist} > {tolerance}"
                )
        else:
            raise ValueError(f"Non-zero part of covariance matrix is not positive semi-definite. Matrix is: {cov}")

    # Now we need to add back the rows and columns that we deleted before.
    # We do this by adding back zero rows and columns.
    cov = np.zeros((observations.shape[1], observations.shape[1]))
    cov[ixgrid] = cov_non_zero
    return cov


def get_cnp_covariance(expectation, observation):
    """Get the combined Neyman-Pearson covaraince matrix.

    This matrix may be used to calculate a chi-square between the expectation and observation
    in a way that gives the optimal compromise between the Neyman and Pearson constructions.


    Parameters
    ----------
    expectation : array_like
        Array of expectation values.
    observation : array_like
        Array of observations.

    Returns
    -------
    cnp_covariance : array_like
        Combined Neyman-Pearson covariance matrix.

    Notes
    -----
    The combined Neyman-Pearson covariance matrix is given by

    .. math::
        C_{ij} = 3 / \\left( \\frac{2}{\\mu_i} + \\frac{1}{n_i} \\right) \\delta_{ij}

    where :math:`\\mu_i` and :math:`n_i` are the expectation and observation, respectively.
    For a mathematical derivation see arXiv:1903.07185.
    In this form, however, we can get division by zero errors when the observation is zero.
    We rearrange the equation instead to

    .. math::
        C_{ij} = 3 \\mu_i n_i \\delta_{ij} / \\left( \\mu_i + 2 n_i \\right)
    """

    expectation = np.asarray(expectation)
    observation = np.asarray(observation)
    if expectation.shape != observation.shape:
        raise ValueError("Expectation and observation must have the same shape.")
    if expectation.ndim != 1:
        raise ValueError("Expectation and observation must be 1D arrays.")
    if np.any(expectation <= 0):
        raise ValueError("Expectation must be positive.")
    if np.any(observation < 0):
        raise ValueError("Observation must be non-negative.")
    cnp_covariance = np.diag(3 * expectation * observation / (expectation + 2 * observation))
    return cnp_covariance


def chi_square(observation: np.ndarray, expectation: np.ndarray, systematic_covariance: np.ndarray) -> float:
    """
    Calculate the chi-square value for a given observation, expectation, and systematic covariance.

    Parameters:
    observation (np.ndarray): The observed data.
    expectation (np.ndarray): The expected data.
    systematic_covariance (np.ndarray): The systematic covariance matrix.

    Returns:
    float: The chi-square value.

    """
    n = observation
    mu = expectation

    # we need to mask off bins where the prediction is empty, otherwise
    # the covariance matrix will be singular
    mask = mu > 0
    syst_covar = systematic_covariance[np.ix_(mask, mask)]
    mu = mu[mask]
    n = n[mask]

    stat_covar = get_cnp_covariance(mu, n)
    total_covar = syst_covar + stat_covar
    covar_inv = np.linalg.inv(total_covar)
    chi2 = np.dot(n - mu, np.dot(covar_inv, n - mu))
    return chi2


def sideband_constraint_correction(
    sideband_measurement,
    sideband_central_value,
    obs_central_value=None,
    observations=None,
    sideband_observations=None,
    concat_covariance=None,
    sideband_covariance=None,
):
    """Calculate the corrections to the covariance and nominal values given the sideband measurement.

    This follows the Block-matrix prescription that can be found in
    https://en.wikipedia.org/wiki/Covariance_matrix#Block_matrices

    Parameters
    ----------
    sideband_measurement : array_like
        Measurement of the sideband.
    sideband_central_value : array_like
        Central value of the sideband observations.
    obs_central_value : array_like or None
        Central value of the observations. Not needed if concat_covariance is given.
    observations : array_like or None
        Array of observations. If None, the concat_covariance must be given.
    sideband_observations : array_like or None
        Array of sideband observations. If None, the concat_covariance must be given.
    concat_covariance : array_like, optional
        Covariance matrix of the concatenated observations. If not given, it will be calculated
        from the observations and the central value.
    sideband_covariance : array_like, optional
        Covariance matrix of the sideband. If given, this replaces the lower right corner of the
        concatenated covariance. This is expected to only contain the MC uncertainty of the
        sideband, as the Neyman-Pearson covariance that takes care of the data uncertainty is
        added in this function.

    Returns
    -------
    mu_offset : array_like
        Offset of the nominal values.
    covariance_correction : array_like
        Correction to the covariance matrix.
    """

    if concat_covariance is None:
        assert observations is not None
        assert sideband_observations is not None
        # First, we concatenate the observations and the sideband observations.
        # These must come from the same "universes" and therefore have the same length.
        assert observations.shape[0] == sideband_observations.shape[0]
        concat_observations = np.concatenate((observations, sideband_observations), axis=1)
        concat_central_value = np.concatenate((obs_central_value, sideband_central_value))
        # calculate the covariance matrix of the concatenated observations
        cov = covariance(concat_observations, concat_central_value)
        n = len(obs_central_value)
    else:
        cov = concat_covariance
        n = cov.shape[0] - len(sideband_central_value)
    # extract the blocks of the covariance matrix
    Kxx = cov[:n, :n]
    Kxy = cov[:n, n:]
    Kyx = cov[n:, :n]
    Kyy = cov[n:, n:]

    if sideband_covariance is not None:
        # check that the size is the same
        assert sideband_covariance.shape[0] == Kyy.shape[0]
        assert sideband_covariance.shape[1] == Kyy.shape[1]
        Kyy = sideband_covariance
    # We also add the CNP covariance to the sideband
    Kyy += get_cnp_covariance(sideband_central_value, sideband_measurement)

    # now we can calculate the conditional mean
    mu_offset = Kxy @ np.linalg.inv(Kyy) @ (sideband_measurement - sideband_central_value)
    # and the conditional covariance
    covariance_correction = -Kxy @ np.linalg.inv(Kyy) @ Kyx

    return mu_offset, covariance_correction


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


def error_propagation_multiplication(x1, x2, C1, C2):
    """
    Compute the result of element-wise multiplication of x1 by x2 and the associated covariance matrix.

    Parameters
    ----------
    x1 : array_like
        First array to be multiplied.
    x2 : array_like
        Second array to multiply by.
    C1 : array_like
        Covariance matrix of x1.
    C2 : array_like
        Covariance matrix of x2.

    Returns
    -------
    y : array_like
        Result of element-wise multiplication of x1 by x2.
    Cy : array_like
        Covariance matrix of y.
    """

    # Element-wise multiplication to get y
    y = x1 * x2

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


class TestMatrix(unittest.TestCase):
    def test_matrix_random(self):
        """Unit test producing a number of random matrices and checking if the
        approximated matrix is indeed PSD.
        """
        m_test = np.array([[1, -1], [2, 4]])
        check_frob_psd(m_test)
        for i in range(100):
            m_test = np.random.randn(3, 3)
            check_frob_psd(m_test)

class TestCovariance(unittest.TestCase):

    # here we test if the naive loop implementation of the covariance matrix
    # calculation is equivalent to the numpy implementation
    def loop_covariance(observations, central_value):
        cov = np.zeros((observations.shape[1], observations.shape[1]))
        for i in range(observations.shape[1]):
            for j in range(observations.shape[1]):
                cov[i, j] = np.sum((observations[:, i] - central_value[i]) * (observations[:, j] - central_value[j]))
        cov = cov / observations.shape[0]
        return cov
    # This is the optimized implementation that we actually use in the code
    def numpy_covariance(observations, central_value):
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
        cov_loop = TestCovariance.loop_covariance(observations, central_value)
        # calculate the covariance matrix with the numpy implementation
        cov_numpy = TestCovariance.numpy_covariance(observations, central_value)
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
        sideband_measurement = np.random.multivariate_normal(sideband_central_value, true_cov_yy, 1)[0]
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
