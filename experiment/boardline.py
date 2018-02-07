import numpy as np
import scipy as sp
import scipy as special
import scipy.stats
from scipy.special import comb

from blockline import theoretical as blk_theoretical


def theoretical(k, e, s):
    """Compute the PMF of the number of take-aways occurred in each board row/col

    Arguments:
        k {int} -- width of each block.
        e {int} -- expected number of take-aways in each block.
        s {float} -- standard deviation of the number of take-aways.

    Returns:
        {[k*(k-1)], float} -- PMF and expectation.
    """

    pmf, _ = blk_theoretical(k, e, s)
    conv_pmf = np.copy(pmf)
    for i in range(1, k):
        conv_pmf = np.convolve(conv_pmf, pmf)
    E = 0.0
    for i in range(k*k+1):
        E += i*conv_pmf[i]
    return conv_pmf, E


def empirical(k, e, s, N=10000):
    """Sample of the PMF of the number of take-aways occurred in each board row/col

    Arguments:
        k {int} -- width of each block.
        e {int} -- expected number of take-aways in each block.
        s {float} -- variable of the number of take-aways.

    Keyword Arguments:
        N {int} -- number of sample to be drawn (default: {10000})

    Returns:
        {[k*(k-1), float]} -- Sampled PMF and expectation.
    """

    board = np.zeros([k*k, k*k], dtype=np.int)
    pmf = np.zeros([k*k+1])
    E = 0.0

    for n in range(N):
        T = np.floor(sp.stats.truncnorm.rvs(
            a=0, b=k, loc=e, scale=s, size=k*k))
        for i in range(k):
            for j in range(k):
                block = board[i*k:(i+1)*k, j*k:(j+1)*k]
                a = np.reshape(block, [k*k])
                Tij = int(T[i*k + j])
                a[:Tij] = 1
                a[Tij:] = 0
                np.random.shuffle(a)
                board[i*k:(i+1)*k, j*k:(j+1)*k] = np.reshape(a, [k, k])
        row_stats = np.sum(board, axis=1)
        for t in row_stats:
            pmf[t] += 1

    pmf /= (k*k*N)
    for i in range(k*k+1):
        E += i*pmf[i]

    return pmf, E


if __name__ == "__main__":
    print("Theoretical result: ")
    exp_pmf, exp_E = theoretical(5, 3, 0.8)
    print("PMF[i]: ")
    print(exp_pmf)
    print("E[i] = " + str(exp_E))

    print("Empirical result: ")
    obs_pmf, obs_E = empirical(5, 3, 0.8)
    print("PMF[i]: ")
    print(obs_pmf)
    print("E[i] = " + str(obs_E))

    print("Chi-square statistics: ")
    chisq, p = sp.stats.chisquare(f_obs=obs_pmf*1000,
                                  f_exp=exp_pmf*1000)
    print("chisq = " + str(chisq))
    print("p = " + str(p))
