import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.special import comb


def theoretical(k, e, s):
    """Compute the PMF of the number of take-aways occurred in each block row/col

    Arguments:
        k {int} -- width of the block.
        e {int} -- expected number of take-aways in a block.
        s {float} -- standard deviation of the number of take-aways.

    Returns:
        {[k], float} -- PMF and expectation.
    """

    pmf = np.zeros([k+1])
    E = 0.0
    for i in range(0, k+1):
        # pmf(i|k,e,s) = Sum_{T} (Pr(i takeaways in a row | T takeways) * Pr(T takeaways))
        # pmf(i|k,e,s) = Sum_{i<=t<=k} (binom_pmf(i,t,1/k)*(truncnorm_cdf(t+1,e,s) - truncnorm_cdf(t,e,s)))
        marginal = 0.0
        for t in range(i, k+1):
            marginal += stats.binom.pmf(k=i, n=t, p=1.0/k) * \
                (sp.stats.truncnorm.cdf(t+1, a=0, b=k, loc=e, scale=s) -
                 sp.stats.truncnorm.cdf(t, a=0, b=k, loc=e, scale=s))
        pmf[i] = marginal
        E += i*pmf[i]
    return pmf, E


def empirical(k, e, s, N=10000):
    """Sample of the PMF of the number of take-aways occurred in each block row/col

    Arguments:
        k {int} -- width of the block.
        e {int} -- expected number of take-aways in a block.
        s {float} -- standard deviation of the number of take-aways.

    Keyword Arguments:
        N {int} -- number of sample to be drawn (default: {10000})

    Returns:
        {[k], float} -- Sampled PMF and expectation.
    """

    block = np.zeros([k, k], dtype=np.int)
    pmf = np.zeros([k+1])
    E = 0.0

    r = np.floor(sp.stats.truncnorm.rvs(a=0, b=k, loc=e, scale=s, size=N))
    for i in range(N):
        T = int(r[i])
        a = np.reshape(block, [k*k])
        a[0:T] = 1
        a[T:] = 0
        np.random.shuffle(a)
        stats = np.sum(block, axis=1)
        for t in stats:
            pmf[t] += 1

    pmf /= (N*k)
    for i in range(k+1):
        E += (pmf[i]*i)
    return pmf, E


if __name__ == "__main__":
    print("Theoretical result: ")
    exp_pmf, exp_E = theoretical(5, 3, 0.8)
    print("PMF[i]:")
    print(exp_pmf)
    print("E[i] = " + str(exp_E))

    print("Empirical result: ")
    obs_pmf, obs_E = empirical(5, 3, 0.8)
    print("PMF[i]:")
    print(obs_pmf)
    print("E[i] = " + str(obs_E))

    print("Chi-square statistics: ")
    chisq, p = sp.stats.chisquare(f_obs=obs_pmf*100,
                                  f_exp=exp_pmf*100)
    print("chisq = " + str(chisq))
    print("p = " + str(p))
