from scipy.stats import bernoulli as bern
import numpy as np


def pull(mus: np.ndarray, arm: int) -> int:
    """Simulate a arm pull, each arm follows an IID bernoulli distribution.

    Arguments:
        mus {np.ndarray} 
            -- expected reward of each arm.
        arm {int} 
            -- which arm to pull.

    Returns:
        int 
            -- obtained reward {0, 1}.
    """
    return bern.rvs(p=mus[arm], size=1)[0]


def ucb_reward(mus: np.ndarray, n: int, rounds: int=1000) -> float:
    """Contextless UCB.

    Arguments:
        mus {np.ndarray} 
            -- expected reward of each arm (Unknown by the algorithm).
        n {int} 
            -- maximum number of pulls.

    Returns:
        float 
            -- culmulative reward.
    """
    r = 0
    for _ in range(rounds):
        ws = np.zeros(mus.shape)
        ns = np.zeros(mus.shape) + 1
        ucbs = np.zeros(mus.shape)
        for t in range(mus.shape[0]):
            ws[t] = pull(mus, t)
        while t < n:
            ucbs = ws / ns + np.sqrt(2 * np.log(ns) / t)
            arm = np.argmax(ucbs)
            ws[arm] += pull(mus, arm)
            ns[arm] += 1
            t += 1
        r += np.sum(ws)
    return r/rounds


if __name__ == "__main__":
    mus = np.array([0.2, 0.8])
    print(ucb_reward(mus, 1000))
