import numpy as np
import board as bd
import gamerule as gr


class trial_limit:
    """[summary]
    """

    def __init__(self, t, max_trials):
        self.t = t
        self.max_trials = max_trials
        self.best_ans = None
        self.best_k = 0


def __csp_mrv_deg_solve(t_stack, question, rvs, c_rvs, g, k, n, limit):
    """[summary]

    Arguments:
        t_stack {[type]} -- [description]
        question {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
        k {[type]} -- [description]
        n {[type]} -- [description]
        limit {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if k > limit.best_k:
        # record best answer so far.
        limit.best_ans = np.copy(question)
        limit.best_k = k

    if k == n:
        assert(bd.validate(question))
        limit.best_ans = np.copy(question)
        limit.best_k = k
        return True
    else:
        for action in bd.board_state_expansion(g, rvs, c_rvs):
            bd.board_state_transition(t_stack, question, rvs, c_rvs, g, action)
            terminate = __csp_mrv_deg_solve(
                t_stack, question, rvs, c_rvs, g, k+1, n, limit)
            bd.board_state_restore(t_stack, question, rvs, c_rvs, g)

            if terminate:
                return True

            if limit.t >= limit.max_trials:
                # out of trial limit
                return True
            limit.t += 1

    return False


def csp_mrv_deg(question, rvs, c_rvs, g, max_trials=1000):
    """Solve the cnp using csp with minimum remaining values + degree heuristics.

    Arguments:
        question {ndarray<N,N>} 
            -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
        rvs {ndarray<N,N,N>} 
            -- a boolean tensor representing the remaining values for each slot.
        c_rvs {dict<(i,j),int>}
            -- a dictionary of remaining value counters
        g {dict<(i,j,m), set<(i,j)>>} 
            -- a dependency graph.

    Keyword Arguments:
        max_trials {int} -- maximum number of trials before giving up (default: {1000})

    Returns:
        {ndarray<N,N>, boolean} 
            -- a solved cnp, if possible;
            -- indicate whether the answer is complete.
    """
    limit = trial_limit(1, max_trials)
    __csp_mrv_deg_solve(list(), question, rvs, c_rvs, g,
                        0,
                        np.where(question == 0)[0].shape[0],
                        limit)
    return limit.best_ans, limit.best_k, limit.t


if __name__ == "__main__":
    # test
    np.random.seed(13)
    k = 3
    e = 4
    s = 1
    board, stones, question = bd.create_random(k, e, s)
    print(board)
    print(stones)
    print(question)
    print(bd.validate(board))
    print(bd.validate(question))
    rvs, c_rvs, g = bd.remaining_values_deg(question)
    print(rvs[0: 2, 0: 2, :])
    print("total dof: " + str(np.sum(rvs)))
    print("average dof: " + str(np.sum(rvs)/np.sum(1 - stones)))

    old_question = np.copy(question)
    old_rvs = np.copy(rvs)
    old_g = g.copy()
    old_crvs = c_rvs.copy()

    answer, k, t = csp_mrv_deg(question, rvs, c_rvs, g)

    print("sanity check: ")
    print(np.all(old_question == question))
    print(np.all(old_rvs == rvs))
    print(old_g == g)
    print(old_crvs == c_rvs)

    print("answer")
    print(answer)
    print(bd.validate(answer))
    n = np.sum(question == 0)
    print("completion=" + str(k) + "/" +
          str(n) + "->" + str(gr.win_metric(k, n)))
    print("number of trials=" + str(t))
    print(np.all(board == answer))
