import numpy as np
import board as bd

X = 0


def __csp_mrv_deg_expansion(g, rvs, c_rvs):
    actions = list()
    for slot, s_rvs in c_rvs.items():
        if s_rvs is not None:
            for s_rv in s_rvs:
                actions.append((slot[0], slot[1], s_rv))
    actions.sort(key=lambda action: len(g[action]) if g.get(
        action) is not None else 0, reverse=True)
    actions.sort(key=lambda action: len(
        c_rvs[(action[0], action[1])]), reverse=False)
    return actions


def __csp_mrv_deg_solve(t_stack, question, rvs, c_rvs, g, k, n):
    if k == n:
        assert(bd.validate(question))
        return np.copy(question), 1
    else:
        b = 1
        for action in __csp_mrv_deg_expansion(g, rvs, c_rvs):
            bd.board_state_transition(t_stack, question, rvs, c_rvs, g, action)
            answer, nc = __csp_mrv_deg_solve(
                t_stack, question, rvs, c_rvs, g, k+1, n)
            bd.board_state_restore(t_stack, question, rvs, c_rvs, g)

            b += nc
            if answer is not None:
                return answer, b
    global X
    if b > X:
        print(question)
        print(b)
        X = b
    return None, b


def csp_mrv_deg(question, rvs, c_rvs, g):
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
    Returns:
        {ndarray<N,N>} 
            -- a solved cnp, if possible.
    """
    return __csp_mrv_deg_solve(list(), question, rvs, c_rvs, g, 0, np.where(question == 0)[0].shape[0])


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
    t_stack = list()

    answer, nc = csp_mrv_deg(question, rvs, c_rvs, g)
    print(answer)
    print(np.all(board == answer))
    print(nc)
