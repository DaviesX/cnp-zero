
class theta:
    """MCTS parameters
    """

    def __init__(self, the, N):
        self.the = the
        self.N = N


class node:
    def __init__(self, score, p):
        self.score = score
        self.p = p


def policy_improvement_operator(board, rvs, c_rvs, g, f, the):
    """ MCTS with UCB: Q(s,a) + alpha*(Pr(s,a | f_the)/(1 + N(s, a))).

    Arguments:
        question {ndarray<N,N>} 
            -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
        rvs {ndarray<N,N,N>} 
            -- a boolean tensor representing the remaining values for each slot.
        c_rvs {dict<(i,j),int>}
            -- a dictionary of remaining value counters
        g {dict<(i,j,m), set<(i,j)>>} 
            -- a dependency graph.
        f {policy-value lambda} 
            -- policy-value network parameters.
        the {theta} 
            -- MCTS parameters.
    """
    n = node(0, 0)
    while True:
        continue
    pass
