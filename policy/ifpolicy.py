class if_policy:
    """[summary]

    Returns:
        [type] -- [description]
    """

    def __init__(self):
        pass

    def init(self, board, t_stack, rvs, c_rvs, g):
        """Initialize the policy from current state.

        Arguments:
            board {ndarray<N,N>} 
                -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
            t_stack {list<transition>} 
                -- transition stack.
            rvs {ndarray<N,N,N>} 
                -- a boolean tensor representing the remaining values for each slot.
            c_rvs {dict<(i,j),int>} 
                -- a dictionary of remaining value counters
            g {dict<(i,j,m), set<(i,j)>>} 
                -- a dependency graph.
        """
        pass

    def sample(self, d, td, board, t_stack, rvs, c_rvs, g):
        """Generate a path based on the policy 
           and return whether such path leads to a win.

        Arguments:
            d {int}
                -- current depth.
            td {int}
                -- total depth.
            board {ndarray<N,N>} 
                -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
            t_stack {list<transition>} 
                -- transition stack.
            rvs {ndarray<N,N,N>} 
                -- a boolean tensor representing the remaining values for each slot.
            c_rvs {dict<(i,j),int>} 
                -- a dictionary of remaining value counters
            g {dict<(i,j,m), set<(i,j)>>} 
                -- a dependency graph.

        Returns:
            win {float} 
                -- 1 if won, or else 0.
            solution {list<action>}
                -- the path if a solution is found, or else None.
        """
        return 0, None
