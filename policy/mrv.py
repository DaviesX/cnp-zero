import policy.ifpolicy as ifpolicy
import gamerule as gr
import board as bd
import math as m
import scipy.stats as stats


class mrv(ifpolicy.if_policy):
    """[summary]

    Arguments:
        ifpolicy {[type]} -- [description]

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
            float -- 1 if won, or else 0.
        """
        sln_path = list()
        while True:
            actions = bd.board_state_expansion(g, rvs, c_rvs)
            if not actions:
                # end of the game.
                for _ in range(len(sln_path)):
                    bd.board_state_restore(t_stack, board, rvs, c_rvs, g)
                return (gr.win_metric(d + len(sln_path), td), None) if d + len(sln_path) < td else (m.inf, sln_path)
            else:
                # create a exponential distribution over the actions.
                rv = stats.expon.rvs(loc=0, scale=len(actions)/10)
                selected_action = actions[int(rv) if rv < len(
                    actions) else len(actions) - 1]
                bd.board_state_transition(
                    t_stack, board, rvs, c_rvs, g, selected_action)
                sln_path.append(selected_action)
