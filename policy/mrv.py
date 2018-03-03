import ifpolicy
import gamerule as gr
import board as bd
import math as m
import scipy.stats as stats


def __node_expansion(g, rvs, c_rvs):
    """[summary]

    Arguments:
        g {dict<(i,j,m), set<(i,j)>>} 
            -- a dependency graph.
        rvs {ndarray<N,N,N>} 
            -- a boolean tensor representing the remaining values for each slot.
        c_rvs {dict<(i,j),int>}
            -- a dictionary of remaining value counters

    Returns:
        {list<i,j,m>} -- Available actions.
    """

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
        l = 0
        while True:
            actions = __node_expansion(g, rvs, c_rvs)
            if not actions:
                # end of the game.
                for _ in range(l):
                    bd.board_state_restore(t_stack, board, rvs, c_rvs, g)
                return gr.win_metric(d + l, td) if d + l < td else m.inf
            else:
                # create a exponential distribution over the actions.
                rv = stats.expon.rvs(loc=0, scale=len(actions))[0]
                selected_action = actions[int(rv*len(actions))]
                bd.board_state_transition(
                    t_stack, board, rvs, c_rvs, g, selected_action)
                l += 1
