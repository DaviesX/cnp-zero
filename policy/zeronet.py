import numpy as np
import tensorflow as tf
from policy.ifpolicy import if_policy


def create(path_graphdef):
    pass


class zeronet(if_policy):
    """[summary]

    Arguments:
        ifpolicy {[type]} -- [description]
    """

    def __init__(self, path_graphdef: str):
        self.graph_def = tf.import_graph_def(path_graphdef)

    def init(self,
             board: np.ndarray, t_stack: list,
             rvs: np.ndarray, c_rvs: dict, g: dict) -> None:
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

    def sample(self, d: int, td: int,
               board: np.ndarray, t_stack: list,
               rvs: np.ndarray, c_rvs: dict, g: dict) -> tuple:
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
