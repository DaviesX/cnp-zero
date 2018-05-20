import os
import numpy as np
import tensorflow as tf
from policy.ifpolicy import if_policy
from dataset.ifdataset import if_dataset


def conv_block(board_node: tf.Operation,
               i1: int, i1d: int, o1d: int
               epsilon: float = 1e-3) -> tf.Operation:
    with tf.name_scope("conv_block"):
        # conv -> batch norm -> rec
        w = tf.Variable(tf.truncated_normal([i1, i1, i1d, o1d], stddev=0.1),
                        name="w")
        h = tf.nn.conv2d(board_node, w,
                         strides=[1, 1, 1, 1],
                         padding="SAME",
                         name="conv")
        mu, sigma2 = tf.nn.moments(h, [0])
        alpha = tf.Variable(tf.ones([o1d]), name="alpha")
        beta = tf.Variable(tf.zeros([o1d]), name="beta")
        hn = alpha*(h - mu)/tf.sqrt(sigma2 + epsilon) + beta
        return tf.nn.relu(hn)


def residual_block(block_node: tf.Operation,
                   li: int, i1d: int, o1d: int,
                   epsilon: float = 1e-3) -> tf.Operation:
    with tf.name_scope("residual_block"): 
        # conv block -> conv block -> skip conn -> rec
        c1 = conv_block(block_node, li, lid, o1d, epsilon)
        c2 = conv_block(c1, li, lid, o1d, epsilon)
        return tf.nn.relu(c2 + block_node)


def cell_policy(res_node: tf.Operation,
                n_cells: int):
    with tf.name_scope("cell_policy"): 
        pass
    pass


def cell_cond_action_policy(res_node: tf.Operation,
                            cell_pmf_node: tf.Operation,
                            nactions: int):
    with tf.name_scope("cell_cond_action_policy"): 
        pass
    pass


def cell_action_value(res_node: tf.Operation):
    with tf.name_scope("cell_cond_action_policy"): 
        pass
    pass


def bootstrap(path_graphdef: str) -> None:
    with tf.name_scope("cell_cond_action_policy"): 
        tf.placeholder(
    pass


class zeronet(if_policy):
    """Zero net policy.
    """
    def __init__(self, 
                 path_graphdef: str, 
                 cnp: int=3):
        # if the network structure is specified then use it as starting point
        # otherwise, create the network structure with RVs ~ Norm
        self.graph_def = tf.import_graph_def(path_graphdef) \
                if os.path.is_file(path_graphdef) \
                else bootstrap(path_graphdef, cnp)

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
