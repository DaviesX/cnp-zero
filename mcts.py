import math as m
import board as bd


class theta:
    """MCTS parameters
    """

    def __init__(self, c=1.41, N=1):
        """[summary]

        Keyword Arguments:
            c {[type]} -- [description] (default: {1})
            N {[type]} -- [description] (default: {1})
        """
        self.c = c
        self.N = N


class node:
    """MCT node
    """

    def __init__(self, action):
        self.action = action
        self.score = 0
        self.N = 0
        self.n = 0
        self.w = 0
        self.children = list()
        self.child_lookup = dict()

    def __lt__(self, rhs):
        return self.score < rhs.score

    def add_child(self, c):
        """Add a child node.

        Arguments:
            c {node}
                -- child node
        """
        self.children.append(c)
        self.child_lookup[c.action] = c

    def max_child(self):
        """Return a child that has max score.

        Returns:
            node -- child node that has max score.
        """
        return max(self.children)

    def update_child(self, c):
        """update the child considering its having a new score.

        Arguments:
            c {node} -- the child to be updated.
        """
        pass


def __mcts_nav2mpm(node, board, t_stack, rvs, c_rvs, g):
    """Navigate to most promising move.

    Arguments:
        node {[type]} -- [description]
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
    """
    if not node.children:
        return node, [node]
    else:
        greatest_child = node.max_child()
        bd.board_state_transition(
            t_stack, board, rvs, c_rvs, g, greatest_child.action)
        candid, path = __mcts_nav2mpm(
            greatest_child, t_stack, board, rvs, c_rvs, g)
        path.append(node)
        return candid, path


def __mcts_expand(candid, c_rvs):
    """Expand the candidate node.

    Arguments:
        candid {node} -- [description]
        c_rvs {dict<slot, list<int>} -- [description]

    Returns:
        list<node> 
            -- [description]
    """
    candids = list()
    for slot, s_rvs in c_rvs.items():
        if s_rvs is not None:
            for s_rv in s_rvs:
                child = node((slot[0], slot[1], s_rv))
                candid.add_child(child)
                candids.append(child)
    return candids


def __mcts_rollout(candids, board, t_stack, rvs, c_rvs, g, N, f):
    """[summary]

    Arguments:
        candids {[type]} -- [description]
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
        N {[type]} -- [description]
        f {[type]} -- [description]
    """
    wins = list()
    for candid in candids:
        bd.board_state_transition(t_stack, board, rvs, c_rvs, g, candid.action)
        w = 0
        for _ in range(N):
            w += f.sample(board, t_stack, rvs, c_rvs, g)
        wins.append(w)
    return wins


def __mcts_backprop(candids, wins, path, c, N, board, t_stack, rvs, c_rvs, g):
    """Back propagation.

    Arguments:
        candids {[type]} -- [description]
        wins {[type]} -- [description]
        path {[type]} -- [description]
        c {[type]} -- [description]
        N {[type]} -- [description]
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
    """
    for node in path:
        for candid, win in (candids, wins):
            c = node.child_lookup.get(candid)
            if c is not None:
                c.N += N
                c.w += win
                c.score = c.w/c.N + c*m.sqrt(m.log())
                node.update_child(c)
        bd.board_state_restore(t_stack, board, rvs, c_rvs, g)


def mcts_heavy(board, rvs, c_rvs, g, f, the, max_trials=10000):
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
            -- policy-value lambda function.
        the {theta}
            -- MCTS parameters.
        max_trials {}
    """
    root = node(None)
    t_stack = list()
    for i in range(max_trials):
        mpn, path = __mcts_nav2mpm(root, board, t_stack, rvs, c_rvs, g)
        # Apply the moves.
        candids = __mcts_expand(mpn, c_rvs)
        wins = __mcts_rollout(candids, board, t_stack, rvs, c_rvs, g, the.N, f)
        __mcts_backprop(candids, wins, path,
                        the.c, the.N,
                        board, t_stack, rvs, c_rvs, g)
    pass
