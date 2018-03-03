import math as m
import numpy as np
import board as bd
import policy


class theta:
    """MCTS parameters
    """

    def __init__(self, b=1.0, c=1.41, N=1):
        """[summary]

        Keyword Arguments:
            b {float}
                -- RAVE to truth transition factor. (default: {1.0})
            c {float} 
                -- Exploration factor (default: {1.41})
            N {int} 
                -- Number of rollouts per iteration (default: {1})
        """
        self.c = c
        self.b = b
        self.N = N


class node:
    """MCT node
    """

    def __init__(self, action):
        self.action = action
        self.nh = 0
        self.nw = 0
        self.n = 0
        self.w = 0
        self.children = list()
        self.child_lookup = dict()

    def __str__(self):
        return "node: {" + self.action + ", " + self.w/self.w + ", " + self.nw/self.nh + "}"


def __mcts_score(node, b, c, t):
    """[summary]

    Arguments:
        node {[type]} -- [description]
        b {[type]} -- [description]
        c {[type]} -- [description]
        t {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    beta = node.nh/(node.n + node.nh + 4*b*b*node.n*node.nh)
    return (1 - beta)*node.w/node.n + beta*node.wh/node.nh + c*m.sqrt(m.log(t)/node.n)


def __mcts_nav2mpm(node, b, c, t, board, t_stack, rvs, c_rvs, g):
    """Navigate to most promising move.

    Arguments:
        node {[type]} -- [description]
        b {[type]} -- [description]
        c {[type]} -- [description]
        t {[type]} -- [description]
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]

    Returns:
        candid {node} 
            -- candidate node
        path {list<node>}
            -- from candidate node to root.
    """
    if t is 0:
        return node, []

    if not node.children:
        return node, [node]
    else:
        greatest_child = node.children[0]
        grestest_score = __mcts_score(node.children[0], b, c, t)
        for i in range(1, len(node.children)):
            score = __mcts_score(node.children[i], b, c, t)
            if score > grestest_score:
                greatest_child = node.children[i]
                grestest_score = score
        bd.board_state_transition(
            t_stack, board, rvs, c_rvs, g, greatest_child.action)
        candid, path = __mcts_nav2mpm(
            greatest_child, b, c, t, t_stack, board, rvs, c_rvs, g)
        path.append(node)
        return candid, path


def __mcts_expand(candid, c_rvs):
    """Expand the candidate node.

    Arguments:
        candid {node} -- [description]
        c_rvs {dict<(i,j),int>} 
            -- a dictionary of remaining value counters

    Returns:
        candids {list<node>}
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


def __mcts_rollout(candids, d, td, board, t_stack, rvs, c_rvs, g, N, f):
    """[summary]

    Arguments:
        d {int}
            -- current depth.
        td {int}
            -- total depth.
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
        N {[type]} -- [description]
        f {[type]} -- [description]

    Returns:
        wins {list<int>}
            -- number of wins for each candidate.
        nt {int}
            -- number of rollout performed after the parent move.
    """
    wins = list()
    for candid in candids:
        bd.board_state_transition(t_stack, board, rvs, c_rvs, g, candid.action)
        f.init(board, t_stack, rvs, c_rvs,)
        w = 0
        for _ in range(N):
            w += f.sample(d + 1, td, board, t_stack, rvs, c_rvs, g)
        wins.append(w)
        bd.board_state_restore(t_stack, board, rvs, c_rvs, g)

    return wins, N*len(candids)


def __mcts_backprop(candids, N, wins, nt, wt, path, board, t_stack, rvs, c_rvs, g):
    """Back propagation.

    Arguments:
        candids {[type]} -- [description]
        N {[type]} -- [description]
        wins {[type]} -- [description]
        nt {int}
            -- total number of rollouts after the parent move.
        wt {int}
            -- total number of wins after the parent move.

        path {[type]} -- [description]
        board {[type]} -- [description]
        t_stack {[type]} -- [description]
        rvs {[type]} -- [description]
        c_rvs {[type]} -- [description]
        g {[type]} -- [description]
    """
    # update the newly expanded level.
    for i in range(len(candids)):
        candids[i].n += N
        candids[i].w += wins[i]

    # back prop.
    for node in path:
        node.n += nt
        node.w += wt
        for i in range(len(candids)):
            candid = candids[i]
            win = wins[i]
            c = node.child_lookup.get(candid.action)
            if c is not None:
                c.nh += N
                c.wh += win
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
        max_trials {int}
            -- number of MCTS runs.
    Returns:
        wins {list<int>}
            -- number of wins for each candidate.
        nt {int}
            -- number of rollout performed after the parent move.
    """
    td = len(c_rvs)

    root = node(None)
    t_stack = list()
    t = 0
    for _ in range(1, max_trials + 1):
        mpn, path = __mcts_nav2mpm(
            root, the.b, the.c, t, board, t_stack, rvs, c_rvs, g)
        candids = __mcts_expand(mpn, c_rvs)
        wins, nt = __mcts_rollout(
            candids, len(path), td, board, t_stack, rvs, c_rvs, g, the.N, f)
        t += nt
        __mcts_backprop(candids, the.N, wins, nt, sum(wins), path,
                        board, t_stack, rvs, c_rvs, g)

    # return the most simulated child.
    grestest_n = root.children[0].n
    best_child = root.children[0]
    for i in range(1, len(root.children)):
        if root.children[i].n > grestest_n:
            best_child = root.children[i]
            grestest_n = root.children[i].n

    return best_child


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

    best_child = mcts_heavy(question, rvs, c_rvs, g, policy.mrv.mrv(), theta())
    print(best_child)
