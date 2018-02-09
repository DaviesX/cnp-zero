import numpy as np
import scipy.stats
import scipy as sp


def create_random(k, e_block, e_dev):
    """ Return a randomly filled [k*k, k*k] shaped integer matrix
        and a boolean matrix of same size indicating the fillable positions.

    Arguments:
        k {int} -- width of each block.
        e_block {float} -- the expected number of removal for each block.
        e_dev {float} -- deviation from the removal expectation.
    Returns:
        {[k*k, k*k], [k*k, k*k], [k*k, k*k]} -- answer, free slots, question.
    """
    if k < 2:
        raise Exception("Block width k < 2.")

    if e_block > k*k:
        raise Exception("Cannot remove more than what could be in a block.")

    width = k * k

    # board generation.
    board = np.zeros([width, width], dtype=np.int)
    for i in range(k):
        for j in range(k):
            if i == 0 and j == 0:
                # bootstrap an initial grid.
                board[0:k, 0:k] = np.reshape(np.arange(1, k*k+1), [k, k])
            elif i >= 1 and j == 0:
                # round-robin on block columns.
                board[i*k:(i+1)*k, 1:k] = board[(i-1)*k:i*k, 0:k-1]
                board[i*k:(i+1)*k, 0] = board[(i-1)*k:i*k, k-1]
            else:
                # round-robin on rows.
                board[i*k+1:(i+1)*k, j*k:(j+1)*k] = board[i *
                                                          k:(i+1)*k-1, (j-1)*k:j*k]
                board[i*k, j*k:(j+1)*k] = board[(i+1)*k-1, (j-1)*k:j*k]

    # structure-wise shuffling.
    row_shuffle = np.arange(0, width)
    col_shuffle = np.arange(0, width)

    for i in range(k):
        np.random.shuffle(row_shuffle[i*k:(i+1)*k])
    for j in range(k):
        np.random.shuffle(col_shuffle[j*k:(j+1)*k])

    t1 = np.zeros([width, width], dtype=np.int)
    for i in range(width):
        t1[i, :] = board[row_shuffle[i], :]
    for i in range(width):
        board[:, i] = t1[:, col_shuffle[i]]

    # number replacement.
    replacements = np.arange(1, k*k+1)
    np.random.shuffle(replacements)
    t2 = np.zeros([width, width], dtype=np.int)
    for i in range(1, k*k + 1):
        t2[board == i] = replacements[i-1]
    board = t2

    # take away elements.
    stones = np.ones([width, width], dtype=np.int)
    rvt = np.floor(sp.stats.truncnorm.rvs(
        0, k, loc=e_block, scale=e_dev, size=k*k))
    for i in range(k):
        for j in range(k):
            n_takeawys = int(rvt[j + i*k])
            takeaway_rows = np.random.randint(0, k, n_takeawys)
            takeaway_columns = np.random.randint(0, k, n_takeawys)

            for m in takeaway_rows:
                for n in takeaway_columns:
                    stones[i*k + m, j*k + n] = 0

    return [board, stones, board * stones]


def validate(board):
    """Test if the board a valid answer

    Arguments:
        board {[N, N]} -- a [k*k, k*k] shaped integer matrix.
    Returns:
        {bool} -- whether the solution is valid.
    """
    k = int(np.sqrt(board.shape[0]))
    for row in board:
        u = np.unique(row[row != 0])
        if u.shape[0] != board.shape[0]:
            return False
    for col in np.transpose(board):
        u = np.unique(row[row != 0])
        if u.shape[0] != board.shape[0]:
            return False
    for i in range(k):
        for j in range(k):
            u = np.unique(board[i*k:(i+1)*k, j*k:(j+1)*k])
            if u.shape[0] != board.shape[0]:
                return False
    return True


def remaining_values_deg(question):
    """Compute a remaining value voxel and a dependency graph.

    Arguments:
        question {[N, N]} -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
    Returns:
        {[N, N, N], g} -- a boolean tensor representing the remaining values for each slot;
                          a dependency graph
    """
    N = question.shape[0]
    k = int(np.sqrt(N))
    universe = np.arange(1, N+1)
    row_rvs = np.zeros([N, N], dtype=np.bool)
    col_rvs = np.zeros([N, N], dtype=np.bool)
    block_rvs = np.zeros([k, k, N], dtype=np.bool)

    # calculate remaining values in each row.
    for i in range(N):
        remaining = np.setdiff1d(universe, question[i, :])
        row_rvs[i][remaining-1] = True

    # calculate remaining values in each column.
    for j in range(N):
        remaining = np.setdiff1d(universe, question[:, j])
        col_rvs[j][remaining-1] = True

    # calculate remaining values in each block.
    for i in range(k):
        for j in range(k):
            block = question[i*k:(i+1)*k, j*k:(j+1)*k]
            remaining = np.setdiff1d(universe, np.reshape(block, [k*k]))
            block_rvs[i, j, remaining-1] = True

    # calculate remaining values overall.
    rvs = np.zeros([N, N, N], dtype=np.bool)
    free_slots = np.stack(np.where(question == 0), axis=1)
    for i, j in free_slots:
        rvs[i, j, :] = row_rvs[i, :] & col_rvs[j, :] & block_rvs[i//k, j//k, :]

    # compute dependencies.
    g = dict()
    deg = dict()
    for i, j in free_slots:
        # find related slots.
        bi = i//k
        bj = i//k
        block_belongings = free_slots//k
        related = free_slots[(free_slots[:, 0] == i) |
                             (free_slots[:, 1] == j) |
                             (block_belongings[:, 0] == bi) |
                             (block_belongings[:, 1] == bj)]

        # create edges and calculate degree.
        for slot in related:
            if tuple(slot) != (i, j):
                linked = np.where(rvs[i, j, :] & rvs[slot[0], slot[1], :])[0]
                for m in linked:
                    neighbors = g.get((i, j, m))
                    neighbors = set() if neighbors is None else neighbors
                    neighbors.add(tuple(slot))
                    g[(i, j, m)] = neighbors
    return rvs, g


class transition:
    """repesents board state transition
    """

    def __init__(self, i, j, m, related_links):
        self.related_links = related_links
        self.action = (i, j, m)
        self.affected_links = set()

    def add_affected_link(self, slot):
        self.affected_links.add(slot)


def board_state_transition(t_stack, question, rvs, g, i, j, m):
    """transition the current state by taking the action (i,j,m).

    Arguments:
        t_stack {list<transition>} -- transition stack
        question {ndarray<N,N>} -- [description]
        rvs {ndarray<N,N,N>} -- [description]
        g {dict<(i,j,m), set<(i,j)>>} -- [description]
        i {int} -- [description]
        j {int} -- [description]
        m {int} -- [description]
    """

    # record the state and action to be taken
    related_links = g.get((i, j, m))
    cur_transition = transition(i, j, m, related_links)

    # update dependency graph and validity.
    if related_links is not None:
        for slot in related_links:
            link_from_slot = g[(slot[0], slot[1], m)]
            if (i, j) in link_from_slot:
                link_from_slot.remove((i, j))
                cur_transition.add_affected_link(slot)
                rvs[slot[0], slot[1], m] = False

        g[(i, j, m)] = None

    rvs[i, j, m] = False
    question[i, j] = m + 1

    t_stack.append(cur_transition)


def board_state_restore(t_stack, question, rvs, g):
    """backtrack the state by reverting the action recorded in t_stack.

    Arguments:
        t_stack {list<transition>} -- [description]
        question {ndarray<N,N>} -- [description]
        rvs {ndarray<N,N,N>} -- [description]
        g {dict<(i,j,m), set<(i,j)>>} -- [description]
    """

    t = t_stack.pop()
    i, j, m = t.action

    # restore the dependency graph and validity.
    g[(i, j, m)] = t.related_links
    if t.affected_links is not None:
        for slot in t.affected_links:
            link_from_slot = g[(slot[0], slot[1], m)]
            if link_from_slot is not None:
                link_from_slot.add((i, j))
            rvs[slot[0], slot[1], m] = True

    rvs[i, j, m] = True
    question[i, j] = 0


def __transit_first(t_stack, question, rvs, g):
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
            if question[i, j] == 0:
                m = np.where(rvs[i, j])[0][0]
                board_state_transition(t_stack, question, rvs, g, i, j, m)
                return


def __print_first_question_slot(question, rvs, g):
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
            if question[i, j] == 0:
                print(rvs[i, j, :])
                print(g[(i, j, 0)])
                return


if __name__ == "__main__":
    # test
    np.random.seed(13)
    k = 3
    e = int(k*k/3)
    s = 1
    board, stones, question = create_random(k, e, s)
    print(board)
    print(stones)
    print(question)
    print(validate(board))
    print(validate(question))
    rvs, g = remaining_values_deg(question)
    print(rvs[0: 2, 0: 2, :])
    __print_first_question_slot(question, rvs, g)
    print("total dof: " + str(np.sum(rvs)))
    print("average dof: " + str(np.sum(rvs)/np.sum(1 - stones)))
    t_stack = list()

    old_question = np.copy(question)
    old_rvs = np.copy(rvs)
    old_g = g.copy()

    __transit_first(t_stack, question, rvs, g)
    board_state_restore(t_stack, question, rvs, g)

    print(np.all(old_question == question))
    print(np.all(old_rvs == rvs))
    print(old_g == g)
