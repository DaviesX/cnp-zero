import numpy as np
import scipy.stats
import scipy as sp


def create_random(k, e_block, e_dev):
    """ Return a randomly filled [k*k, k*k] shaped integer matrix
        and a boolean matrix of same size indicating the fillable positions.

    Arguments:
        k {int} 
            -- width of each block.
        e_block {float} 
            -- the expected number of removal for each block.
        e_dev {float} 
            -- deviation from the removal expectation.
    Returns:
        {[k*k, k*k], [k*k, k*k], [k*k, k*k]} 
            -- answer, free slots, question.
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
        board {[N, N]} 
            -- a [k*k, k*k] shaped integer matrix.
    Returns:
        {bool} 
            -- whether the solution is valid.
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
        question {[N,N]} 
            -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
    Returns:
        {ndarray<N,N,N>, dict<(i,j),int>, dict<(i,j,m),set<i,j>>} 
                -- a boolean tensor representing the remaining values for each slot;
                -- a dictionary of remaining values (compact representation of the first);
                -- a dependency graph.
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

    # remaining value counters
    c_rvs = dict()
    for i, j in free_slots:
        c_rvs[(i, j)] = set(np.where(rvs[i, j, :])[0])

    # compute dependencies.
    g = dict()
    for i, j in free_slots:
        # find related slots.
        bi = i//k
        bj = j//k
        block_belongings = free_slots//k
        related = free_slots[(free_slots[:, 0] == i) |
                             (free_slots[:, 1] == j) |
                             ((block_belongings[:, 0] == bi) &
                              (block_belongings[:, 1] == bj))]

        # create edges.
        for slot in related:
            if tuple(slot) != (i, j):
                linked = np.where(rvs[i, j, :] & rvs[slot[0], slot[1], :])[0]
                for m in linked:
                    neighbors = g.get((i, j, m))
                    neighbors = set() if neighbors is None else neighbors
                    neighbors.add(tuple(slot))
                    g[(i, j, m)] = neighbors

    return rvs, c_rvs, g


class transition:
    """repesents board state transition
    """

    def __init__(self, i, j, m, c_rvs):
        self.action = (i, j, m)
        self.c_rvs = c_rvs
        self.rvs_removed = list()

    def add_removed_rvs(self, slot):
        self.rvs_removed.append(slot)


def board_state_transition(t_stack, question, rvs, c_rvs, g, action):
    """transition the current state by taking the action (i,j,m).

    Arguments:
        t_stack {list<transition>} 
            -- transition stack.
        question {ndarray<N,N>} 
            -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
        rvs {ndarray<N,N,N>} 
            -- a boolean tensor representing the remaining values for each slot.
        c_rvs {dict<(i,j),int>}
            -- a dictionary of remaining value counters
        g {dict<(i,j,m), set<(i,j)>>} 
            -- a dependency graph.
        action {<(i,j,m)>}
            -- transitioning to the ith column, jth row and mth value, (zero offset)
    """
    i, j, m = action

    # record the state and action to be taken
    related_links = g.get((i, j, m))
    t = transition(i, j, m, c_rvs[(i, j)])

    # update remain values.
    if related_links is not None:
        for related_link in related_links:
            rvs_at_link = c_rvs[related_link]
            if rvs_at_link is not None and m in rvs_at_link:
                t.add_removed_rvs(related_link)
                rvs_at_link.remove(m)
                rvs[related_link[0], related_link[1], m] = False

    c_rvs[(i, j)] = None
    rvs[i, j, :] = False

    question[i, j] = m + 1

    t_stack.append(t)


def board_state_restore(t_stack, question, rvs, c_rvs, g):
    """backtrack the state by reverting the action recorded in t_stack.

    Arguments:
        t_stack {list<transition>} 
            -- transition stack.
        question {ndarray<N,N>} 
            -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
        rvs {ndarray<N,N,N>} 
            -- a boolean tensor representing the remaining values for each slot.
        c_rvs {dict<(i,j),int>}
            -- a dictionary of remaining value counters
        g {dict<(i,j,m), set<(i,j)>>} 
            -- a dependency graph.
    """

    t = t_stack.pop()
    i, j, m = t.action

    question[i, j] = 0

    # restore previous remaining value states.
    c_rvs[(i, j)] = t.c_rvs

    for cur_rv in t.c_rvs:
        rvs[i, j, cur_rv] = True

    for rv_removed in t.rvs_removed:
        c_rvs[rv_removed].add(m)
        rvs[rv_removed[0], rv_removed[1], m] = True


def __transit_first(t_stack, question, rvs, c_rvs, g):
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
            if question[i, j] == 0:
                m = np.where(rvs[i, j])[0][0]
                board_state_transition(
                    t_stack, question, rvs, c_rvs, g, (i, j, m))
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
    rvs, c_rvs, g = remaining_values_deg(question)
    print(rvs[0: 2, 0: 2, :])
    __print_first_question_slot(question, rvs, g)
    print("total dof: " + str(np.sum(rvs)))
    print("average dof: " + str(np.sum(rvs)/np.sum(1 - stones)))
    t_stack = list()

    old_question = np.copy(question)
    old_rvs = np.copy(rvs)
    old_g = g.copy()
    old_crvs = c_rvs.copy()

    __transit_first(t_stack, question, rvs, c_rvs, g)
    board_state_restore(t_stack, question, rvs, c_rvs, g)

    print(np.all(old_question == question))
    print(np.all(old_rvs == rvs))
    print(old_g == g)
    print(old_crvs == c_rvs)
