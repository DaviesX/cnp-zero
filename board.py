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


def remaining_values(question):
    """Compute a remaining value voxel.

    Arguments:
        question {[N, N]} -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
    Returns:
        {[N, N, N]} -- a boolean tensor representing the remaining values for each slot.
    """
    N = question.shape[0]
    k = int(np.sqrt(N))
    universe = np.arange(1, N+1)
    row_rvs = np.zeros([N, N], dtype=np.bool)
    col_rvs = np.zeros([N, N], dtype=np.bool)
    block_rvs = np.zeros([k, k, N], dtype=np.bool)

    for i in range(N):
        remaining = np.setdiff1d(universe, question[i, :])
        row_rvs[i][remaining-1] = True

    for j in range(N):
        remaining = np.setdiff1d(universe, question[:, j])
        col_rvs[j][remaining-1] = True

    for i in range(k):
        for j in range(k):
            block = question[i*k:(i+1)*k, j*k:(j+1)*k]
            remaining = np.setdiff1d(universe, np.reshape(block, [k*k]))
            block_rvs[i, j, remaining-1] = True

    rvs = np.zeros([N, N, N], dtype=np.bool)
    for i in range(N):
        for j in range(N):
            if question[i, j] == 0:
                rvs[i, j, :] = np.logical_and(np.logical_and(row_rvs[i, :],
                                                             col_rvs[j, :]),
                                              block_rvs[int(i/k), int(j/k), :])

    return rvs


def print_first_question_slot(question, rvs):
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
            if question[i, j] == 0:
                print(rvs[i, j, :])
                return


if __name__ == "__main__":
    # test
    np.random.seed(13)
    k = 16
    e = int(k*k/3)
    s = 1
    board, stones, question = create_random(k, e, s)
    print(board)
    print(stones)
    print(question)
    print(validate(board))
    print(validate(question))
    rvs = remaining_values(question)
    print(rvs[0:2, 0:2, :])
    print_first_question_slot(question, rvs)
    print("total dof: " + str(np.sum(rvs)))
    print("average dof: " + str(np.sum(rvs)/np.sum(1 - stones)))
