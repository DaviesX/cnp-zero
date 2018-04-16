import numpy as np
import itertools as it
from dataset.ifdataset import if_dataset, construct_question_solutions_pairs


def rotate_question(question: np.ndarray) -> np.ndarray:
    """Rotate the board 90 degrees clockwise.

    Arguments:
        question {np.ndarray}
            -- board to be rotated.

    Returns:
        np.ndarray
            -- The rotated board.
    """
    h, w = question.shape[0], question.shape[1]
    rot_question = np.zeros(question.shape, dtype=np.int)
    for i in range(h):
        for j in range(w):
            rot_question[w - 1 - j, i] = question[i, j]
    return rot_question


def rotate_solutions(question: np.ndarray, slns: np.ndarray) -> np.ndarray:
    """Rotate the solution set 90 degrees clockwise.

    Arguments:
        question {np.ndarray}
            -- Question for which the solutions are given.
        slns {np.ndarray}
            -- The solutions set to the question.

    Returns:
        np.ndarray
            -- The rotated solution set.
    """
    _, w = question.shape[0], question.shape[1]
    rot_slns = np.zeros(slns.shape)
    for i in range(slns.shape[0]):
        rot_slns[i, 2:] = slns[i, 2:]
        rot_slns[i, 0] = w - 1 - slns[i, 1]
        rot_slns[i, 1] = slns[i, 0]
    return rot_slns


def replace_question(question: np.ndarray, mapper: list) -> np.ndarray:
    """Replace the board with other numbers.

    Arguments:
        question {np.ndarray}
            -- board to be rotated.
        mapper {list}
            -- number mapper: mapper[input] -> output.

    Returns:
        np.ndarray
            -- the replaced board.
    """
    rep_question = np.zeros(question.shape, dtype=np.int)
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
            rep_question[i, j] = mapper[question[i, j]]
    return rep_question


def replace_solutions(slns: np.ndarray, mapper: list) -> np.ndarray:
    """Replace the action placement number in solution set with the mapped value.

    Arguments:
        slns {np.ndarray}
            -- The solution set to be replaced.
        mapper {list}
            -- number mapper: mapper[input] -> output.

    Returns:
        np.ndarray
            -- The replaced solution set.
    """
    rep_slns = slns.copy()
    for i in range(slns.shape[0]):
        rep_slns[i, 2] = mapper[int(rep_slns[i, 2]) + 1] - 1
    return rep_slns


def reorder_question_rows(question: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    """Swap rows of the board.

    Arguments:
        question {np.ndarray}
            -- the in which rows are to be swapped.
        start {int}
            -- starting row index.
        end {int}
            -- ending row index.
        mapper {list}
            -- row index mapper: mapper[row_index] -> swapped index.

    Returns:
        np.ndarray
            -- the reordered board.
    """
    reordered = question.copy()
    k = end - start + 1
    for i in range(k):
        reordered[start + mapper[i], :] = question[start + i, :]
    return reordered


def reorder_solutions_rows(slns: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    """Swap the row index for the solution set.

    Arguments:
        slns {np.ndarray}
            -- the solution set to be swapped.
        start {int}
            -- starting row index.
        end {int}
            -- ending row index.
        mapper {list}
            -- row index mapper: mapper[row_index] -> swapped index.

    Returns:
        np.ndarray
            -- The row reordered solution set.
    """
    reordered = slns.copy()
    for i in range(reordered.shape[0]):
        if reordered[i, 0] >= start and reordered[i, 0] <= end:
            reordered[i, 0] = start + mapper[int(reordered[i, 0]) - start]
    return reordered


def reorder_question_cols(question: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    """Swap columns of the board.

    Arguments:
        question {np.ndarray}
            -- the in which rows are to be swapped.
        start {int}
            -- starting column index.
        end {int}
            -- ending column index.
        mapper {list}
            -- column index mapper: mapper[col_index] -> swapped index.

    Returns:
        np.ndarray
            -- the reordered board.
    """
    reordered = question.copy()
    k = end - start + 1
    for i in range(k):
        reordered[:, start + mapper[i]] = question[:, start + i]
    return reordered


def reorder_solutions_cols(slns: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    """Swap the column index for the solution set.

    Arguments:
        slns {np.ndarray}
            -- the solution set to be swapped.
        start {int}
            -- starting column index.
        end {int}
            -- ending column index.
        mapper {list}
            -- column index mapper: mapper[col_index] -> swapped index.

    Returns:
        np.ndarray
            -- The row reordered solution set.
    """
    reordered = slns.copy()
    for i in range(reordered.shape[0]):
        if reordered[i, 1] >= start and reordered[i, 1] <= end:
            reordered[i, 1] = start + mapper[int(reordered[i, 1]) - start]
    return reordered


def dataset_create_one(k: int,
                       question_file: str, solutions_file: str,
                       dst_question_file: str, dst_solutions_file: str,
                       perm: int) -> None:
    question = np.load(question_file)
    solutions = np.load(solutions_file)

    number_mapper = [i for i in range(1, k*k+1)]
    variation = 0

    for _ in range(4):
        # rotation.
        rot_question = rotate_question(question)
        rot_solutions = rotate_solutions(question, solutions)
        question = rot_question
        solutions = rot_solutions
        for m in range(k):
            for j0 in it.permutations([i for i in range(k)], k):
                # row swap.
                row_question = reorder_question_rows(rot_question,
                                                     k * m, k * (m + 1) - 1,
                                                     j0)
                row_solutions = reorder_solutions_rows(rot_solutions,
                                                       k * m, k * (m + 1) - 1,
                                                       j0)
                rot_question = row_question
                rot_solutions = row_solutions

                for n in range(k):
                    for j1 in it.permutations([i for i in range(k)], k):
                        # col swap.
                        col_question = reorder_question_cols(row_question,
                                                             k * n,
                                                             k * (n + 1) - 1,
                                                             j1)
                        col_solutions = reorder_solutions_cols(row_solutions,
                                                               k * n,
                                                               k * (n + 1) - 1,
                                                               j1)
                        row_question = col_question
                        row_solutions = col_solutions
                        for _ in range(2):
                            # number replacement.
                            np.random.shuffle(number_mapper)
                            actual_number_mapper = [0] + number_mapper
                            rep_question = replace_question(row_question,
                                                            actual_number_mapper)
                            rep_solutions = replace_solutions(row_solutions,
                                                              actual_number_mapper)

                            file_prefix = dst_question_file + \
                                "_" + str(variation)
                            print("generating " + file_prefix)
                            np.save(file_prefix + ".q",
                                    rep_question)
                            np.save(file_prefix + ".s",
                                    rep_solutions)
                            variation += 1


def dataset1_create(k: int, src_path: str, dst_path: str, perm=10) -> None:
    question_files, solutions_files = \
        construct_question_solutions_pairs(src_path)

    for i in range(len(question_files)):
        question_file, solutions_file = question_files[i], solutions_files[i]
        dataset_create_one(k,
                           src_path + "/" + question_file,
                           src_path + "/" + solutions_file,
                           dst_path + "/" + question_file.split(".")[0],
                           dst_path + "/" + solutions_file.split(".")[0],
                           perm)


class dataset1(if_dataset):
    def __init__(self, storage_path: str):
        pass

    def ring_fetch(self, amount: int) -> tuple:
        pass

    def random_fetch(self, amount: int) -> tuple:
        pass
