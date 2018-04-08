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
    _, w = question.shape[0], question.shape[1]
    rot_question = np.zeros(question.shape, dtype=np.int)
    for i in range(question.shape[0]):
        for j in range(question.shape[1]):
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
    pass


def replace_solutions(slns: np.ndarray, mapper: list) -> np.ndarray:
    pass


def reorder_question_rows(question: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    pass


def reorder_solutions_rows(slns: np.ndarray, start: int, end: int, mapper: list) -> np.ndarray:
    pass


def dataset1_create(k: int, src_path: str, dst_path: str, perm=10) -> None:
    question_files, solution_files = \
        construct_question_solutions_pairs(src_path)

    for i in range(len(question_files)):
        question_file, solution_file = question_files[i], solution_files[i]
        question = np.load(question_file)
        solutions = np.load(solution_file)
        for i in range(4):
            # rotation.
            question = rotate_question(question)
            solutions = rotate_solutions(question, solutions)
            for m in range(k):
                # row swap
                for n in range(k):
                    # col swap
                    for j in it.permutations([i for i in range(k)], k):
                        # row and col permutations.
                        for l in range(10):
                            # number replacement.
                            pass


class dataset1(if_dataset):
    def __init__(self, storage_path: str):
        pass

    def ring_fetch(self, amount: int) -> tuple:
        pass

    def random_fetch(self, amount: int) -> tuple:
        pass
