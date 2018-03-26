import os
import numpy as np


def construct_question_solutions_pairs(src_path: str) -> tuple:
    # traverse directory
    problem_set = dict()
    for file_name in os.listdir(src_path):
        parts = file_name.split(".")
        problem_id = parts[0]
        file_type = parts[1]

        problem = problem_set.get(problem_id)
        if problem is None:
            problem = [None, None]
        if file_type == "q":
            # question
            problem[0] = file_name
        elif file_type == "s":
            # solution
            problem[1] = file_name
        problem_set[problem_id] = problem

    # flatten
    questions = list()
    solutions = list()
    for _, t in problem_set.items():
        questions.append(t[0])
        solutions.append(t[1])

    return questions, solutions


def compact_solutions(solutions: np.ndarray) -> np.ndarray:
    """[summary]

    Arguments:
        solutions {np.ndarray} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    mov_freq = dict()
    for solution in solutions:
        action = (solution[0], solution[1], solution[2])
        freq = mov_freq.get(action)
        if freq is None:
            freq = [1, solution[3]]
            mov_freq[action] = freq
        else:
            freq[0] += 1
            freq[1] += solution[3]

    compacted = np.zeros((len(mov_freq), 5))
    i = 0
    for action, freq in mov_freq.items():
        mu_step = freq[1] / freq[0]
        p_selected = freq[0] / solutions.shape[0]

        compacted[i, :3] = np.array(action)
        compacted[i, 3:] = np.array([mu_step, p_selected])

        i += 1

    return compacted


class if_dataset:
    def __init__(self):
        pass

    def ring_fetch(self, amount: int) -> np.ndarray:
        pass

    def random_fetch(self, amount: int) -> np.ndarray:
        pass


if __name__ == "__main__":
    q, s = construct_question_solutions_pairs("./cnp3/original")
    for i in range(len(q)):
        print("question: " + str(q[i]) + ", " + "solution: " + str(s[i]))
    for sln_file_name in s:
        slns = np.load("./cnp3/original/" + sln_file_name)
        comp_slns = compact_solutions(slns)
        np.save("./cnp3/compacted/" + sln_file_name, comp_slns)
