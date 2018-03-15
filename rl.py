import sys
import uuid
import mcts as s
import board as bd
import numpy as np
from policy.ifpolicy import if_policy
from policy.mrv import mrv


def guid(i: int) -> str:
    return ((str(i) + "_") if i is not None else "") + str(uuid.uuid4())


def save(dst: str, question: np.ndarray, possible_actions: list) -> None:
    np.save(dst + ".q", question)

    actions = np.zeros([len(possible_actions), 4], dtype=np.int)
    for i in range(len(possible_actions)):
        actions[i, 0] = possible_actions[i][0][0]
        actions[i, 1] = possible_actions[i][0][1]
        actions[i, 2] = possible_actions[i][0][2]
        actions[i, 3] = possible_actions[i][1]

    np.save(dst + ".s", actions)


def best_action(possible_actions: list) -> tuple:
    """[summary]

    Arguments:
        possible_actions {list} -- [description]

    Returns:
        tuple -- [description]
    """
    freq = dict()
    for possible_action in possible_actions:
        record = freq.get(possible_action[0])
        if record is None:
            record = 0
        freq[possible_action[0]] = record + 1

    most_sim = None
    max_num_sim = 0
    for action, f in freq.items():
        if f > max_num_sim:
            max_num_sim = f
            most_sim = action

    return most_sim


def solution_collection(storage_path: str,
                        policy: if_policy,
                        k=3, e=4.0, std=1.0, max_iters=10000, num_boards=10) -> None:
    for step in range(num_boards):
        # prepare question.
        _, _, question = bd.create_random(k, e, std)
        num_holes = np.sum(question == 0)
        print("question: " + str(question) + ", holes " + str(num_holes))
        # prepare valid moves.
        t_stack = list()
        rvs, c_rvs, g = bd.remaining_values_deg(question)
        # solve through the question.
        for _ in range(num_holes):
            # search for possible actions.
            possible_actions = list()
            i = 0
            while i < max_iters:
                _, completion, t, actions = s.mcts_heavy(question,
                                                         rvs, c_rvs, g,
                                                         policy, s.theta(), max_trials=max_iters)
                i += t
                if completion == num_holes:
                    print("Completed at " + str(i) +
                          ", choose " + str(actions[0]))
                    possible_actions.append((actions[0], t))
            # save current example.
            dpid = guid(step)
            save(storage_path + "/" + dpid, question, possible_actions)
            # select the best action and move to the next state.
            next_action = best_action(possible_actions)
            bd.board_state_transition(t_stack,
                                      question,
                                      rvs, c_rvs, g,
                                      next_action)
            print("Transition to " + str(next_action))


if __name__ == "__main__":
    solution_collection("./dataset/cnp3/original", mrv(), k=3, e=4)
