import numpy as np
from dataset.ifdataset import if_dataset, construct_question_solutions_pairs


def dataset1_create(src_path: str, dst_path: str, perm=10) -> None:
    for i in range(4):
        # rotation.
        pass


class dataset1(if_dataset):
    def __init__(self, storage_path: str):
        pass

    def ring_fetch(self, amount: int) -> tuple:
        pass

    def random_fetch(self, amount: int) -> tuple:
        pass
