
def csp_mrv_expansion(question, rvs):


def csp_mrv_solve(question, rvs, i, j, m):


def csp_mrv(question, rvs):
    """Solve the cnp using csp with minimum remaining values heuristics.

    Arguments:
        question {[N, N]} -- a [k*k, k*k] shaped integer matrix, with zero representing the free slot.
        rvs {[N, N, N]} -- a boolean tensor representing the remaining values for each slot.
    Returns:
        {[N, N]} -- a solved cnp.
    """
