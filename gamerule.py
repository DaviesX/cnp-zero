
def win_metric(D, T):
    """Poportion done transformed to the amount of win (smooth, continuous d^2y/dx^2).

    Arguments:
        D {float} -- number of slot solved.
        T {float} -- total amount of slots.

    Returns:
        float 
            -- amount of win.
    """
    return (D/T)**10
