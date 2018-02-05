

def policy_improvement_operator(board, valid_movs, f_the, m_the):
        """ MCTS with UCB: Q(s,a) + alpha*(Pr(s,a | f_the)/(1 + N(s, a))).

        Arguments:
                board {[NxN]} -- the board.
                valid_movs {[NxN set]} -- NxN sets representing valid moves.
                f_the {[policy-value network]} -- policy-value network parameters.
                m_the {[alpha]} -- MCTS parameters.
        """

        pass
