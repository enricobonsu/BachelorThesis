import numpy as np


class CarlaWorld:

    def __init__(self, states=89):

        self.actions = [0, 1]  # stop or not

        self.n_states = states
        self.n_actions = len(self.actions)

        # self.p_transition = self._transition_prob_table()
