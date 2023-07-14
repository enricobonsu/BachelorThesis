import numpy as np
import pandas as pd
from DataFrameTrajectory import DataFrameTrajectory
from carlaTrajectory import Trajectory


class Demonstration:
    def __init__(self):
        dfTrajectories = DataFrameTrajectory()
        self.trajectories = []

        # State table contains the state and its features as a dataframe.
        self.stateTable = dfTrajectories.stateTable
        # print(self.stateTable)

        # dfTrajectories.dfs is a list of trajectories that are stores as df
        for df in dfTrajectories.dfs:
            self.trajectories.append(Trajectory(df, self.stateTable))

        # The terminal states observed during the demonstration
        self.terminalStates = self.terminalState(self.trajectories)

        # The transition table for every state-action pair
        self.transitionTable = self.generateTransitionTable()

        # The probability of transitioning to state s for every state-action pair
        self.p_transition = self.generateProbTransition()

    def generateProbTransition(self, debug=False):
        """
        Function used to create the transition probabilities for every (s,a,s')
        """
        n_states = self.stateTable.shape[0]

        stateActions = list(self.transitionTable.keys())
        possibleActions = set()
        for stateAction in stateActions:
            possibleActions.add(int(stateAction.split("-")[1]))

        n_actions = len(possibleActions)
        pTable = np.zeros(shape=(n_states, n_actions, n_states))

        # Set the transition probability for each state-action pair to equally likely for all the possible next-states
        for stateAction, outcomeStates in self.transitionTable.items():
            pair = stateAction.split("-")
            outcomes = set(outcomeStates)
            n_outcomes = len(outcomes)
            while outcomes:
                state = outcomes.pop()
                pTable[int(pair[0]), int(pair[1]), state] = 1/n_outcomes

        # list of custom probabilities holding [s,a,s',p]
        customProbabilities = [[24, 1, 24, 0.99], [24, 1, 25, 0.01],
                               [24, 0, 26, 0.99], [24, 0, 27, 0.01],

                               [26, 1, 26, 0.99], [26, 1, 27, 0.01],
                               [26, 0, 28, 0.99], [26, 0, 29, 0.01],

                               [28, 1, 28, 0.99], [28, 1, 29, 0.01],
                            #    [28, 0, 28, 0.99], [28, 1, 29, 0.01],
                               
                               [25, 1, 25, 0.99], [25, 1, 24, 0.01],
                               [25, 0, 27, 0.99], [25, 0, 26, 0.01],

                               [27, 1, 27, 0.99], [27, 1, 26, 0.01],
                               [27, 0, 29, 0.99], [27, 0, 28, 0.01],

                               [29, 1, 29, 0.99], [29, 1, 28, 0.01],
                               ]
        
        for probabilty in customProbabilities:
            pTable[probabilty[0], probabilty[1], probabilty[2]] = probabilty[3]

        if debug:
            for s in range(pTable.shape[0]):
                for a in range(pTable.shape[1]):
                    for s_prime in range(pTable.shape[2]):
                        if pTable[s][a][s_prime]>0:
                            print("(", s,a, s_prime,") Has probability", pTable[s][a][s_prime])
 
        if debug:
            countPossibleTransitions = 0
            for transitions in pTable:
                countPossibleTransitions += np.count_nonzero(transitions)
            print("Number of possible transitions", countPossibleTransitions)

        return pTable

    def generateTransitionTable(self):
        """"
        Function which generates a dictonary with the transitions for each state-action pair
        """
        transitionDf = pd.read_csv("stateTransitions.csv")

        stateActionTransitions = dict()
        for _, row in transitionDf.iterrows():
            stateActionTransitions[row['state-action']
                                   ] = set(eval(row['transitions']))
        return stateActionTransitions

    def terminalState(self, trajectories):
        """
        Function to obtain the terminal states for every trajectory
        """
        terminals = set()
        for traj in trajectories:
            terminals.add(traj.terminalState)
        return terminals


def main():
    demo = Demonstration()


if __name__ == '__main__':
    main()
