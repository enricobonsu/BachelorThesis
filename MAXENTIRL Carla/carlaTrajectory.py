import numpy as np
import pandas as pd
from DataFrameTrajectory import DataFrameTrajectory


class Trajectory:

    def __init__(self, df, stateTable):
        self.featureNames = df.columns.values

        # The initial state for this trajectory
        self.startState = self.findStateInStateTable(
            df.iloc[0, :2].to_numpy(dtype=np.float64), stateTable)

        # The terminal state for this trajectory
        self.terminalState = self.findStateInStateTable(
            df.iloc[-1, :2].to_numpy(dtype=np.float64), stateTable)
        
        # Transitions is a matrix with the transition observed during the trajectory
        # stateVisitationFrequency contains the frequency that a state has been visited during this trajectory
        self.transitions, self.stateVisitationFrequency = self.generateTrajectionTransition(
            df, stateTable)

    def generateTrajectionTransition(self, dfTrajectory, stateTable, debug=False):
        # Returns matrix that contains the state,action,state' for every step,
        # with row 0 the intial state.
        # Also returns state visitation frequencey for this trajectory.

        steps = dfTrajectory.to_numpy(dtype=np.float64)
        prevState = self.startState
        prevAction = steps[0, 2]    
        steps = np.delete(steps, 0, axis=0)
        trajectoryTransition = np.empty((0, 3), dtype=int)

        # Create a matrix that contains the (state, action, next-state) transition
        for step in steps:
            newState = self.findStateInStateTable(step[:2], stateTable)
            trajectoryTransition = np.append(trajectoryTransition, np.array(
                [[prevState, prevAction, newState]]), axis=0)

            prevState = newState
            prevAction = step[2]

        vistedStates, stateVisitationFrequency = np.unique(
            trajectoryTransition[:, 2], axis=0, return_counts=True)

        # Add the visit of the initial state to the arrays
        vistedStates = np.append([self.startState], vistedStates)
        stateVisitationFrequency = np.append([1], stateVisitationFrequency)

        trajectoryTransition, count = np.unique(
            trajectoryTransition, axis=0, return_counts=True)
        stateVisitationFrequency = dict(
            zip(vistedStates, stateVisitationFrequency))

        if debug:
            print("State feature names",
                  self.featureNames[:-1], "with action name", self.featureNames[-1])
            for index, i in enumerate(trajectoryTransition):
                print(i, "beginStateFeatures", stateTable.iloc[int(i[0])].tolist(), "- action", i[1],
                      "- endStateFeatures", stateTable.iloc[int(i[2])].tolist(), "count",  count[index])
            print("\n\n\n\n")

        return trajectoryTransition, stateVisitationFrequency

    def findStateInStateTable(self, state, stateTable):
        return stateTable.loc[(stateTable['lightIsRed'] == (state[0])) & (stateTable['distanceToGoal'] == (state[1]))].index[0]


def main():
    x = DataFrameTrajectory()
    traj = Trajectory(x.dfs[3], x.stateTable)



if __name__ == '__main__':
    main()
