import numpy as np
import pandas as pd
import pandas as pd
from enum import IntEnum
import math


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class CarlaFeatures(object):
    def __init__(self) -> None:
        self.actions = [True, False]
        self.featuresNames = ["lightIsRed", "distanceToGoal","passedTrafficLight", "performedStop"]

        # Parameters needed for saving trajectories
        self.currentTrajectory = []
        self.currentTrajectoryNumber = 0

        # The number of trajectories observed in the demonstration
        self.maxTrajectories = 30
        self.initalDistance = None

        # Parameters used to create transitions
        self.states = pd.DataFrame(columns=self.featuresNames[:-1])
        self.transitions = dict()
        # Holds the current states from which we performed the action
        self.curStateSet = set()
        # Holds the states to which we transitioned.
        self.nextStateSet = set()

    def generateFeatures(self, trafficLightMsg, agent, destination, debug=False, initDistance=None):
        """
        The function used to extract the features from a timestep
        """
        # The two features [isRed,  distanceToGoal]
        features = [0, 0]

        # Set the isRedLight feature if traffic light is red.
        if trafficLightMsg == 1:
            features[0] = 1

        # Calculate percentage traveled for road section
        veh_loc = agent._vehicle.get_location()
        total = 0
        # Calculate distance to the destination
        if not self.initalDistance:
            lastwp = None
            for wp, _ in agent._global_planner.trace_route(initDistance, destination.location):
                if not lastwp:
                    lastwp = wp.transform.location                
                total += lastwp.distance(wp.transform.location)
                lastwp = wp.transform.location
            self.initalDistance = math.ceil(total)
            features[1] = 1.0
        else:
            lastwp = None
            for wp, _ in agent._global_planner.trace_route(veh_loc, destination.location):        
                if not lastwp:
                    lastwp = wp.transform.location                
                total += lastwp.distance(wp.transform.location)
                lastwp = wp.transform.location
            features[1] = round(total / self.initalDistance, 3)

        if debug:
            print("isRed", features[0], "distanceToGoal", features[1])

        return features

    def addStep(self, features, actionToPerform):
        """
        Add a timestep as features and performed action to the current (observing) trajectory
        """
        self.currentTrajectory.append((features, actionToPerform))

    def saveTrajectory(self, stepsToIgnore=3):
        """
        Saves the current trajectory as a csv file.

        stepsToIgnore:  Part of the trajectory which will not be saved.
                        By default the first and last 3 observations will be ignored.
        """
        df = pd.DataFrame(columns=self.featuresNames)
        for step in self.currentTrajectory[stepsToIgnore:-stepsToIgnore]:
            step[0].append(step[1])
            df.loc[len(df)] = pd.Series(step[0], index=self.featuresNames)
        df.to_csv('traj'+str(self.currentTrajectoryNumber)+'.csv', index=False)
        self.currentTrajectoryNumber += 1
        self.currentTrajectory = []
        if self.currentTrajectoryNumber == self.maxTrajectories:
            exit()

    def addTransition(self, features):
        """
        Add the transition to the transitions dictonary.
        """
        stateId = self.getStateNumber(features)
        if not stateId:
            # features are not known as a state yet
            stateId = self.addState(features)

        # Add state to the set of possible transition states.
        self.nextStateSet.add(stateId[0])

    def addState(self, features):
        """
        Add state to states dataframe. State is already in df then the state number will be returned
        """
        stateId = self.getStateNumber(features)
        if not stateId:
            # No state is known with the features.
            index = len(self.states)
            self.states.loc[index] = pd.Series(
                features, index=self.featuresNames[:-1])
            return [index]
        else:
            return stateId

    def saveTransition(self, action):
        # Save the observed transition.
        for i in list(self.curStateSet):
            if str(i)+"-"+str(action) in self.transitions:
                self.transitions[str(
                    i)+"-"+str(action)].update(self.nextStateSet)
            else:
                self.transitions[str(
                    i)+"-"+str(action)] = set(self.nextStateSet)

        # During the creation of the transition function first the brake actions are performed,
        # afterwards the no-brake actions are performed. In order to obtain the correct transition
        # After the no-brake actions are observed the obtained next state will be the starting states
        # for the next brake action transitions
        if not action:
            self.curStateSet = set(self.nextStateSet)
        self.nextStateSet.clear()

    def getStateNumber(self, features):
        """
        Returns the state number for the given state features.
        Will return an empty list if there does not exists a state with the features.
        """
        return self.states[(self.states[self.featuresNames[0]] == features[0])
                           & (self.states[self.featuresNames[1]] == features[1])].index.tolist()

    def getStateFeatures(self, stateNumber):
        features = self.states.loc[stateNumber, :].values.flatten().tolist()
        return features

    def saveStateAndTransitions(self):
        # Saves the transitions and states observed (including the state-features)
        self.states.to_csv("stateFeatures.csv", index=True)
        print(self.states)
        print("\n\n\n")
        df = pd.DataFrame([list(self.transitions.keys()), list(
            self.transitions.values())]).transpose()
        df.columns = ['state-action', 'transitions']
        df.to_csv('stateTransitions.csv', index=False)
        print(df)


class IRLReward(object):

    def __init__(self, featureweights):
        self.featureweights = np.array(featureweights)

        self.stateTable = self.loadStates()

        # The transition table for every state-action pair
        self.transitionTable = self.generateTransitionTable()

        # The probability of transitioning to state s for every state-action pair
        self.p_transition = self.generateProbTransition()

    def loadStates(self):
        df = pd.read_csv("stateFeatures.csv", index_col=0)
        return df

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

    def calculateStateActionValue(self, features, actionToPerform):
        currentState = self.findStateInStateTable(features)
        print("Current state",currentState)
        if not currentState:
            print("state not found based on features", features)
            exit()

        nextStates = set(self.transitionTable[str(
            currentState)+"-" + str(actionToPerform)])
        reward = 0
        while nextStates:
            state = nextStates.pop()
            p = self.p_transition[int(currentState),
                                  int(actionToPerform), state]
            reward += p * np.dot(self.findStateFeatures(state).to_numpy()
                           ,self.featureweights.T)
        return reward

    def findStateInStateTable(self, state):
        return self.stateTable.loc[(self.stateTable['lightIsRed'] == (state[0])) & (self.stateTable['distanceToGoal'] == (state[1]))].index[0]

    def findStateFeatures(self, stateNumber):
        return self.stateTable.iloc[stateNumber]

    def calculateRewardPerAction(self, startingState, actionList, debug=False):
        actionToTake = actionList[0]
        actionList = actionList[1:]
        reward = np.array([])
        reward = np.append(reward, self.calculateStateActionValue(startingState, actionToTake))

        for action in actionList:
            reward = np.append(reward, self.calculateStateActionValue(startingState, action))


        if debug:
                print("rewards", reward, "for actions",actionList)
        return reward
 

def main():
    demo = IRLReward([1, -1])
    actionList = [0,1]
    rewards =demo.calculateRewardPerAction([0, 0.843], actionList)
    print(rewards)
    print(actionList[np.argmax(rewards)])


if __name__ == '__main__':
    main()
