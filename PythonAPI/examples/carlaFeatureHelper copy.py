import carla
import trafficLoop
import argparse
from script import RecordData
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
        self.featuresNames = ["inDistance", "lightIsRed", "onNewRoad",
                              "percentagedTraveled", "distanceToGoal", "performedStop"]

        # Parameters needed for saving trajectories
        self.currentTrajectory = []
        self.currentTrajectoryNumber = 0
        self.maxTrajectories = 30

        # Parameters needed for feature generation
        self.intersectionWP = None
        self.roadEnd = None
        self.roadBegin = None
        self.currrentRoadId = None

        # Parameters used to create transitions
        self.states = pd.DataFrame(columns=self.featuresNames[:-1])
        self.transitions = dict()
        # Holds the current states from which we performed the action
        self.curStateSet = set()
        # Holds the states to which we transitioned.
        self.nextStateSet = set()

    def generateFeatures(self, trafficLightMsg, agent, world, topology, destination, debug=True):
        """
        trafficLightMsg contains the observed trafficlight state with possible values:
            - 1 = vehicle is in distance of trafficlight and the trafficlight is red
            - 2 = vehicle is in distance of trafficlight and the trafficlight is not red
            - 0 = Not in distance of a traffic light

        """
        totalTraveled = 0
        # inDistance, isRed, onNewRoad, percentageCrossed, distanceToGoal
        features = [0, 0, 0, 0, 0]

        # set correct Traffic light values
        if trafficLightMsg == 1:
            features[0] = 1
            features[1] = 1
        elif trafficLightMsg == 2:
            features[0] = 1

        # Calculate percentage traveled for road section
        veh_loc = agent._vehicle.get_location()
        currentWaypoint = world.map.get_waypoint(location=veh_loc)

        # Calculate distance to the destination
        features[4] = math.ceil(destination.location.distance(veh_loc))
        
        # Find endpoint in intersection
        if currentWaypoint.get_junction():
            # Set begin and end road point for intersections if not yet set
            if self.intersectionWP is None:
                for tup in currentWaypoint.get_junction().get_waypoints(carla.LaneType.Driving):
                    for wp in agent._local_planner._waypoints_queue:
                        if tup[1].transform.location.distance(wp[0].transform.location) < 1.0:
                            self.roadEnd = tup[1]
                            self.roadBegin = tup[0]
                            self.intersectionWP = True
            else:
                totalRoadLength = self.roadEnd.transform.location.distance(
                    self.roadBegin.transform.location)

                totalTraveled = self.roadEnd.transform.location.distance(
                    veh_loc) / totalRoadLength * 100
        else:
            if self.intersectionWP:
                # vehicle was on an intersection, now it moved passed it
                self.intersectionWP = None
            road = [item for item in topology if item[0].road_id ==
                    currentWaypoint.road_id and item[0].lane_id == currentWaypoint.lane_id]
            self.roadBegin, self.roadEnd = road[0]
            
            totalRoadLength = self.roadEnd.transform.location.distance(
                self.roadBegin.transform.location)

            totalTraveled = self.roadEnd.transform.location.distance(
                veh_loc) / totalRoadLength * 100

        features[3] = math.ceil(totalTraveled)
        if self.currrentRoadId is None:
            self.currrentRoadId = currentWaypoint.road_id

        elif self.currrentRoadId != currentWaypoint.road_id and totalTraveled > 50:
            features[2] = 1
            self.currrentRoadId = currentWaypoint.road_id

        if debug:
            print("inDistance", features[0], "isRed", features[1], "onNewRoad",
                  features[2], "percentageCrossed",
                  features[3], "distanceToGoal", features[4])

        return features

    def addStep(self, features, actionToPerform):
        self.currentTrajectory.append((features, actionToPerform))

    def saveTrajectory(self, stepsToIgnore=3):
        """
        stepsToIgnore:  Part of the trajectory which will not be saved.
                        By default the first and last 3 observations will be ignored.
        """
        df = pd.DataFrame(columns=self.featuresNames)
        for step in self.currentTrajectory[stepsToIgnore:-stepsToIgnore]:
            step[0].append(step[1])
            df.loc[len(df)] = pd.Series(step[0], index=self.featuresNames)
        print(df)
        df.to_csv('traj'+str(self.currentTrajectoryNumber)+'.csv', index=False)
        self.currentTrajectoryNumber +=1
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
        tempRemoved = set()
        print("before")
        print("self.curStateSet", self.curStateSet)
        print("self.nextStateSet", self.nextStateSet)
        print("")
        if action:
            # When we perfrom the stop actions, every observed state can transition to any other observed state.
            # It is not possible to transition to a state with onNewRoad.
            for state in list(self.nextStateSet):
                if self.states.iloc[state]['onNewRoad'] == 1:
                    self.nextStateSet.remove(state)
                    tempRemoved.add(state)
        else:
            # The agent did not stop, and this caused a road transition
            for state in list(self.nextStateSet):
                if self.states.iloc[state]['onNewRoad'] == 1:
                    for state in list(self.nextStateSet):
                        features = self.getStateFeatures(state)
                        if features[2] == 0:
                            self.nextStateSet.remove(state)
                            tempRemoved.add(state)
                        features[2] = 1
                        self.addTransition(features)
                    break
        print("after")
        print("self.curStateSet", self.curStateSet)
        print("self.nextStateSet", self.nextStateSet)
        print("")
        for i in list(self.curStateSet):
            if str(i)+"-"+str(action) in self.transitions:
                self.transitions[str(
                    i)+"-"+str(action)].update(self.nextStateSet)
            else:
                self.transitions[str(
                    i)+"-"+str(action)] = set(self.nextStateSet)
        print(self.transitions)
        print(self.states)
        
        # It is assumed that the transitionable states are now the states for which we
        # again look for the transitions.
        if not action:
            self.nextStateSet.update(tempRemoved)
            self.curStateSet = set(self.nextStateSet)
        else:
            if tempRemoved:
                print("tempRemoved is", tempRemoved)
                self.curStateSet.clear()
                self.curStateSet.update(tempRemoved)
                print("new self.curStateSet", self.curStateSet )
        self.nextStateSet.clear()

    def getStateNumber(self, features):
        """
        Returns the state number for the given state features.
        Will return an empty list if there does not exists a state with the features.
        """
        return self.states[(self.states[self.featuresNames[0]] == features[0])
                           & (self.states[self.featuresNames[1]] == features[1])
                           & (self.states[self.featuresNames[2]] == features[2])
                           & (self.states[self.featuresNames[3]] == features[3])
                           & (self.states[self.featuresNames[4]] == features[4])].index.tolist()
    
    def getStateFeatures(self, stateNumber):
        features = self.states.loc[stateNumber, :].values.flatten().tolist()
        return features

    def saveStateAndTransitions(self):
        self.states.to_csv("stateFeatures.csv", index=True)
        print(self.states)
        print("\n\n\n")
        df = pd.DataFrame([list(self.transitions.keys()), list(
            self.transitions.values())]).transpose()
        df.columns = ['state-action', 'transitions']
        df.to_csv('stateTransitions.csv', index=False)
        print(df)
