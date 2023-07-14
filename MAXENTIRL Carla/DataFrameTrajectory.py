
import numpy as np
import pandas as pd
import glob


class DataFrameTrajectory(object):

    def __init__(self, prefix="traj", path=None):
        files = self.findTrajectories(prefix, path)

        def retrieveCsvNumber(fileName):
            """
            Retrieves the trajectory number from the csv file name
            """
            x = fileName.split(".")[0]
            x = x[4:]
            return int(x)

        files.sort(key=lambda x: retrieveCsvNumber(x))

        # dfs contains all trajectories, stored as a list of dfs
        self.dfs = []
        for file in files:
            self.dfs.append(self.csvToDf(file))
 
        self.stateTable = self.generateStateTable()


    def generateStateTable(self):
        """
        Read the state-feature data
        """
        df = pd.read_csv("stateFeatures.csv",index_col=0)
        return df


    def findTrajectories(self, prefix="traj", path=None):
        """
        Finds all csv files for a given prefix either in the current folder, or a provided path folder. 
        """
        path = prefix + r'*.csv'
        files = glob.glob(path)
        files.pop(0) # The first trajectory is always inconsistent, and is thus removed
        return files

    def csvToDf(self, file):
        df = pd.read_csv(file)

        columns_titles = ["lightIsRed", "distanceToGoal", "performedStop"]
        df = df.reindex(columns=columns_titles) # make the action (stop) the last action

        return df


def main():
    traj = DataFrameTrajectory(prefix="traj")
    print(traj.dfs[5])
    print(traj.stateTable)


if __name__ == '__main__':
    main()
