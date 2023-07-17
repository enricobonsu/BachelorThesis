
# BachelorThesis
This repository contains all the code required for the experiments performed in the bachelor thesis by Enrico Bonsu.

This repository contains two directories. The directory "MAXENTIRL Carla" contains the code used to perform the maximum entropy inverse reinforcement learning (MAXENTIRL) algorithm. The second directory "PythonAPI" contains the code used for performing the experiments in the CARLA simulator software [[1]](#1). 
Python version 3.7 is used for both implementations.

## MAXENTIRL Carla
This folder contains the code to perform the MAXENTIRL algorithm. The MAXENTIRL algorithm was  first published by Ziebert et al. (2008) [[2]](#2). 
The repository by Maximilian Luz [[3]](#3) was used as inspiration and starting point for the implementation.

### Files
* `DataFrameTrajectory.py`
Reads the trajectories and the state features CSV files.
* `carlaTrajectory.py`
Extract the start and termial states from each trajectory and generates the trajectory transitions for each trajectory. This will also count the state visitation frequency for each state per trajectory.
* `carlaDemonstration.py`
Generates the state transition probability for every state-action pair using the `stateTransitions.csv` as possible transitions. For the experiments manual transition probabilities are set for the non deterministic states.
* `carlaMaxIRL.py`
Contains the setup for the MAXENTIRL algorithm by providing the initial feature weights and the type of gradient-ascent optimizer used during the algorithm.
* `optimizer.py`
The optimizer as provided by Maximilian Luz [[2]](#2) which contains generic stochastic gradient-ascent based optimizers.
* `maxentCarla.py`
A modified version of the `maxent.py` file provided by Maximilian Luz [[2]](#2) which performs MAXENTIRL algorithm.

### How to use
The directory contains multiple trajectories as a CSV file with the prefix 'traj'.
Each row in the file is a unique time step.

To run the algorithm 
```
py -3.7 carlaMaxIRL.py
```
The algorithm will return the final feature weights and the reward for every state and state-action pair.
The feature weights can then be used as input for the experiments.

## PythonAPI
The PythonAPI directory contains the (modified) files that are needed to perform the experiments in CARLA.
To run the experiment CARLA version 0.9.14 must be installed.
Once installed the folders contained in `PythonAPI/carla` must be placed in the installed carla directory `../PythonAPI/carla/agents`. 
The files contained in `PythonAPI/examples` must be placed in the installed carla directory `../PythonAPI/examples`. 

### Files
Many files are a slightly modified versions of the original provided by the CARLA software. However there are some additional files added.
* `carlaFeatureHelper.py`
This file is used to generate states and their appropriate features. This files is also store and save trajectories, the state with their feature and state transitions.
This file also contains the IRLReward class which is used to calculate the reward for the IRL agent. 

* `mainTicker.py`
This the main file used during the experiment. The main function within this file must be provided the appropriate feature weights obtained from the MAXENTIRL algorithm. In addition one of two scenarios can be chosen to be performed by the agent.

### How to use

To run the experiment 
```
py -3.7 mainTicker.py
```
The `mainTicker.py` file contains a main function which parameters can be adjusted to perform different tasks.


## Videos
* Expert agent demonstration traffic light red (scenario 1): https://youtu.be/WH3uXHikUok
* Expert agent demonstration traffic light red (scenario 1): https://youtu.be/-OwEiZlexac

* IRL Agent crossing intersection with red light (scenario 1): https://youtu.be/iQQzLxn280U
* IRL Agent crossing intersection with green light (scenario 1): https://youtu.be/Vh74B2-Wcdk

* IRL Agent crossing intersection with red light (scenario 2): https://youtu.be/7_WyDZzWzdE
## References
<a id="1">[1]</a> 
https://github.com/carla-simulator/carla

<a id="2">[2]</a> 
Ziebart, Brian & Maas, Andrew & Bagnell, J. & Dey, Anind. (2008). Maximum Entropy Inverse Reinforcement Learning. 1433-1438. 

<a id="3">[3]</a> 
https://github.com/qzed/irl-maxent


