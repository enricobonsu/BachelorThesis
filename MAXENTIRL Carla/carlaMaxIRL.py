import optimizer as O
import maxentCarla as M
from carlaDemonstration import Demonstration


def maxent(demonstration):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """

    terminalStates = demonstration.terminalStates
    p_transition = demonstration.p_transition
     # terminal will always end into itself

    trajectories = demonstration.trajectories
    stateToFeatures = demonstration.stateTable
    
    # choose our parameter initialization strategy:
    # initialize parameters with constant
    init = O.Constant(0.1)

    # choose our optimization strategy:
    # we select exponentiated gradient descent with linear learning-rate decay
    optim = O.Sga(lr=O.exponential_decay(lr0=0.001))

    # actually do some inverse reinforcement learning
    reward = M.irl(p_transition,stateToFeatures,
                   terminalStates, trajectories, optim, init)

    # Print the reward obtain for reaching any of the states
    for index, i in enumerate(reward):
        print(index, i)
    return reward


def main():
    demonstration = Demonstration()
    rewardFunction = maxent(demonstration)

    # Calculate the state-action values
    for key, value in demonstration.transitionTable.items():
        reward = 0.0
        beginState, action = key.split("-")
        while value:
            resultingState = value.pop()
            reward += (rewardFunction[resultingState] * demonstration.p_transition[int(beginState),int(action),int(resultingState)])
        print(key, reward)


if __name__ == '__main__':
    main()
