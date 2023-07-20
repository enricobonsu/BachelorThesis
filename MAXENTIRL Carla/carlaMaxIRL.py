import optimizer as O
import maxentCarla as M
from carlaDemonstration import Demonstration
import matplotlib.pyplot as plt


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
    init = O.Constant(0.2)

    # choose our optimization strategy:
    # we select exponentiated gradient descent with linear learning-rate decay
    optim = O.Sga(lr=O.exponential_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward, delta_list, theta_list = M.irl(p_transition,stateToFeatures,
                   terminalStates, trajectories, optim, init)

    # Print the reward obtain for reaching any of the states
    for index, i in enumerate(reward):
        print(index, i)
    return reward, delta_list, theta_list


def main():
    demonstration = Demonstration()
    rewardFunction, deltas, thetas = maxent(demonstration)

    # Calculate the state-action values
    for key, value in demonstration.transitionTable.items():
        reward = 0.0
        beginState, action = key.split("-")
        while value:
            resultingState = value.pop()
            reward += (rewardFunction[resultingState] * demonstration.p_transition[int(beginState),int(action),int(resultingState)])
        print(key, reward)

    plt.plot(deltas, 'ro',markersize=3)
    plt.title("Difference in the feature weight vector after each optimization step")
    plt.xlabel('Number of optimization steps') 
    plt.ylabel('Delta between old and new feature weights') 
    plt.xticks(range(0,50,5))
    plt.savefig("delta.png")
    plt.show()
    
    plt.plot(thetas[0], 'bo',markersize=3)
    plt.title("Value of feature \'an observed traffic light is red\' during optimization")
    plt.xlabel('Number of optimization steps')
    plt.yticks(range(10,-50,-10))
    plt.xticks(range(0,50,5))
    plt.savefig("isRed.png")
    plt.show()

    plt.plot(thetas[1], 'bo',markersize=3)
    plt.title("Value of feature \'distance from the destination\' during optimization")
    plt.xlabel('Number of optimization steps') 
    plt.xticks(range(0,50,5))
    plt.savefig("distanceTop.png")
    plt.show()
    


if __name__ == '__main__':
    main()
