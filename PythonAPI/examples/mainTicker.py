import carla
import trafficLoop
import argparse
from carlaFeatureHelper import CarlaFeatures, IRLReward
import numpy as np


def setup():
    # Setting up the client and the environment
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    sim_world = client.get_world()
    settings = sim_world.get_settings()

    # The environment will only perform one timestep when client.get_world().tick() is called.
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.12  # 0.12
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 16
    sim_world.apply_settings(settings)

    # Despawn graphical resources
    listOfBuildings = client.get_world().get_environment_objects(
        carla.CityObjectLabel.Buildings)
    if listOfBuildings:
        print("Despawning buildings")
        client.load_world(map_name="Town10HD_Opt",
                          reset_settings=False, map_layers=carla.MapLayer.NONE)

    # initiate traffic manager (which manages the traffic lights in the experiment)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(0.0)
    return client, traffic_manager


# Default Carla commandline args
def args():
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default='Constant')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()
    args.sync = True  # Sync by default
    args.loop = True  # Loop by default
    args.width, args.height = [int(x) for x in args.res.split('x')]

    return args


def main(scenario=0, case=0, drawObservationValues=False, drawStart=False, drawDestination=False, featureweights=None):
    world = None
    irlRewardClass = None
    if featureweights:
        irlRewardClass = IRLReward(featureweights)

    try:
        # Initializing CARLA client and server
        test = args()
        (client, traffic_manager) = setup()
        (_, world) = trafficLoop.game_loop(
            test, client, traffic_manager, scenario=scenario)

        client.get_world().tick()
        world.setup_sensors()

        (world, agent, destination) = trafficLoop.game_loop2(
            traffic_manager, world, scenario=scenario)

        # Configures the agents vehicle.
        configVehicle(agent)

        features = None

        if drawDestination:
            client.get_world().debug.draw_point(
                location=destination.location, size=0.2, color=carla.Color(0, 255, 0), life_time=10.0)
        client.get_world().tick()

        # Initialize the feature helper class
        carlaFeatureHelper = CarlaFeatures()

        # The initial spawn location of the vehicle. This is used to calculate the distance to the goal.
        initDistance = agent._vehicle.get_location()

        while True:
            if case == 1:
                carlaFeatureHelper.generateFeatures(
                    trafficLightMsg=0, agent=agent, destination=destination, initDistance=initDistance)

                trajectoryDone = False

                # Perform one trajectory run in order to obtain the correct baseline start location for the run thereafter.
                while not trajectoryDone:
                    trafficLightCode, trajectoryDone, brake = trafficLoop.tick_action(world,
                                                                                      agent, destination, traffic_manager, scenario=scenario, client=client)
                    client.get_world().tick()
                client.get_world().tick()
                # The number of unique transformations to be performed for each action.
                numberOfTransitions = 250
                startingTransformation = agent._vehicle.get_transform()
                isDone = False
                startStateFeatures = None

                # Continue performing steps until we reach the destination.
                while not isDone:

                    # Perform one of the two actions
                    for action in [True, False]:
                        # Run the transitions N times
                        for i in range(numberOfTransitions):
                            agent._vehicle.set_transform(
                                startingTransformation)
                            # Currently only two actions possible: stop or do not stop.
                            isDone, trafficLightCode = trafficLoop.tick_generate_transitions(world,
                                                                                             agent, destination, traffic_manager, brake=action)

                            # perform the action
                            client.get_world().tick()

                            # ignore first observation, which can be inconsistent
                            if i == 0:
                                continue

                            # Observe the outcome.
                            newStateFeatures = carlaFeatureHelper.generateFeatures(
                                trafficLightMsg=trafficLightCode, agent=agent, destination=destination)

                            # Add the transitions to the set of possible state to transition to
                            carlaFeatureHelper.addTransition(newStateFeatures)

                            print("From ", startStateFeatures, "To",
                                  newStateFeatures, "with action", int(action))

                        startStateFeatures = newStateFeatures
                        # Save the transition
                        carlaFeatureHelper.saveTransition(int(action))
                    startingTransformation = agent._vehicle.get_transform()
                carlaFeatureHelper.saveStateAndTransitions()
                exit()

            elif case == 2:
                if not irlRewardClass:
                    print("Feature weights not provided")
                    exit()

                isDone = False
                possibleActions = [0, 1]
                actionToTake = possibleActions[0]

                # Init generateFeatures function
                carlaFeatureHelper.generateFeatures(
                    trafficLightMsg=0, agent=agent, destination=destination, initDistance=initDistance)

                client.get_world().tick()
                client.get_world().tick()
                # for _ in range(2):
                #     client.get_world().tick()

                while not isDone:
                    isDone, trafficLightCode = trafficLoop.tick_irl_agent(world,
                                                                          agent, destination, traffic_manager, brake=actionToTake)

                    client.get_world().tick()
                    # client.get_world().tick()

                    currentObservation = carlaFeatureHelper.generateFeatures(
                        trafficLightMsg=trafficLightCode, agent=agent, destination=destination)
                    if currentObservation[0] == 1:
                        # Light is observed RED
                        print("traffic light is red")

                    # exit when agent is within 5 % of the goal.
                    if (currentObservation[1] < 0.05):
                        exit()
                    rewardPerAction = irlRewardClass.calculateRewardPerAction(startingState=currentObservation,
                                                                              actionList=possibleActions)
                    print("rewardPerAction", rewardPerAction)
                    drawObservation(client, agent, currentObservation,
                                    rewardPerAction, possibleActions, case)
                    actionToTake = possibleActions[np.argmax(rewardPerAction)]
                    # print("actionToTake", actionToTake)

            elif case == 3:
                # Run the expert agent and observe the trajectories
                carlaFeatureHelper.generateFeatures(
                    trafficLightMsg=0, agent=agent, destination=destination, initDistance=initDistance)

                trafficLightCode, trajectoryDone, brake = trafficLoop.tick_action(world,
                                                                                  agent, destination, traffic_manager, scenario=scenario, client=client)
                if trajectoryDone:
                    carlaFeatureHelper.saveTrajectory()
                else:
                    features = carlaFeatureHelper.generateFeatures(
                        trafficLightMsg=trafficLightCode, agent=agent, destination=destination)
                    drawObservation(client, agent, features,
                                    None, None, case)
                    carlaFeatureHelper.addStep(
                        features=features, actionToPerform=brake)

                client.get_world().tick()
            else:
                exit()
    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()


def drawObservation(client, agent, currentObservation, rewardPerAction, possibleActions, case):
    veh_loc = agent._vehicle.get_location()
    # print(currentObservation)
    actionReward = ""

    if case == 1:
        # Draw observed feature values above the vehicle
        client.get_world().debug.draw_string(location=carla.Location(veh_loc.x+2.5, veh_loc.y, veh_loc.z + 2),
                                             text="Transitions\nobserved state features - " + str(currentObservation[0]) + " | " + str(currentObservation[1]), color=carla.Color(0, 0, 0, 0), draw_shadow=True, life_time=.02)

    if case == 2:
        for reward, action in zip(rewardPerAction, possibleActions):
            actionReward += str(round(action, 3)) + \
                " = " + str(round(reward, 3))
            actionReward += " |\n"
        # Draw observed feature values above the vehicle
        client.get_world().debug.draw_string(location=carla.Location(veh_loc.x+2.5, veh_loc.y, veh_loc.z + 2),
                                             text="IRL Learner\nobserved state features - " + str(currentObservation[0]) + " | " + str(currentObservation[1]) + "\nactionReward-\n"+actionReward, color=carla.Color(0, 0, 0, 0), draw_shadow=True, life_time=.02)

    if case == 3:
        # Draw observed feature values above the vehicle
        client.get_world().debug.draw_string(location=carla.Location(veh_loc.x+2.5, veh_loc.y, veh_loc.z + 2),
                                             text="Expert\nobserved state features - " + str(currentObservation[0]) + " | " + str(currentObservation[1]), color=carla.Color(0, 0, 0, 0), draw_shadow=True, life_time=.02)


def attachSpectatorToAgent(spectator, agent):
    agent_transform = agent._vehicle.get_transform()
    agent_transform.location.z += 4
    spec_transform = spectator.get_transform()
    print(spec_transform)
    abs_vector = abs(agent_transform.rotation.get_forward_vector())
    print(abs_vector)
    changing_axis = 0
    if (abs_vector.y > abs_vector.x):
        changing_axis = 1

    if (changing_axis):
        spec_transform.location.y = agent_transform.location.y
    else:
        spec_transform.location.x = agent_transform.location.x
    # Get directation that the vehicle is moving
    # print(agent_transform.rotation.get_right_vector())
    print(agent_transform.location)
    print("\n\n\n")
    spectator.set_transform(spec_transform)


def configVehicle(agent):
    # Set frication of wheels to 0.0 for the expert agent,
    physics_control = agent._vehicle.get_physics_control()
    wheels = []
    for wheel in physics_control.wheels:
        wheelToAdd = carla.WheelPhysicsControl(
            tire_friction=0.0, damping_rate=wheel.damping_rate, max_steer_angle=wheel.max_steer_angle, radius=wheel.radius,
            max_brake_torque=wheel.max_brake_torque, max_handbrake_torque=wheel.max_handbrake_torque, position=wheel.position,
            long_stiff_value=wheel.long_stiff_value, lat_stiff_max_load=wheel.lat_stiff_max_load, lat_stiff_value=wheel.lat_stiff_value)
        wheels.append(wheelToAdd)

    physics_control.wheels = wheels
    agent._vehicle.apply_physics_control(physics_control)


if __name__ == '__main__':
    """
    CASE:
    * 1 = Generate state transition function for the environment
    * 2 = Run learner agent (with reward function)
    * 3 = Observe demonstration performed by the expert agent.

    scenario:
    * 0 = agent acts on the observed trajectory states.
    * 1 = agent is spawn on the unobserved trajectory states.
    """
    # [0.14229569, -0.8753701]
    # [0.26073989,-0.46541897]
    # [ 0.41621339 -0.47571724]
    main(scenario=0, case=2, featureweights= [ 0.41621339,-0.47571724])
