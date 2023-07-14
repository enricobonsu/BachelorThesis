import carla
import math
import numpy as np
import pandas as pd
import misc
import time


class RecordData(object):

    def __init__(self, client):
        # --------------
        # Spawn ego vehicle
        # --------------
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.actor_list = self.world.get_actors()

        self.vehicle = self.actor_list.filter('vehicle.*')[0]
        # self.vehicle.set_simulate_physics(False)
        self.lights_list = self.actor_list.filter("*traffic_light*")
        self.bbs = self.world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        self.debugger = self.world.debug
        self._last_traffic_light = None
        self.demo = []
        self.traj = []

        # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location
        self._lights_map = {}
        self.lastLoc = None

    # Draws waypoints that are the triggers for the traffic lights.
    def drawTrafficLightTriggers(self, seconds=1):
        for light in self.lights_list:
            for point in light.get_affected_lane_waypoints():
                begin = carla.Location(
                    point.transform.location.x, point.transform.location.y, 0)
                end = carla.Location(point.transform.location.x,
                                     point.transform.location.y, 50)
                # life_time = 0 is permanent.
                self.debugger.draw_line(begin, end, thickness=0.1,
                                        life_time=seconds, color=carla.Color(0, 255, 0, 0))

    #  Draws forward vector line of the vehicle.
    def forwardVectorLine(self, seconds=1):
        loc = self.vehicle.get_location()
        begin = carla.Location(loc.x, loc.y, loc.z +
                               self.vehicle.bounding_box.extent.z*2)
        end = carla.Location(loc.x + self.vehicle.get_transform().get_forward_vector().x * 3, loc.y +
                             self.vehicle.get_transform().get_forward_vector().y * 3, loc.z + self.vehicle.bounding_box.extent.z * 2)
        # life_time = 0 is permanent.
        self.debugger.draw_line(begin, end, thickness=0.1, life_time=seconds)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=0.0):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if not lights_list:
            lights_list = self.world.get_actors().filter("*traffic_light*")

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (1, self._last_traffic_light)

        ego_vehicle_location = self.vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = misc.get_trafficlight_trigger_location(
                    traffic_light)
                trigger_wp = self.map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue
            # print("distance",trigger_wp.transform.location.distance(ego_vehicle_location))
            # print("trigger_wp.road_id", trigger_wp.road_id,
            #       "ego_vehicle_waypoint.road_id", ego_vehicle_waypoint.road_id)

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue
            # print("distance",trigger_wp.transform.location.distance(ego_vehicle_location))
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            # print("TEST")

            if dot_ve_wp < 0:
                return (2, None)

            if traffic_light.state != carla.TrafficLightState.Red:
                if misc.is_within_distance(trigger_wp.transform, self.vehicle.get_transform(), max_distance, [-0.1, 90.0]):
                    return (3, None)
                continue
           #  dot negative == car is between 90 and 270 degrees from the light trigger wp, thus faced backwards.
            if misc.is_within_distance(trigger_wp.transform, self.vehicle.get_transform(), max_distance, [-0.1, 90.0]):
                self._last_traffic_light = traffic_light
                return (1, traffic_light)

        return (0, None)

    def recordPassingLightDemonstration(self, goal_loc=None, record=True, creatingTransitions = True):
        start = self.map.get_waypoint(
            location=carla.Location(x=-45.0, y=78.0, z=0.0)).transform
        start.location.z += 0.1
        end = goal_loc

        returnCode, _ = self._affected_by_traffic_light(
            max_distance=2.0 + 0.3 * 30)


        isInDistance = 0
        isRedLight = 0
        passedIntersection = 0
        stop = 0
        if (returnCode == 0):
            isInDistance = 0
            isRedLight = 0
            passedIntersection = 0
            stop = 1 if self.vehicle.get_velocity().length() < 1.0 else 0
        elif (returnCode == 1):
            isInDistance = 1
            isRedLight = 1
            passedIntersection = 0
            stop = 1 if self.vehicle.get_velocity().length() < 1.0 else 0
        elif (returnCode == 2):
            isInDistance = 0
            isRedLight = 0
            passedIntersection = 1
            stop = 1 if self.vehicle.get_velocity().length() < 1.0 else 0
        elif (returnCode == 3):
            isInDistance = 1
            isRedLight = 0
            passedIntersection = 0
            stop = 1 if self.vehicle.get_velocity().length() < 1.0 else 0

        if record:
            self.traj.append([isInDistance, isRedLight, passedIntersection, stop,
                          end.location.distance(self.vehicle.get_location())])
            if passedIntersection == 1:
                if (len(self.traj) < 30):
                    self.traj.clear()
                    passedIntersection = 0
                else:

                    self.demo.append(self.traj.copy())
                    print("traj added")
                    self.traj.clear()

                    if (len(self.demo) == 30):
                        isInDistances = []
                        isRedLights = []
                        passedIntersections = []
                        stops = []
                        distances = []
                        for count, trajectory in enumerate(self.demo):
                            isInDistances = [item[0] for item in trajectory]
                            isRedLights = [item[1] for item in trajectory]
                            passedIntersections = [item[2]
                                                for item in trajectory]
                            stops = [item[3] for item in trajectory]
                            distances = [item[4] for item in trajectory]

                            pd.DataFrame({"isInDistance": isInDistances, "isRedLight": isRedLights, "passedIntersection": passedIntersections,
                                        "stop": stops, "distanceToGoal": distances}).to_csv('traj'+str(count) + '.csv', index=False)
                    else:
                        print("len(demo)", len(self.demo))
        
        if creatingTransitions:
            return [isInDistance, isRedLight, passedIntersection, round(end.location.distance(self.vehicle.get_location()))]
        return [isInDistance, isRedLight, passedIntersection, stop, round(end.location.distance(self.vehicle.get_location()))]
