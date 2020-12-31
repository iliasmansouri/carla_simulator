import glob
import math
import os
import random
import sys
import time

import numpy as np

from sensors import CollisionSensor, RGBSensor
from utils import gaussian, get_distance, points_in_circle

try:
    sys.path.append(
        glob.glob(
            "../../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla


class World:
    def __init__(
        self,
        sensors_lst=["rgb", "collision"],
        img_size_x=320,
        img_size_y=240,
        img_channels=3,
        episode_length=50,
        destination_radius=50,
    ):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

        self.actor_list = []
        self.vehicle = None
        self.spawn_point = None

        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.img_channels = img_channels
        self.sensors_lst = sensors_lst
        self.sensors = None

        self.episode_start = 0
        self.episode_length = episode_length

        self.destination_radius = destination_radius
        self.action_space = [0, 1, 2, 3]

        self.throttle = 1
        self.steer_amount = 1

    def destroy_all(self):
        for actor in self.actor_list:
            actor.destroy()

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.img_size_x * self.img_size_y * self.img_channels

    def spawn_actors(self):
        self.spawn_vehicle()
        self.sensors = self.get_sensors(self.sensors_lst)

    def spawn_vehicle(self):
        vehicle = self.blueprint_library.filter("model3")[0]
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle, self.spawn_point)
        self.actor_list.append(self.vehicle)

    def init_rbg_sensor(self):
        return RGBSensor(self.vehicle, self.img_size_x, self.img_size_y)

    def init_collision_sensor(self):
        return CollisionSensor(self.vehicle)

    def get_sensors(self, sensors_lst):
        accessor = {
            "collision": self.init_collision_sensor,
            "rgb": self.init_rbg_sensor,
        }

        sensors = {}
        for s in sensors_lst:
            sensor = accessor.get(s, "Invalid sensor")()
            if sensor:
                sensors[s] = sensor
                self.actor_list.append(sensor)

        return sensors

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=-1 * self.steer_amount)
            )
        if action == 2:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=1 * self.steer_amount)
            )
        if action == 3:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0, steer=1 * self.steer_amount)
            )

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        collision_data = self.get_collision_data()
        image_data = self.get_image_data()

        done = False

        if len(collision_data) == 0:
            distance = get_distance(self.get_vehicle_location(), self.destination)
            if distance != 0:
                reward = int(100 * gaussian(distance))
            else:
                reward = 500
        else:
            done = True
            reward = -500

        if self.episode_start + self.episode_length < time.time():
            done = True

        return image_data, reward, done, None

    def reset(self):
        self.destroy_all()
        self.actor_list = []
        self.spawn_actors()

        while self.get_image_data() is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        self.destination = self.select_destination_point()
        self.world.debug.draw_point(
            self.destination,
            color=carla.Color(r=77, g=255, b=0, a=255),
            size=0.5,
            life_time=10,
        )

        return self.get_image_data()

    def get_collision_data(self):
        return self.sensors["collision"].get_collision_history()

    def get_image_data(self):
        return self.sensors["rgb"].get_image_data()

    def sample_action_space(self):
        return np.random.randint(0, 4)

    def get_vehicle_location(self):
        return self.vehicle.get_transform().location

    def select_destination_point(self):
        # TODO
        # select point as found in https://carla.readthedocs.io/en/latest/python_api/#carlalabelledpoint
        location = self.get_vehicle_location()
        points = list(points_in_circle(self.destination_radius, location.x, location.y))
        coords = random.choice(points)
        return carla.Location(x=coords[0], y=coords[1], z=2.5)

    def draw_point(self):
        debug = self.world.debug
        self.spawn_vehicle()

        time.sleep(0.01)
        location = self.get_vehicle_location()
        points = points_in_circle(25, location.x, location.y)

        for p in points:
            debug.draw_point(carla.Location(x=p[0], y=p[1], z=2.5))
