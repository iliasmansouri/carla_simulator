import glob
import os
import sys
import random

from sensors import CollisionSensor, RGBSensor

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
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.actor_list = []

        self.vehicle = self.blueprint_library.filter("model3")[0]
        self.agent = self.world.spawn_actor(self.vehicle, self.spawn_point)
        self.actor_list.append(self.agent)

        self.sensors = self.get_sensors(["rgb", "collision"])

        self.destroy_all()

    def destroy_all(self):
        for actor in self.actor_list:
            actor.destroy()

    def get_sensors(self, sensors_lst):
        sensors = {
            "collision": CollisionSensor(self.agent),
            "rgb": RGBSensor(self.agent),
        }

        for s in sensors_lst:
            sensor = sensors.get(s, "Invalid sensor")
            if sensor:
                self.actor_list.append(sensor)

        return sensors

    def step(self, action):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    world = World()
