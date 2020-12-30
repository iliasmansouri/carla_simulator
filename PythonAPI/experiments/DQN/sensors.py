import collections
import glob
import math
import os
import sys
import weakref

import cv2
import numpy as np
from PIL import Image

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


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


class CollisionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class RGBSensor:
    def __init__(self, parent_actor, img_size_x=640, img_size_y=480, fov=110):
        self.sensor = None
        self.img_data = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", f"{img_size_x}")
        bp.set_attribute("image_size_y", f"{img_size_y}")
        bp.set_attribute("fov", f"{fov}")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: self._on_image(data))

    def process_img(self, image):
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        img_bgra = i.reshape((image.height, image.width, 4))
        b_channel, g_channel, r_channel, _ = cv2.split(img_bgra)
        img = cv2.merge((b_channel, g_channel, r_channel))
        return img

    def _on_image(self, image):
        self.img_data = self.process_img(image)

    def get_image_data(self):
        return self.img_data

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()