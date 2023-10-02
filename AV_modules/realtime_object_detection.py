import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import torch
import random
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image


def image_callback(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    model = YOLO('yolov8n.pt')
    results = model.predict(i3)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        # im.save("_out/%06d.png" % image.frame)
        im.show()  # show image


IM_WIDTH = 640
IM_HEIGHT = 480


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter("model3"))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        camera_bp.set_attribute('sensor_tick', '1')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda image: image_callback(image))

        time.sleep(50)

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()
