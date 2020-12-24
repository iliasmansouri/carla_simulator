import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time

def main():
    actor_list = []
    try:
        pass
    finally:
        print('Cleaning...')
        for actor in actor_list:
            actor.destroy()
        print('Done')

if __name__ == "__main__":
    main()