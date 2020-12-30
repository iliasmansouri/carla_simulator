from math import sqrt, pi
import numpy as np


def get_distance(point_1, point_2):
    return sqrt((point_1.x - point_2.x) ** 2 + (point_1.y - point_2.y) ** 2)


def points_in_circle(radius, pos_x=0, pos_y=0):
    x_ = np.arange(pos_x - radius - 1, pos_x + radius + 1, dtype=int)
    y_ = np.arange(pos_y - radius - 1, pos_y + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - pos_x) ** 2 + (y_ - pos_y) ** 2 <= radius ** 2)

    for x, y in zip(x_[x], y_[y]):
        yield x.astype(float), y.astype(float)


def gaussian(x, mu=0, sig=10):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
