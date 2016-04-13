
"""
equation for hyperbolic plain:

z^2 = x^2+y^2+1

gradient = (2x,2y,-2z)

encoding for plane:

f(x,y,z) = A*x+B*y+C*z+D = 0
or (A,B,C,D)
"""

import numpy as np


def get_tangent_plane(p):
    norm = (2 * p[0], 2 * p[1], -2 * p[2])
    norm_mag = sum(map(lambda x: x * x, norm))**0.5
    A = norm[0] / norm_mag
    B = norm[1] / norm_mag
    C = norm[2] / norm_mag
    D = A * p[0] + B * p[1] + C * p[2]
    return (A, B, C, -1 * D)


def closest_point_on_plane(p, plane):
    A, B, C, D = plane
    x0, y0, z0 = p
    distance = (A * x0 + B * y0 + C * z0 + D) / (A * A + B * B + C * C)
    x1 = x0 + A * distance
    y1 = y0 + B * distance
    z1 = z0 + C * distance
    return (x1, y1, z1)


def get2dCoords(origin, plane, point):
    y_axis = (0, 0, 1)
