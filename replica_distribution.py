"""
This module works under the assumption that the network is connected such that distance in the given
metric space approximates the latency required to retrive the record.


"""
import random
import math
#from typing import Iterable


class h2dPoint(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def random(self):
        r = random.random()**0.9
        theta = random.random() * math.pi * 2
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        return h2dPoint(x, y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def eDist(self, other):
        return ((self.x - other.x)**2.0 + (self.y - other.y)**2.0)**0.5

    def dist(self, other):

        sigma = 2 * self.eDist(other)**2.0 / ((1 - self.dot(self)) * (1 - other.dot(other)))
        return math.acosh(1 + sigma)


class e2dPoint(object):

    @classmethod
    def random(self):
        return e2dPoint([random.random() for x in range(2)])

    def __init__(self, vec):
        self.vec = tuple(vec)

    def dist(self, other):
        return sum(map(lambda x, y: (x - y) * (x - y), self.vec, other.vec))**0.5


class Universe(object):

    def __init__(self, metric=e2dPoint):
        self.records = {}
        self.metric = metric

    def add_record(self, r_id):
        self.records[r_id] = [self.metric.random()]

    def replicate(self, r_id, times=1):
        for i in range(times):
            self.records[r_id].append(self.metric.random())

    def get_closest(self, origin, r_id):
        return min(self.records[r_id], key=lambda x: origin.dist(x))


def Exponential_dist():
    u = Universe(h2dPoint)
    max_dist = h2dPoint(0, 0.9).dist(h2dPoint(0, 0))
    print(max_dist)
    records = list(range(1000))
    for r in records:
        u.add_record(r)
        u.replicate(r, r // 2)

    import matplotlib.pyplot as plt
    distances = []
    for r in records:
        dist = 0
        for i in range(100):
            p = u.metric.random()
            dist += p.dist(u.get_closest(p, r))
        distances.append(dist / 100)
    plt.plot(records, distances)
    plt.show()


if __name__ == "__main__":
    Exponential_dist()
