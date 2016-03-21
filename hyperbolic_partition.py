import random
import math


def point():
    dim = 2

    r = random.random()**0.5
    theta = random.random() * math.pi * 2
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    return (x, y)


def fEq(a, b):
    return abs(a - b) <= 0.000001


def dot(a, b):
    return sum(map(lambda x, y: x * y, a, b))


def cross(a):
    # assumes 2d
    return (-1 * a[1], a[0])


def klein2poincare(s):
    assert(dot(s, s) < 1)
    return tuple(map(lambda x: x / (1 + (1 - dot(s, s))**0.5), s))


def poincare2klein(u):
    assert(dot(u, u) < 1)
    return tuple(map(lambda x: (x * 2) / (1 + dot(u, u)), u))


def eMid(a, b):
    return tuple(map(lambda x, y: (x + y) / 2, a, b))


def eDist(a, b):
    return sum(map(lambda x, y: (x - y)**2.0, a, b))**0.5


def hDist(a, b):

    sigma = 2 * eDist(a, b)**2.0 / ((1 - dot(a, a)) * (1 - dot(b, b)))
    return math.acosh(1 + sigma)

K = 10
pop = 1000
population = [point() for x in range(pop)]

samples = 10000

sample_pairs = []

for i in range(samples):
    target = random.choice(population)
    replicas = sorted(population, key=lambda x: hDist(target, x))[:K]

    source = point()
    sink = point()
    pop_partition_size = len(list(filter(lambda x: hDist(source, x) < hDist(sink, x), population)))
    replica_partition_size = len(
        list(filter(lambda x: hDist(source, x) < hDist(sink, x), replicas)))
    replica_loss = 0.0
    if replica_partition_size > 0:
        replica_loss = 1.0
    sample_pairs.append((pop_partition_size / pop, replica_loss))

import matplotlib.pyplot as plt

ratios = [0.1 * x for x in range(11)]
ratio_samples = {x: [] for x in range(11)}
for r, v in sample_pairs:
    r = int(r * 10)

    ratio_samples[r].append(v)

ratio_averages = [sum(ratio_samples[x]) / len(ratio_samples[x]) for x in range(11)]

plt.plot(ratios, ratio_averages)
plt.show()
