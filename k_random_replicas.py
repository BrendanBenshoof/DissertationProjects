import random
import matplotlib.pyplot as plt


class Point(object):

    def __init__(self):
        self.x = random.random()
        self.y = random.random()

    def dist(self, other):
        return ((self.x - other.x)**2.0 + (self.y - other.y)**2.0)**0.5

    def __hash__(self):
        return hash(self.x) & hash(self.y)


def simulate_churn(net_size, k_replicas, fail_rate_per_tick):
    nodes = set(map(lambda x: Point(), range(net_size)))
    replicas = random.sample(nodes, k_replicas)
    survivor_dist = []
    for i in range(0, 100):
        nodes = set(filter(lambda x: random.random() > fail_rate_per_tick, nodes))
        replicas = filter(lambda x: x in nodes, replicas)
        min_dist = 1.4
        p = Point()
        for r in replicas:
            d = r.dist(p)
            if d < min_dist:
                min_dist = d
        survivor_dist.append(min_dist)
    #print(survivors[0], len(nodes))
    return survivor_dist

totals = [0] * 1000
for i in range(100):
    for v in range(1000):
        trial = simulate_churn(1000, 10, 0.05)
        totals[v] += trial[v]
plt.plot(range(100), map(lambda x: x / 100.0, totals))
plt.show()
