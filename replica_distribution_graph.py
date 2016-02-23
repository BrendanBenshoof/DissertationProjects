import networkx as nx
import random


class Universe(object):

    def __init__(self, size):
        self.g = nx.scale_free_graph(size, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.12,
                                     delta_out=0, create_using=None, seed=None).to_undirected()
        self.nodes = self.g.nodes()
        self.distances = nx.all_pairs_shortest_path_length(self.g)
        self.records = {}

    def add_record(self, r_id, replicas=1):
        self.records[r_id] = random.sample(self.nodes, replicas)

    def closest_dist(self, origin, r_id):
        closest = min(self.records[r_id], key=lambda x: self.distances[origin][x])
        return self.distances[origin][closest]


def Exponential_dist():
    import matplotlib.pyplot as plt
    u = Universe(10000)
    print(u)
    #max_dist = h2dPoint(0, 0.5).dist(h2dPoint(0, 0))
    # print(max_dist)
    records = list(range(1, 100))
    for r in records:
        u.add_record(r, r)

    distances = []
    for r in records:
        dist = 0
        for i in range(1000):
            p = random.choice(u.nodes)
            dist += u.closest_dist(p, r)
        distances.append(dist / 1000)
    plt.plot(records, distances)
    plt.show()


if __name__ == "__main__":
    Exponential_dist()
