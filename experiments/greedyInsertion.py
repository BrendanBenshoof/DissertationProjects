import math
import random
import networkx as nx
import matplotlib.pyplot as plt


def dot(a, b):
    return sum(map(lambda x, y: x * y, a, b))


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


def dist(a, b):

    sigma = 2 * eDist(a, b)**2.0 / ((1 - dot(a, a)) * (1 - dot(b, b)))
    return math.acosh(1 + sigma)


def average(locList):
    x_sum = sum([x[0] for x in locList])
    y_sum = sum([y[1] for y in locList])
    return (x_sum / len(locList), y_sum / len(locList))


class Space(object):

    def __init__(self, root):
        self.loc2objects = {(0, 0): root}  # location->node
        self.objects2loc = {root: (0, 0)}  # node->location
        self.objects = [root]
        self.overlay = nx.Graph()
        self.overlay.add_node(root)

    def remove(self, target):
        self.objects.remove(target)
        loc = self.objects2loc[target]
        del(self.loc2objects[loc])
        del(self.objects2loc[target])
        self.calculate_overlay()

    def insert(self, new_node, latency):
        root = self.objects[0]
        done = False
        while not done:
            print(root)
            canidates = self.overlay.neighbors(root) + [root]
            best = min(canidates, key=lambda x: latency[x][new_node] + random.random() * 0.5)
            if root == best:
                done = True
            else:
                root = best

        rootloc = self.objects2loc[root]
        magnitude = dot(rootloc, rootloc)**0.5
        if magnitude > 0:
            rem = (1 - dot(rootloc, rootloc)**0.5) * 0.5
            noise = (rootloc[0] + random.random() * rem * (math.fabs(rootloc[0]) / rootloc[0]),
                     rootloc[1] + random.random() * rem * (math.fabs(rootloc[1]) / rootloc[1]))
            newloc = noise  # average([self.objects2loc[best], self.objects2loc[root], noise])
            self.objects.append(new_node)
            self.objects2loc[new_node] = newloc
            self.loc2objects[newloc] = new_node
        else:
            angle = random.random() * math.pi * 2
            x = math.sin(angle) * 0.5
            y = math.cos(angle) * 0.5
            newloc = (x, y)
            self.objects.append(new_node)
            self.objects2loc[new_node] = newloc
            self.loc2objects[newloc] = new_node
        self.calculate_overlay()

    def calculate_overlay(self):
        new_overlay = nx.Graph()
        new_overlay.add_nodes_from(self.objects)
        for n in self.objects:
            focus_loc = self.objects2loc[n]
            candiates = sorted(self.objects, key=lambda x: dist(focus_loc, self.objects2loc[x]))
            peers = [candiates[1]]
            for c in candiates[2:]:
                accept = True
                target_dist = dist(focus_loc, self.objects2loc[c])
                for p in peers:
                    if dist(self.objects2loc[p], self.objects2loc[c]) < target_dist:
                        accept = False
                        break
                if accept:
                    peers.append(c)
            for p in peers:
                new_overlay.add_edge(n, p)
        self.overlay = new_overlay

g = nx.scale_free_graph(20).to_undirected()
latency = nx.all_pairs_shortest_path_length(g)
root = g.nodes()[0]

test = Space(root)
nodes = g.nodes()[1:]
for n in nodes:
    test.insert(n, latency)
random.shuffle(nodes)
for n in g.nodes()[1:]:
    test.remove(n)
    test.insert(n, latency)


locs = test.objects2loc

nx.draw(g, pos=locs, labels={x: x for x in g.nodes()})
plt.show()
