import networkx as nx
from hashlib import sha1
from functools import reduce
import random
BASE = 8
MAX = 2**BASE


def buildChord(size):

    population = sorted([int(sha1(bytes(str(x), "UTF-8")).hexdigest(), 16) %
                         MAX for x in range(size)])
    g = nx.DiGraph()
    g.add_nodes_from(population)
    for p in population:
        for f in range(BASE):
            ideal = (p + 2**f) % MAX
            finger = min(population, key=lambda x: (x - ideal) % MAX)
            g.add_edge(p, finger)
    nx.write_gpickle(g, "chord.pickle")
    return g


def layout_chord(g):
    import math
    pos = {}
    nodes = sorted(g.nodes())
    angle = 0
    delta_angle = 2 * math.pi / len(nodes)
    for n in nodes:
        pos[n] = (math.sin(angle), math.cos(angle))
        angle += delta_angle
    return pos

"""

"""


def prefix_match(A, B):
    output = 0
    size = 0
    done = False
    for i in range(BASE)[::-1]:
        A_L = A // (2**i)
        B_L = B // (2**i)
        if A_L == B_L:
            output += A_L * 2**i
            size += 1
        else:
            break
    return size

B_SIZE = 5


def build_buckets_2(t, pop):

    buckets = {0: [t], 1: []}
    random.shuffle(pop)
    peers = []
    for p in pop:
        if p == t:
            continue
        pointer = buckets
        height = BASE - 1
        delta = t ^ p
        done = False
        while not done:
            bit = (delta // 2**height) % 2
            print(height)
            #print(p, height, bit, pointer)
            if type(pointer[bit]) == type([]):
                if len(pointer[bit]) < B_SIZE:
                    pointer[bit].append(p)
                    peers.append(p)
                    done = True
                else:
                    # bucket it full
                    if t in pointer[bit]:
                        # we have to split it

                        todo = pointer[bit] + [t, p]
                        pointer[bit] = {0: [], 1: []}
                        pointer = pointer[bit]
                        for r in todo:
                            sub_delta = delta = r ^ p
                            sub_bit = (delta // 2**(height - 1)) % 2
                            pointer[sub_bit].append(r)
                        peers.append(p)
                    done = True
            else:
                height -= 1
                pointer = pointer[bit]
    print(p, buckets)
    return peers


def build_buckets(t, pop):
    buckets = [[t]]
    random.shuffle(pop)
    for p in pop:

        if p == 1:
            continue
        best = max(range(len(buckets)), key=lambda x: prefix_match(p, buckets[x][0]))
        if len(buckets[best]) < B_SIZE - 1:
            buckets[best].append(p)
        else:
            if t in buckets[best]:
                todo = buckets[best]
                other = min(todo, key=lambda x: prefix_match(p, x))
                todo.remove(other)
                todo.append(p)
                buckets.remove(buckets[best])
                L_bucket = [t]
                R_bucket = [other]
                for n in todo:
                    if prefix_match(n, t) >= prefix_match(n, other):
                        L_bucket.append(n)
                    else:
                        R_bucket.append(n)
                buckets.append(L_bucket)
                buckets.append(R_bucket)
    # print(t)
    #print(sorted(buckets, key=len))
    return list(reduce(lambda x, y: x + y, buckets))


def DGVH(t, pop):
    pop = sorted(pop, key=lambda x: -1 * (x ^ t))
    peers = [pop[0]]
    pop.remove(pop[0])
    for p in pop:
        if t ^ p < min(peers, key=lambda x: x ^ p) ^ p:
            peers.append(p)
    return peers


def buildKAD(size):

    population = sorted([int(sha1(bytes(str(x), "UTF-8")).hexdigest(), 16) %
                         MAX for x in range(size)])
    g = nx.DiGraph()
    g.add_nodes_from(population)
    for p in population:
        peers = build_buckets_2(p, population)
        print(len(peers))
        for other in peers:
            g.add_edge(p, other)
    nx.write_gpickle(g, "kad.pickle")
    return g

g = buildKAD(100)
from matplotlib import pyplot as plt
nx.draw(g, pos=layout_chord(g), labels={x: str(x) for x in g.nodes()})
print(nx.is_strongly_connected(g))
plt.show()
