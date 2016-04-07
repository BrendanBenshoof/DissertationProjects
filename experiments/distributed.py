"""
Distributed hyperbolic embedding algorithim simulation

"""
import networkx as nx
import random
import math

import numpy as np


import math


def norm(a):
    return sum(map(lambda x: x * x, a))**0.5


def poincareTo3d(x, y):
    # disk_point = (x,y,-1)
    # magnitude = (x*x+y*y+1)**0.5
    # unit_vec = (x/magnitude,y/magnitude,-1/magnitude)
    # z^2 = x^2+y^2+1
    # scale^2*z^2 = scale^2*x^2 + scale^2*y^2 + 1
    #(scale^2)*(z^2-x^2-y^2) = 1
    # scale = (z^2-x^2-y^2)^-0.5
    scale = (1 - x**2.0 - y**2.0)**-0.5
    # (unit_vec[0]*scale, unit_vec[1]*scale, unit_vec[2]*scale)
    return (x * scale, y * scale, -1 * scale)


def poincare_dist_from_3d(a, b):
    return poincareDist(poincareFrom3d(a), poincareFrom3d(b))


def poincareFrom3d(v):
    x, y, z = v
    unit_vec = (-1 * x / z, -1 * y / z)
    assert(norm(unit_vec) < 1.0)
    return unit_vec


def poincareDist(a, b):
    # #pint(a,b)
    try:
        sigma = norm(np.subtract(a, b))**2.0 / ((1 - np.dot(a, a)) * (1 - np.dot(b, b)))
        return math.acosh(1 + 2 * sigma)
    except ValueError as e:
        # pint(a)
        # pint(b)
        raise(e)


test_point = poincareTo3d(0.8, 0.1)
test_point
# poincareDist((0,0.9999999999),(0.0,-0.9999999999))


def get_tangent_plane(p):
    normal = (2 * p[0], 2 * p[1], -2 * p[2])
    # normal = (0, 0, 1)
    norm_mag = norm(normal)
    A = normal[0] / norm_mag
    B = normal[1] / norm_mag
    C = normal[2] / norm_mag
    D = A * p[0] + B * p[1] + C * p[2]
    return (A, B, C,  -1 * D)  # im owrried about the sign on D

test_plane = get_tangent_plane(test_point)
test_plane


def closest_point_on_plane(p, plane):
    A, B, C, D = plane
    x0, y0, z0 = p
    distance = (A * x0 + B * y0 + C * z0 + D) / (A * A + B * B + C * C)**0.5
    x1 = x0 - A * distance
    y1 = y0 - B * distance
    z1 = z0 - C * distance
    assert(math.fabs(A * x1 + B * y1 + C * z1 + D) < 0.01)
    return (x1, y1, z1)


def closest_point_from_plane(p, plane):
    Z = -1 * (p[0] * p[0] + p[1] * p[1] + 1)**0.5
    return (p[0], p[1], Z)
    """
    N = np.array(plane[:3])
    A = N[0] * N[0] + N[1] * N[1] - N[2] * N[2]
    B = 2 * (p[2] * N[2] - p[0] * N[0] - p[1] * N[1])
    C = p[0] * p[0] + p[1] * p[1] - p[2] * p[2] + 1
    # #pint("discriminate",B*B-4*A*C>=0)
    l_high = (-1 * B + (B * B - 4 * A * C)**0.5) / (2 * A)
    l_low = (-1 * B - (B * B - 4 * A * C)**0.5) / (2 * A)
    p_high = (p[0] - l_high * N[0], p[1] - l_high * N[1], p[2] - l_high * N[2])
    p_low = (p[0] - l_low * N[0], p[1] - l_low * N[1], p[2] - l_low * N[2])
    high_curve_error = p_high[0] * p_high[0] + p_high[1] * p_high[1] + 1.0 - p_high[2] * p_high[2]
    low_curve_error = p_low[0] * p_low[0] + p_low[1] * p_low[1] + 1.0 - p_low[2] * p_low[2]

    if high_curve_error == 0.0:
        return p_high
    if high_curve_error < low_curve_error:

        correction = (-1 / high_curve_error)**0.5
        # pint("correction", correction)
        return np.array(p_high) * correction
#    #pint("L-error", low_curve_error)
    correction = (- 1 / low_curve_error)**0.5
    # pint("correction", correction)
    return np.array(p_low) * correction
    """

test_proj_point = closest_point_on_plane((0, 0, 0), test_plane)
# test_reproj_point = np.add(test_proj_point,(9,200,-10))
# #pint(test_reproj_point)
# x=closest_point_from_plane(test_reproj_point,test_plane)
# #pint(x)


def get2dCoords(origin, point):
    plane = get_tangent_plane(origin)
    projected_point = closest_point_on_plane(point, plane)

    real_y_axis = np.array([0, 0, -1])
    if plane[0] == 0 and plane[1] == 0:
        real_y_axis = np.array([0, 1, 0])

    projected_y_axis = real_y_axis - np.array(plane[:3]) * np.dot(real_y_axis, plane[:3])
    real_x_axis = np.cross(plane[:3], real_y_axis)
    projected_x_axis = real_x_axis - np.array(plane[:3]) * np.dot(real_x_axis, plane[:3])
    x = np.dot(projected_x_axis, projected_point)
    y = np.dot(projected_y_axis, projected_point)
    return x, y

get2dCoords(test_point, test_point)


import random
import math
random.seed(0)


def genPpoint():
    angle = random.random() * math.pi * 2
    r = math.log(1 + 1 * random.random()) / math.log(10)
    # #pint(r)
    x = math.sin(angle) * r
    y = math.cos(angle) * r
    return (x, y)

pop = [genPpoint() for x in range(100)]

"""
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([x[0] for x in pop],[x[1] for x in pop],"o")
plt.axis([-1,1,-1,1])
plt.show()
"""


# In[20]:


"""
center = poincareTo3d(0.9,0.0)
_3dpop = [poincareTo3d(x[0],x[1]) for x in pop]

top = (0,0,-1)

proj_center = get2dCoords(center,center)


proj_top = np.subtract(proj_center,get2dCoords(center,top))


projected = [np.subtract(proj_center,get2dCoords(center,x)) for x in _3dpop]


plt.plot([x[0] for x in projected],[x[1] for x in projected],"ob")
plt.plot([proj_top[0]],[proj_top[1]],"or")
plt.plot([0],[0],"og")
plt.show()
"""

# In[65]:


def unit(v):
    norm_mag = sum(map(lambda x: x * x, v))**0.5
    if norm_mag == 0.0:
        return v
    return np.array(v) * (1.0 / norm_mag)
"""
poincare_origin = (0.95,0.0)
center = poincareTo3d(*poincare_origin)

_3dpop_dists = [(poincareTo3d(x[0],x[1]) ,poincareDist(poincare_origin,x)) for x in pop]

proj_center = get2dCoords(center,center)


top = (0,0,-1)

proj_top = unit(np.subtract(proj_center,get2dCoords(center,top)))* \
                poincareDist(poincare_origin,(0,0))



projected = [unit(np.subtract(proj_center,get2dCoords(center,x[0])))*x[1] for x in _3dpop_dists]


plt.plot([x[0] for x in projected],[x[1] for x in projected],"ob")
plt.plot([proj_top[0]],[proj_top[1]],"or")
plt.plot([0],[0],"og")
plt.show()
"""


def spring_force(actual, ideal):
    # #pint(actual, ideal)
    return 0.01 * (ideal - actual)


def calculate_jiggle(center, point_dict, ideal_distances, t):
    keys = [k for k in point_dict if ideal_distances[k] > 0]
    poincare_center = poincareFrom3d(center)
    poincare_points = {k: poincareFrom3d(point_dict[k]) for k in keys}
    poincare_distances = {k: poincareDist(poincare_points[k], poincare_center) for k in keys}
    norm_plane = get_tangent_plane(center)
    normal_vector = np.array(norm_plane[:3])
    proj_points = {k: closest_point_on_plane(point_dict[k], norm_plane) for k in keys}
    proj_unit_distorted_points = {k: unit(np.subtract(center, proj_points[k])) for k in keys}
    forces = {k: spring_force(poincare_distances[k], ideal_distances[k]) for k in keys}
    force_vectors = {k: forces[k] * proj_unit_distorted_points[k] for k in keys}
    keys.append("G")
    G = np.array([0, 0, -1])
    Projected_G = np.subtract(
        G, normal_vector * (np.dot(G, normal_vector) / norm(normal_vector)**2.0))
    force_vectors["G"] = unit(Projected_G) * len(keys) * 1000
    delta = (sum([force_vectors[k][0] for k in keys]) * t, sum([force_vectors[k][1]
                                                                for k in keys]) * t, sum([force_vectors[k][2] for k in keys]) * t)
    while norm(delta) > 0.5:
        delta = unit(delta) * 0.5
    # #pint("angle", math.acos(np.dot(delta, norm_plane[:3]) / norm(delta) / norm(norm_plane[:3])))
    new_center = np.add(center, delta)
    return closest_point_from_plane(new_center, norm_plane)


# In[73]:


def scale_free_topology(size, alpha=0.66, beta=0.44, gamma=0.0, delta_in=.12):
    directed = nx.scale_free_graph(size)
    undirected = directed.to_undirected()
    distances = nx.all_pairs_shortest_path_length(undirected)
    return undirected, distances


SHORT_PEERS_MIN = 7
LONG_PEER_MAX = int(SHORT_PEERS_MIN**2)


def dist(a, b):
    # return sum(map(lambda x, y: (x * x - y * y)**2.0, a, b))**0.5
    return poincare_dist_from_3d(a, b)


def subordinate_loc(patron):
    r = 0.5 + random.random() * 0.5
    # #print(patron)
    patron_r = poincareDist((0, 0), poincareFrom3d(patron.get_loc()))
    angle = 2 * math.pi * random.random()
    if patron_r > 0.0:
        angle = 0.5 * ((random.random() - 0.5) * math.pi)
    # print(angle)
    patron_angle = math.atan2(patron.loc[1], patron.loc[0])
    sub_x = patron.loc[0] + math.cos(patron_angle + angle) * r
    sub_y = patron.loc[1] + math.sin(patron_angle + angle) * r
    sub_z = -1 * (sub_x * sub_x + sub_y * sub_y + 1)**0.5
    # pint((sub_x, sub_y, sub_z))
    return (sub_x, sub_y, sub_z)


def areRoutesGreedy(g, locs, path_lengths):
    nodes = g.nodes()
    greedy_paths = 0
    total_paths = 0
    failed = 0
    overlay_graph = g  # build_hyperbolic_overlay(nodes, locs)
    # culled = random.sample(nodes, 20)
    # overlay_graph.remove_nodes_from(culled)
    # nodes = overlay_graph.nodes()[:]
    strech = 0.0
    hops = 0
    total_path_length = 0
    for a in random.sample(nodes, min((100, len(nodes)))):
        for b in random.sample(nodes, min((100, len(nodes)))):
            if a == b:
                continue
            count = 0
            pointer = a
            while pointer != b:

                options = overlay_graph.neighbors(pointer)
                # #print(a, pointer, options)
                closest = min(options, key=lambda x: poincareDist(locs[x], locs[b]))
                count += 1  # path_lengths[pointer][closest]
                hops += path_lengths[pointer][closest]
                pointer = closest
                if count > len(nodes):
                    # print("infinte loop. tis bad")
                    failed += 1
                    break
            if count == path_lengths[a][b]:
                greedy_paths += 1
            # total_path_length += path_lengths[a][b]
            strech += count / path_lengths[a][b]
            total_path_length += count
            total_paths += 1
    # average_path_length = total_path_length/total_paths

    print(total_paths, "Paths")
    print(greedy_paths, "Are prefectly efficent")
    print(failed, "totally failed")
    print(hops / total_paths, "average hops")
    mean_underlay_dist = sum([sum([path_lengths[a][b] for b in nodes])
                              for a in nodes]) / (len(nodes)**2.0)

    print(mean_underlay_dist * math.log(len(nodes)) /
          math.log(2), "expected path length in regular DHT")


class Network(object):

    def __init__(self):
        self.nodes = {}
        self.latencies = {}

    def add_Node(self, node, latencies):
        id = node.id
        self.nodes[id] = node
        self.latencies[id] = {}
        self.latencies[id].update(latencies)
        for n in self.nodes.keys():
            # pint(n, id)
            self.latencies[n][id] = self.latencies[id][n]

    def random_Node(self):
        id = random.choice(list(self.nodes.keys()))
        return self.nodes[id]

    def ping(self, a, b):
        return self.latencies[a.id][b.id]


class Node(object):

    def __init__(self, id, network):
        self.id = id
        self.loc = None
        self.net = network
        self.short_peers = {}
        self.long_peers = {}

    def __repr__(self):
        return "<NODE %d at %s>" % (self.id, str(self.loc))

    def get_loc(self):
        return self.loc

    def getPeers(self):
        peers = {}
        peers.update(self.short_peers)
        return peers

    def getAllPeers(self):
        peers = {}
        peers.update(self.short_peers)
        peers.update(self.long_peers)
        return peers

    def notify(self, origin):
        if origin in self.short_peers.keys():
            self.short_peers[origin] = origin.get_loc()
        else:
            self.long_peers[origin] = origin.get_loc()

    def join(self):
        # seek the origin
        last_hop = None
        next_hop = self.net.random_Node()
        while next_hop == self:
            next_hop = self.net.random_Node()

        while last_hop != next_hop:
            # print(next_hop)
            last_hop = next_hop
            candiates = next_hop.getPeers()

            candiates[next_hop] = next_hop.get_loc()
            # pint(self, candiates)
            next_hop = min(candiates.keys(), key=lambda x: dist((0, 0, -1), x.get_loc()))

        # I've found the center of the network, do the same for latency
        last_hop = None
        while last_hop != next_hop:
            # print(next_hop)
            last_hop = next_hop
            candiates = next_hop.getPeers()
            candiates[next_hop] = next_hop.get_loc()
            next_hop = min(candiates.keys(), key=lambda x: self.net.ping(
                x, self) + random.random() * 0.5)
        patron = next_hop
        self.loc = subordinate_loc(patron)
        self.short_peers = patron.getPeers()
        patron.notify(self)
        self.tick()
        patron.tick()
        for p in patron.getPeers():
            p.tick()

    def tick(self):
        # update peer lists
        # self.update_loc()
        for p in self.short_peers.keys():
            self.long_peers.update(p.getPeers())
        self.dgvh_filter()
        # update location

        # notify those short peers:
        for peer in self.short_peers.keys():
            peer.notify(self)

    def dgvh_filter(self):
        peers = self.long_peers.copy()
        peers.update(self.short_peers)
        if len(peers.keys()) == 0:
            return
        new_short_peers = []
        new_long_peers = []
        long_peer_canidates = []
        canidates = sorted(peers.keys(), key=lambda x: -1 * dist(x.get_loc(), self.get_loc()))
        gimmie_peer = canidates.pop()
        new_short_peers.append(gimmie_peer)
        while len(canidates) > 0:
            considered = canidates.pop()
            to_beat = dist(self.get_loc(), considered.get_loc())
            best_peer = min(new_short_peers, key=lambda x: dist(considered.get_loc(), x.get_loc()))
            if to_beat <= dist(considered.get_loc(), best_peer.get_loc()):
                new_short_peers.append(considered)
            else:
                long_peer_canidates.append(considered)
        if len(new_short_peers) < SHORT_PEERS_MIN:
            new_short_peers += long_peer_canidates[:SHORT_PEERS_MIN - len(new_short_peers)]
            long_peer_canidates = long_peer_canidates[SHORT_PEERS_MIN - len(new_short_peers):]
        new_long_peers = long_peer_canidates[:LONG_PEER_MAX]
        self.short_peers = {k: peers[k] for k in new_short_peers}
        self.long_peers = {k: peers[k] for k in new_long_peers}

    def update_loc(self):
        self.loc = calculate_jiggle(self.loc, {p: p.loc for p in self.short_peers.keys()}, {
            p: self.net.ping(self, p) for p in self.short_peers.keys()}, 0.1)

graph, latency = scale_free_topology(1000)

for size in range(50, 1000, 50):
    #    UNDERLAY_SIZE = 10 * size

    OVERLAY_SIZE = size
    centrality = nx.betweenness_centrality(graph)
    nodes = sorted(random.sample(graph.nodes(), OVERLAY_SIZE), key=lambda x: centrality[x])
    net = Network()
    start = nodes.pop()
    origin = Node(start, net)
    origin.loc = (0, 0, -1)
    net.add_Node(origin, latency[0])

    j = 0
    while(len(nodes) > 0):
        # print(j)
        j += 1
        i = nodes.pop()
        # #print(i)
        new_node = Node(i, net)
        latencies = latency[i]  # {k: random.random() for k in net.nodes.keys()}
        latencies[i] = 0.0
        net.add_Node(new_node, latencies)
        new_node.join()
        # for n in net.nodes.keys():
        #    net.nodes[n].tick()
        # new_node.update_loc()
        # #pint(new_node.loc)
    for i in range(10):
        # print(i)
        for n in net.nodes.keys():
            net.nodes[n].tick()

    g = nx.DiGraph()
    g.add_nodes_from(net.nodes.keys())
    for n in g.nodes():
        peers = net.nodes[n].getAllPeers()
        for p in peers.keys():
            g.add_edge(n, p.id)
    pos = {k: poincareFrom3d(net.nodes[k].loc) for k in g.nodes()}

    print("\n")
    print(size, "SIZE")
    areRoutesGreedy(g, pos, latency)
