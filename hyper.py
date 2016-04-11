
# coding: utf-8

# In[78]:

# get_ipython().magic('matplotlib inline')
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


def poincareFrom3d(v):
    x, y, z = v
    unit_vec = (-1 * x / z, -1 * y / z)
    assert(norm(unit_vec) < 1.0)
    return unit_vec


def poincareDist(a, b):
    # print(a,b)
    try:
        sigma = norm(np.subtract(a, b))**2.0 / ((1 - np.dot(a, a)) * (1 - np.dot(b, b)))
        return math.acosh(1 + 2 * sigma)
    except ValueError as e:
        print(a)
        print(b)
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
    # print("discriminate",B*B-4*A*C>=0)
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
        print("correction", correction)
        return np.array(p_high) * correction
#    print("L-error", low_curve_error)
    correction = (- 1 / low_curve_error)**0.5
    print("correction", correction)
    return np.array(p_low) * correction
    """

test_proj_point = closest_point_on_plane((0, 0, 0), test_plane)
# test_reproj_point = np.add(test_proj_point,(9,200,-10))
# print(test_reproj_point)
# x=closest_point_from_plane(test_reproj_point,test_plane)
# print(x)


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
    # print(r)
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
    # print(actual, ideal)
    return 0.01 * (ideal - actual)


def calculate_jiggle(center, point_dict, ideal_distances, t):
    keys = [k for k in point_dict if ideal_distances[k] > 0]
    poincare_center = poincareFrom3d(center)
    poincare_points = {k: poincareFrom3d(point_dict[k]) for k in keys}
    poincare_distances = {k: poincareDist(poincare_points[k], poincare_center) for k in keys}
    norm_plane = get_tangent_plane(center)
    proj_points = {k: closest_point_on_plane(point_dict[k], norm_plane) for k in keys}
    proj_unit_distorted_points = {k: unit(np.subtract(center, proj_points[k])) for k in keys}
    forces = {k: spring_force(poincare_distances[k], ideal_distances[k]) for k in keys}
    force_vectors = {k: forces[k] * proj_unit_distorted_points[k] for k in keys}
    delta = (sum([force_vectors[k][0] for k in keys]) * t, sum([force_vectors[k][1]
                                                                for k in keys]) * t, sum([force_vectors[k][2] for k in keys]) * t)
#    #if norm(delta) > 0.5:
#        print("truncated",norm(delta))
#        delta = unit(delta)*0.5
    # print("angle", math.acos(np.dot(delta, norm_plane[:3]) / norm(delta) / norm(norm_plane[:3])))
    new_center = np.add(center, delta)
    return closest_point_from_plane(new_center, norm_plane)


# In[73]:

import networkx as nx
import matplotlib.pyplot as plt


def scale_free_topology(size, alpha=0.66, beta=0.44, gamma=0.0, delta_in=.12):
    directed = nx.scale_free_graph(size)
    undirected = directed.to_undirected()
    distances = nx.all_pairs_shortest_path_length(undirected)
    return undirected, distances


# In[ ]:
def build_hyperbolic_overlay(nodes, locs):
    min_short = 7  # minimum number of short peers
    max_long = 0  # maximum number of long peers
    new_g = nx.Graph()
    new_g.add_nodes_from(nodes)
    for a in nodes:
        peers = []
        canidates = sorted(nodes, key=lambda x: -1 * poincareDist(locs[x], locs[a]))
        canidates.remove(a)
        reserve = []
        # assert(a == canidates.pop())  # pop a off
        peers.append(canidates.pop())
        while len(canidates) > 0:
            c = canidates.pop()
            to_beat = poincareDist(locs[a], locs[c])
            best = min(map(lambda x: poincareDist(locs[x], locs[c]), peers))
            if best >= to_beat:
                peers.append(c)
            else:
                reserve.append(c)
        if len(peers) < min_short:
            peers += reserve[:min_short - len(peers)]
        for p in peers:
            new_g.add_edge(a, p)

    nx.draw(new_g, pos=locs)
    plt.show()
    return new_g


# subset = random.sample(g.nodes(), 200)


def areRoutesGreedy(nodes, locs, path_lengths):
    greedy_paths = 0
    total_paths = 0
    failed = 0
    overlay_graph = build_hyperbolic_overlay(nodes, locs)
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
                # print(a, pointer, options)
                closest = min(options, key=lambda x: poincareDist(locs[x], locs[b]))
                count += 1  # path_lengths[pointer][closest]
                hops += path_lengths[pointer][closest]
                pointer = closest
                if count > len(nodes):
                    print("infinte loop. tis bad")
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


def hyperbolic_fit(keys, distances):
    point_locs = {k: poincareTo3d(*genPpoint()) for k in keys}
    t = 1.0
    last_error = 100000
    delta_error = 1.0
    while delta_error > 0.001:
        # yield {k: poincareFrom3d(point_locs[k]) for k in keys}
        # print(t)
        new_locs = {}
        for k in keys:
            new_locs[k] = calculate_jiggle(point_locs[k], point_locs,
                                           distances[k], t)
        point_locs = new_locs
        t -= 0.01
        error = 0.0
        for a in keys:
            for b in keys:
                actual = poincareDist(point_locs[a], point_locs[b])
                ideal = distances[a][b]
                error += (ideal - actual)**2.0
        new_error = error / len(g.nodes())**2.0
        print("error", new_error)
        delta_error = math.fabs(new_error - last_error)
        last_error = new_error
    return {k: poincareFrom3d(point_locs[k]) for k in keys}


def smart_hyperbolic_fit(keys, g):
    distances = nx.all_pairs_shortest_path_length(g)
    centrality = nx.load_centrality(g)
    sorted__by_degree = sorted(keys, key=lambda x: -1 * centrality[x])
    placed = []
    batch_size = 10
    batch_generations = 10
    final_point_locs = {}
    batches = [sorted__by_degree[i:i + batch_size]
               for i in range(0, len(sorted__by_degree), batch_size)]
    i = 0
    for batch in batches:
        #print(i, batch)
        i += 1
        new_point_locs = {}
        if len(final_point_locs.keys()) > 0:
            for b in batch:
                parent = min(final_point_locs.keys(), key=lambda x: distances[
                             x][b])
                # sign_x = final_point_locs[parent][0] / math.fabs(final_point_locs[parent][0])
                # sign_y = final_point_locs[parent][1] / math.fabs(final_point_locs[parent][1])
                # print(final_point_locs[parent])
                parent_angle = math.atan2(final_point_locs[parent][1], final_point_locs[parent][0])
                parent_radius = (final_point_locs[parent][
                                 0] * final_point_locs[parent][0] + final_point_locs[parent][1])
                # print(parent_angle)
                child_angle = (0.5 * math.pi - random.random() * math.pi) / \
                    (1 + parent_radius)   # angle relative to parent's radial
                if parent_radius == 0.0:
                    child_angle = random.random() * math.pi * 2
                r = 0.5
                x = final_point_locs[parent][0] + math.cos(parent_angle + child_angle) * r
                y = final_point_locs[parent][1] + math.sin(parent_angle + child_angle) * r
                z = -1 * (x * x + y * y + 1.0)**0.5
                #print(x, y, z)
                new_point_locs[b] = (x, y, z)
                placed.append(b)
        else:
            new_point_locs[batch[0]] = (0, 0, -1)
            new_point_locs.update({k: poincareTo3d(math.sin(
                k * math.pi * 2 / batch_size) * 0.5, math.cos(k * math.pi * 2 / batch_size) * 0.5) for k in batch[1:]})
            placed += batch
            # print(["%d %f %f %f" % (k, new_point_locs[k][0], new_point_locs[k][1], new_point_locs[k][2]) for k in batch])
        locs = final_point_locs.copy()
        locs.update(new_point_locs)
        t = 1.0
        last_error = 100000
        delta_error = 1.0

        while delta_error > 0.01 and t > 0.01:
            # yield {k: poincareFrom3d(point_locs[k]) for k in keys}
            # print(t)
            for k in batch:
                new_point_locs[k] = calculate_jiggle(new_point_locs[k], locs,
                                                     distances[k], t)
            locs.update(new_point_locs)
            t -= 0.01
            error = 0.0
            for a in batch:
                for b in placed:
                    actual = poincareDist(locs[a], locs[b])
                    ideal = distances[a][b]
                    error += (ideal - actual)**2.0
            new_error = error / len(g.nodes())**2.0
            #print("error", new_error)
            delta_error = math.fabs(new_error - last_error)
            last_error = new_error
            for b in batch:
                new_point_locs[b] = calculate_jiggle(locs[b], locs, distances[b], t)

        final_point_locs.update(locs)
    return {k: poincareFrom3d(final_point_locs[k]) for k in keys}

g, test_dist = scale_free_topology(100)

for size in [32]:
    print("\nSize %d" % size)
    nodes = random.sample(g.nodes(), size)
    pos = smart_hyperbolic_fit(nodes, g)

    areRoutesGreedy(nodes,  pos, test_dist)


#nx.draw(g, pos=pos, nodelist=nodes, labels={k: "%d" % (k) for k in g.nodes()})
# plt.show()
