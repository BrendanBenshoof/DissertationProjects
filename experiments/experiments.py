#!/home/brendan/anaconda3/bin/python3

import networkx as nx
import json
import random


def underlay_failure(g, r):
    #print(len(g.nodes()), r)
    partial = len(g.nodes()) * r
    # print(partial)
    num = int(partial)
    #print(num, r)
    to_remove = random.sample(g.nodes(), num)
    g.remove_nodes_from(to_remove)


def overlay_failure(g,dists, r):
    partial = len(g.nodes()) * r
    # print(partial)
    num = int(partial)
    target = random.choice(g.nodes())
    to_remove = sorted(g.nodes(), key=lambda x: dists[target][x]+random.random()*0.5)[:num]
    g.remove_nodes_from(to_remove)

def random_k(g,dists, k):
    return random.sample(g.nodes(), k)


def nearest_k(g,dists, k):
    target = random.choice(g.nodes())
    return [target] + sorted(g.nodes(), key=lambda x: dists[x][target]+random.random()*0.5)[1:k + 1]


def runtests_underlay():
    underlay_data_failure = {"title": "O Failure Likelihood",
                             "x-axis": "Percentage of nodes removed", "y-axis": "Liklihood of data loss",
                             "plots": {}}

    underlay_data_partition = {"title": "Underlay Partition Chance",
                               "x-axis": "Percentage of nodes removed", "y-axis": "Liklihood of Partition",
                               "plots": {}}
    g_chord = nx.read_gpickle("chord.pickle")
    chord_paths = nx.all_pairs_shortest_path_length(g_chord)
    g_kad = nx.read_gpickle("kad.pickle")
    kad_paths = nx.all_pairs_shortest_path_length(g_kad)

    graphs = {"Chord":(g_chord, chord_paths) , "Kademlia":(g_kad,kad_paths) }
    strats = {"Random K": random_k, "Nearest K": nearest_k}
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R = [0.1 + x * 0.1 for x in range(9)]
    samples = 100

    for graph_name, graph_data in graphs.items():
        for strats_name, strats_func in strats.items():
            for k in K:
                xs = R[:]
                # print(xs)
                failure_likelihoods = []
                partition_chance = []
                for r_ in xs:
                    survivor_total = 0.0
                    partition_total = 0.0
                    for null in range(samples):
                        g = graph_data[0].copy()
                        dists = graph_data[1]
                        replicas = strats_func(g,dists, k)
                        underlay_failure(g, r_)
                        if nx.is_strongly_connected(g):
                            partition_total += 1.0
                        victory = False
                        for r in replicas:
                            if r in g.nodes():
                                victory = True
                        if victory:
                            survivor_total += 1.0
                    survivor_total /= samples
                    partition_total /= samples
                    failure_likelihoods.append(survivor_total)
                    partition_chance.append(partition_total)
                failure_plot = {"x": xs, "y": failure_likelihoods}
                print("%s %s-%d" % (graph_name, strats_name, k))
                underlay_data_failure["plots"]["%s %s-%d Loss Chance" %
                                               (graph_name, strats_name, k)] = failure_plot
                partition__plot = {"x": xs, "y": failure_likelihoods}
                #
                underlay_data_partition["plots"]["%s %s-%d Partition Chance" %
                                                 (graph_name, strats_name, k)] = partition__plot
    with open("Underlay_partition.json", "w+") as fp:
        json.dump(underlay_data_partition, fp)
    with open("Underlay_survival.json", "w+") as fp:
        json.dump(underlay_data_failure, fp)

def runtests_overlay():
    underlay_data_failure = {"title": "Overlay Failure Likelihood",
                             "x-axis": "Percentage of nodes removed", "y-axis": "Liklihood of data loss",
                             "plots": {}}

    g_chord = nx.read_gpickle("chord.pickle")
    chord_paths = nx.all_pairs_shortest_path_length(g_chord)
    g_kad = nx.read_gpickle("kad.pickle")
    kad_paths = nx.all_pairs_shortest_path_length(g_kad)

    graphs = {"Chord":(g_chord, chord_paths) , "Kademlia":(g_kad,kad_paths) }
    strats = {"Random K": random_k, "Nearest K": nearest_k}
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R = [0.1 + x * 0.1 for x in range(9)]
    samples = 100

    for graph_name, graph_data in graphs.items():
        for strats_name, strats_func in strats.items():
            for k in K:
                xs = R[:]
                # print(xs)
                failure_likelihoods = []
                for r_ in xs:
                    survivor_total = 0.0
                    partition_total = 0.0
                    for null in range(samples):
                        g = graph_data[0].copy()
                        dists = graph_data[1]
                        replicas = strats_func(g,dists, k)
                        overlay_failure(g,dists, r_)

                        victory = False
                        for r in replicas:
                            if r in g.nodes():
                                victory = True
                        if not victory:
                            survivor_total += 1.0
                    survivor_total /= samples
                    partition_total /= samples
                    failure_likelihoods.append(survivor_total)
                failure_plot = {"x": xs, "y": failure_likelihoods}
                print("%s %s-%d" % (graph_name, strats_name, k))
                underlay_data_failure["plots"]["%s %s-%d Loss Chance" %
                                               (graph_name, strats_name, k)] = failure_plot

    with open("Overlay_survival.json", "w+") as fp:
        json.dump(underlay_data_failure, fp)

runtests_overlay()
