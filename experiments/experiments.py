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


def overlay_failure(g, r):
    pass


def random_k(g, k):
    return random.sample(g.nodes(), k)


def nearest_k(g, k):
    target = random.choice(g.nodes())
    return [target] + sorted(g.nodes(), key=lambda x: nx.shortest_path_length(g, target, x))[1:k + 1]


def runtests():
    underlay_data_failure = {"title": "Underlay Failure Likelihood",
                             "x-axis": "Percentage of nodes removed", "y-axis": "Liklihood of data loss",
                             "plots": {}}

    underlay_data_partition = {"title": "Underlay Partition Chance",
                               "x-axis": "Percentage of nodes removed", "y-axis": "Liklihood of Partition",
                               "plots": {}}

    graphs = {"Chord": "chord.pickle", "Kademlia": "kad.pickle"}
    strats = {"Random K": random_k, "Nearest K": nearest_k}
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R = [0.1 + x * 0.1 for x in range(9)]
    samples = 100

    for graph_name, graph_file in graphs.items():
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
                        g = nx.read_gpickle(graph_file)
                        replicas = strats_func(g, k)
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

runtests()
