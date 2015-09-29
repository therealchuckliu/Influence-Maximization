# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 12:44:24 2015
Spark greedy functions
@author: charlesliu
"""
from main import cascade_trials

def greedy_trials(sc, num_trials, g, k, N, t=float("inf"), partitions=4):
    results = []
    nodes = []
    grdd = sc.parallelize(g.nodes(), partitions)
    for trial in range(0, num_trials):
        result = greedy_search(grdd, g, k, N, t)
        results.append(result[1])
        nodes.append(result[0])
    return {"nodes": nodes, "results": results}
    
def greedy_search(graph_rdd, graph, select_count, trials, iterations=float("inf")):
    max_influence = (set(), 0)
    for iteration in range(1, select_count+1):
        pairsRDD = graph_rdd.map(lambda x: (max_influence[0] | set([x]), cascade_trials(trials, max_influence[0] | set([x]), graph, iterations)['mean']))
        pairsRDD = pairsRDD.filter(lambda x: len(x[0]) == iteration)
        max_influence = pairsRDD.takeOrdered(1, key=lambda x: -x[1])[0]
    return max_influence

def node_count(results):
    ret = {}
    for result in results["nodes"]:
        for node in result:
            if node not in ret:
                ret[node] = 1
            else:
                ret[node] += 1
    return ret