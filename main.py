# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:31:58 2015

@authors: stephencarr, charlesliu
"""

import json
import time
from functools import wraps
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from numpy import random
from pylab import polyfit
import numpy
import Queue
import math

def timefn(func):
    @wraps(func)
    def calc_time(*args, **kwargs):
        t1=time.time()
        result = func(*args,**kwargs)
        t2=time.time()
        print "@timefn: %.5f Seconds" % (t2-t1)
        return result
    return calc_time

def import_graph(filepath):
    with open(filepath, "r") as graph_json:
        graph_json = json.load(graph_json)
        return json_graph.node_link_graph(graph_json)
        
def print_graph(Graph, S1=None):
    plt.figure(figsize=(16,10))
    color_map = {1: 'b', 0: 'r'}
    pos = nx.random_layout(Graph)
    
    if S1:
        nx.draw_networkx(Graph, pos, with_labels=False, node_size=100, node_shape='.',
                linewidth=None, width=0.2, edge_color='y', 
                node_color=[color_map[Graph.node[node]['action']] for node in Graph],
                edgelist=reduce(lambda x,y: x+y,[Graph.edges(node) for node in S1]))
        nx.draw_networkx_nodes(Graph, pos, nodelist=S1, node_color="b", node_size=150, 
                              node_shape="*", label="Initial Set")
        plt.legend()
    else:
        nx.draw_networkx(Graph, pos, with_labels=False, node_size=100, node_shape='.',
                linewidth=None, width=0.2, edge_color='y', 
                 node_color=[color_map[Graph.node[node]['action']] for node in Graph])
        
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
@timefn
def rank_by_attribute(graph_dict, ret_num):
    top_nodes = graph_dict.keys()
    top_nodes.sort(key= lambda k: graph_dict[k], reverse=True)
    return top_nodes[:ret_num]
    

@timefn
def rank_by_attribute2(graph_dict, ret_num):
    top_nodes = [None]*ret_num
    for node in graph_dict:
        index = 0
        while index < ret_num and top_nodes[index] is not None and graph_dict[top_nodes[index]] > graph_dict[node]:
            index += 1
        if index < ret_num:
            top_nodes.insert(index, node)
    return top_nodes[:ret_num]

def init_ind_cascade(nodes,nc):
    activated = set(nodes)
    q = Queue.Queue()
    for node in nodes:
        q.put(node)
    
    while not q.empty():
        node = q.get()
        edges = nx.edges_iter(nc, nbunch=node)
        for x in edges:
            #if x[1] not in activated and edge_activate(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count']):
            if x[1] not in activated and random.uniform() <= random.beta(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count'])**0.5:
                activated.add(x[1])
                q.put(x[1])
        #ind_cascade(node,nc,activated,q)
        
    return activated

#unused 
def ind_cascade(node_id,nc,activated,q):
    edges = nx.edges_iter(nc,nbunch=node_id)
    for x in edges:
        #print "Edge " + str(x)
        if x[1] not in activated and edge_activate(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count']):
            activated.add(x[1])
            q.put(x[1])
 
#unused 
def edge_activate(a,b):
    v = math.sqrt(random.beta(a,b))
    u = random.uniform()
    #print "Weight: " + str(b) + "\nReview Count:" + str(a)
    #print "beta=" + str(v) + "\nuni=" + str(u) + "\nbeta-uni=" + str(v-u)
    return u <= v   

def cascade_trials(N, nodes, graph):
    results = numpy.zeros(N)
    start = time.time()
    for i in xrange(0, N):
        results[i] = len(init_ind_cascade(nodes, graph))
    return {"time": time.time() - start, "mean": numpy.mean(results), "std": numpy.std(results)}
    
def greedy_max_influence(g, size, infl_trials):
    sel_nodes = set()
    nodes = set(g.nodes())
    while len(sel_nodes) < size:
        inf_max = 0
        max_node = None
        for node in nodes:
            cascade_run = cascade_trials(infl_trials, sel_nodes | set([node]), g)
            if cascade_run["mean"] > inf_max:
                inf_max = cascade_run["mean"]
                max_node = node
        if max_node is not None:
            sel_nodes.add(max_node)
            nodes.remove(max_node)
        else:
            return sel_nodes
    return sel_nodes    
        
if __name__ == '__main__':
    
    NC_digraph = import_graph("nc_mini.json")
    '''  
    previous code for runs
    random.seed(24)
    node = random.randint(0, 240)
    node = NC_digraph.nodes()[node]
    #print "Starting Node " + str(node) + ":" + NC_digraph.nodes()[node]
    #print "Activated:" + str(len(init_ind_cascade(NC_digraph.nodes()[node],NC_digraph)))
    N_arr = [10,50,100,300,500,1000]
    std_arr = []
    for N in N_arr:
        mean_arr = []
        run_size = 100
        for i in xrange(0,run_size - 1):
            results = numpy.zeros(N)  
            tstart = time.clock()
            for i in range(0, N):
                results[i] = len(init_ind_cascade([node],NC_digraph))
        
            mean_arr.append(numpy.mean(results))   
            print str(N) + " done."
                    
            #print "Time: " + str(time.clock() - tstart)
            #print numpy.mean(results)        
            #print numpy.std(results)
        std_arr.append(numpy.std(mean_arr))
    
    plt.plot(N_arr,std_arr)
    print std_arr
    '''
    nodes = ['E6Eh1bz6fpo6EOPtctA-sg', 'VFOwxpOWH9RZ3iMelkRd7A']
    N = 10000    
    #cascade_trials does above in one function, outputting dictionary
    #with time/mean/std
    print cascade_trials(N, nodes, NC_digraph)
    '''
    greedy method for calculating max influence
    takes a while to run with 1000 trials, output was:
    [u'VhI6xyylcAxi0wOy2HOX3w', u'NzWLMPvbEval0OVg_YDn4g', u'ts7EG6Zv2zdMDg29nyqGfA]
    
    print greedy_max_influence(NC_digraph, 3, 1000)
    '''
    
    