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
    print graph_dict[top_nodes[0]]
    print graph_dict[top_nodes[1]]    
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

def init_ind_cascade(node_id,nc):

    activated = [node_id]
    q = Queue.Queue()
    q.put(node_id)
    
    while not q.empty():
        node = q.get()
        ind_cascade(node,nc,activated,q)
        
    return activated
 
@profile
def ind_cascade(node_id,nc,activated,q):
 
    edges = nx.edges(nc,nbunch=node_id)
    for x in edges:
        if x[1] not in activated and edge_activate(nc[x[0]][x[1]]['weight'],nc.node[node_id]['review_count']) == 1:
            activated.append(x[1])
            q.put(x[1])
  
def edge_activate(b,a):
    v = math.sqrt(random.beta(b,a))
    u = random.uniform()
    if u < v:
        return 1
    else:
        return 0
        
        
if __name__ == '__main__':
    #705 = 132
    #469 = 118
    NC_digraph = import_graph("nc_mini.json")
    #print_graph(NC_digraph)
    results = []
    for k in range(0,1000):
        results.append(len(init_ind_cascade(NC_digraph.nodes()[162],NC_digraph))-1)
        
    plt.hist(results)
    print numpy.mean(results)
    print numpy.std(results)
    
    '''
    for k in range (500,520):
        seed = k
        random.seed(506)
        seeds.append(506)       
        results.append(len(init_ind_cascade(NC_digraph.nodes()[47],NC_digraph)))
        #print "seed =" + str(seed)
        #print "activated =" + str(len(init_ind_cascade(NC_digraph.nodes()[number],NC_digraph)))
    print seeds
    print results
    
    random.seed(506)

    random.seed(12)
    print random.choice(NC_digraph.nodes())
    print len(init_ind_cascade(random.choice(NC_digraph.nodes()),NC_digraph))
    
    
    #get top 5 nodes by review_count
    print rank_by_attribute(nx.get_node_attributes(NC_digraph, "review_count"), 3)
    #print rank_by_attribute2(nx.get_node_attributes(NC_digraph, "review_count"), 3)
    
    #get top 5 edges by weight
    #rank_by_attribute(nx.get_edge_attributes(NC_digraph, "weight"), 3)
    '''