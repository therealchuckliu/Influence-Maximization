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

def init_ind_cascade(nodes,nc,max_iterations = float("inf")):
    activated = set(nodes)
    q = Queue.Queue()
    for node in nodes:
        q.put((node, 0))
    
    while not q.empty():
        node, iteration = q.get()
        if iteration <= max_iterations:
            edges = nx.edges_iter(nc, nbunch=node)
            for x in edges:
                #if x[1] not in activated and edge_activate(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count']):
                if x[1] not in activated and random.uniform() <= random.beta(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count'])**0.5:
                    activated.add(x[1])
                    q.put((x[1], iteration+1))
        else:
            return activated
        #ind_cascade(node,nc,activated,q)
        
    return activated
    
def init_full_cascade(nodes,nc,max_iterations= float("inf")):
    activated = set(nodes)
    q = Queue.Queue()
    for node in nodes:
        q.put((node, 0))
    
    while not q.empty():
        node, iteration = q.get()
        if iteration <= max_iterations:
            edges = nx.edges_iter(nc, nbunch=node)
            for x in edges:
                #if x[1] not in activated and edge_activate(nc[x[0]][x[1]]['weight'],nc.node[x[1]]['review_count']):
                if x[1] not in activated and True:
                    activated.add(x[1])
                    q.put((x[1], iteration+1))
        else:
            return activated
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
    
def lambda_trial(nc,seed):
    #attempts to find a best fit for std = N^(lambda)
    #lambda should be ~ -0.5

    #random.seed(seed)
    lambda_arr = []
    mu_arr = []
    means = []
    
    colors = "bgrcmykw"
    color_index = 0
    

    for k in xrange(0,6):
        
        node_check = True
        while node_check:
            print 'checking....'
            node = random.choice(nc.nodes())
            if len(init_ind_cascade([node],nc)) > 2:
                node_check = False
        #print "Starting Node " + str(node) + ":" + NC_digraph.nodes()[node]
        #print "Activated:" + str(len(init_ind_cascade(NC_digraph.nodes()[node],NC_digraph)))
        N_arr = [10,50,100,300,500]
        std_arr = []
        mean_here = 0
        
        print 'loop start'
        for N in N_arr:
            mean_arr = []
            run_size = 100
            for j in xrange(0,run_size - 1):
                results = numpy.zeros(N)  
                for i in range(0, N):
                    results[i] = len(init_ind_cascade([node],nc))
            
                mean_arr.append(numpy.mean(results))
                
            std_arr.append(numpy.std(mean_arr))
            mean_here = numpy.mean(mean_arr)
        
        
        print 'loop stop'
        
        log_N = []
        for N in N_arr:
            log_N.append(math.log(N))
            
        log_std =[]
        for std in log_std:
            log_std.append(math.log(std))
            
        print std_arr
        log_std = map(lambda x: math.log(x), std_arr)
        
        sol = numpy.polyfit(log_N,log_std,1)
        approx_std = map(lambda x: math.exp(sol[1])*x**sol[0], N_arr)
        
        print "lambda = " + str(sol[0])
        lambda_arr.append(sol[0])
        
        print "mu = " + str(math.exp(sol[1]))
        mu_arr.append(math.exp(sol[1]))
        
        print "mean = " + str(mean_here)
        means.append(mean_here)        
        
        #plt.hold(True)
        #plt.plot(N_arr,std_arr,c=colors[color_index], label='true ' + str(mean_here))
        #plt.plot(N_arr,approx_std,c=colors[color_index],label='approx '+ str(mean_here),'-')
        color_index += 1
        
        print str(k) + ' done.'
        
    plt.scatter(means,mu_arr)
    '''
    plt.ylabel(r'$\sigma(f_N)$',fontsize=15)
    plt.xlabel("N")
    plt.title('')
    plt.legend()
    '''
    #fig.text(.55,.80, r'$\sigma(f_N) = \frac{\mu}{N^{\lambda}}$',fontsize = 15)
    #fig.text(.55,.72, '$\lambda$ = ' + str(-sol[0]))
    #fig.text(.55,.67,'    $\mu$ = ' + str(math.exp(sol[1])))
    #plt.show()
    #fig.savefig('lambda_regression.png')
    
    return 

def cascade_trials(N, nodes, graph, max_iterations=float("inf")):
    results = numpy.zeros(N)
    start = time.time()
    for i in xrange(0, N):
        results[i] = len(init_ind_cascade(nodes, graph, max_iterations))
    #plottng utility for our paper
    '''
    fig = plt.figure()
    fig.add_subplot(111)
    plt.hist(results)
    plt.ylabel("Count")
    plt.xlabel("Mean Influence (I(s))")
    plt.title("N="+str(N))
    fig.text(.55,.8, 'mean = ' + str(numpy.mean(results)))
    fig.text(.55,.75,'    std = ' + str(numpy.std(results)))
    plt.show()
    fig.savefig('N1000_influence.png')
    '''
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

# returns an array of nodes from graph which k or more neighbors
def edge_count_pruner(graph,k,nodes=0):
    if nodes == 0:
        nodes = graph.nodes()
    start = time.time()
    pruned = []
    for n in nodes:
        if len(graph.edges(nbunch=n)) > k:
            pruned.append(n)
    print time.time() - start
    
    print float(len(pruned))/(float(len(graph.nodes())))  
    return pruned

# returns an array of nodes from graph with k or more neighbors-of-neighbors (slower)
def edge_count_pruner_2(graph,k,nodes=0):
    
    if nodes == 0:
        nodes = graph.nodes()
    start = time.time()
    pruned = []
    for n in nodes:
        found = []
        count = 0
        edges = nx.edges_iter(graph, nbunch=n)
        for x in edges:
            if x[1] not in found:
                count += 1
                found.append(x[1])
            edges2 = nx.edges_iter(graph,nbunch=x[1])
            for y in edges2:
                if y[1] not in found:
                    count += 1
                    found.append(y[1])
        if count > k:
            pruned.append(n)
    print time.time() - start
    
    print float(len(pruned))/(float(len(graph.nodes())))  
    return pruned
    
if __name__ == '__main__':
      
    nc_digraph = import_graph('NC_full.json')
    print 'imported!'

    #nodes1 = edge_count_pruner(nc_digraph,20)
    nodes2 = edge_count_pruner_2(nc_digraph,500)
    
    
    
    '''
    print 'importing...'    
    NC_digraph = import_graph("nc_mini.json")
    print 'import done'
    lambda_trial(NC_digraph,4)
    '''
   
    # Used for creating lambda regression histogram
    #print lambda_trial(NC_digraph,24)
    # returned:
    # std_arr = [3.912023005428146, 4.605170185988092, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082]
    # lambda = -0.480942917902
    # mu = 2.50377674976
    
    # Used for creating I(s) histogram
    #random.seed(24)
    #node = random.choice(NC_digraph.nodes())    
    #print cascade_trials(1000,[node],NC_digraph)
    
    #nodes = ['E6Eh1bz6fpo6EOPtctA-sg', 'VFOwxpOWH9RZ3iMelkRd7A']
    #N = 10000    
    #cascade_trials does above in one function, outputting dictionary
    #with time/mean/std
    #print cascade_trials(N, nodes, NC_digraph, 10)
    '''
    greedy method for calculating max influence
    takes a while to run with 1000 trials, output was:
    [u'VhI6xyylcAxi0wOy2HOX3w', u'NzWLMPvbEval0OVg_YDn4g', u'ts7EG6Zv2zdMDg29nyqGfA]
    '''
    #print greedy_max_influence(NC_digraph, 3, 1000)
    
    