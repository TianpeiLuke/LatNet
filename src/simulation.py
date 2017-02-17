
# -*- coding: utf-8 -*-
import numpy as np
import pymc 
import networkx as nx
from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime





def ising_2dmodel(G, m, n, Tmax):
    #
    #  p(s_(0,0), s_(0,1), ... , s_(1,0), ..., s_(m,n) ) \proto exp(-E(s,w))
    #         where E = \sum_{((i,j), (u,v))} weight*s_((i,j))*s_((u,v)) + \sum_{(i,j)}s_(i,j)

    #  p(s_(i,j)|s_rest) = \frac{exp(-(attributes_(i,j)+\sum_{(u,v) in G[(i,j)]}weight*s_{(u,v)})(s_(i,j) + 1)  ) }
    #                           {1 + exp(-2(attributes_(i,j) + \sum_{(u,v) in G[(i,j)]}weight*s_{(u,v)})}
    state = -np.ones((m,n))
    history = []
    history.append(np.copy(state))
    #print(history)
    _energy = _ising_energy(G, m, n, state)
    # a simple Gibbs sampler 
    #
    for t in range(Tmax):
        for i, idx in enumerate(G.nodes()):
            s_i = state[idx[0]][idx[1]]
            f = _cond_prob(G, idx, state, -s_i)
            dice = np.random.rand(1)
            #print({float(f): float(dice)})
            if dice <= f:
                # flip sign
                state[idx[0]][idx[1]] = -s_i
            history.append(np.copy(state))
            #print(history)

    return (state, history)
                   

def _cond_prob(G, u, state, val):

    Ep =  _partial_energy(G, u, state)
    log_prob = -Ep*(val + 1) - np.log(1 + np.exp(-2*Ep))
    return np.exp(log_prob)


def _ising_energy(G, m, n, state):
    energy = 0
    node_attributes = nx.get_node_attributes(G, 'attributes')
    for u, v, e in G.edges(data=True):
        s_u = state[u[0]][u[1]]
        s_v = state[v[0]][v[1]]
        energy = energy +  e['weight']*s_u*s_v + s_u*G.node[u]['attributes'] + s_v*G.node[v]['attributes']
    
    return energy


def _partial_energy(G, u, state):
    neighbors = G[u]
    partial_energy = 0
    for v in neighbors:
        s_v = state[v[0]][v[1]]
        partial_energy += G[u][v]['weight']*s_v
    partial_energy += G.node[u]['attributes']
    return partial_energy
