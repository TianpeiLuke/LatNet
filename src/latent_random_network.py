
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
#from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime

class latent_random_field:

    def __init__ (self, size, prob, option):

        try:
            self.seed = option['seed']
        except KeyError:
            self.seed = None

        try:
            self.node_dim = option['node_dim']
        except KeyError:
            self.node_dim = 2

        try:
            self.model_name = option['model']
        except KeyError:
            self.model_name = 'tree'

        try:
            self.k_u = option['cutoff_freq']
        except KeyError:
            self.k_u = 10

        self.prob = prob
        self.size = size
        # if already written 





   #===================================================================

    def graph_build(self, size, prob, option, save_fig=False):
        '''
            build random graph. prob = [p_in, p_out], within-cluster edge prob and between-cluster edge prob. 

         option['model']

                  ="partition", random partition graph 
                      then  
                          size = [size_c1, size_c2, ..., size_ck] for ck clusters
                          prob = [p_in, p_out]

                  ="newman", Newman-Watts-Strogatz graph, small world ring graph
                          size = number of nodes
                          prob = probability of adding a new edge for each edge
                          option['k-NN'] = k, Each node is joined with its k nearest neighbors in a ring topology.

                  ="binomial", Erdős-Rényi graph or binomial graph
                          size = number of nodes
                          prob = probability of adding a new edge 

                  ="power", Power-Law cluster graph 
                          size = [n,m] where n = nodes, m = edges for each node
                          prob = probability of adding a triangle after adding a random edge
                  ="grid", Grid-2D graph
                          size = [m,n] where mxn nodes for 2D grid

                  ="tree", Power-law tree graph


 
         option['seed'] for random seed 
         option['node_dim'] for the dimension of node attributes
        '''
        seed = option['seed']
        node_dim = option['node_dim']

        if option['model'] == 'partition': 
            if len(prob) < 2:
                p_in = prob
                p_out = 1 - p_in
           
            [p_in, p_out] = prob
            G = nx.random_partition_graph(sizes=size, p_in=p_in, p_out=p_out, seed=seed, directed=False)
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*size[0]+['b']*size[1], font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
             

        elif option['model'] == 'newman':
            try:
                k = option['k-NN']
            except KeyError:
                k = 2

            if type(size) == list:
                size = sum(size)

            G=  nx.newman_watts_strogatz_graph(size, k=k, p=prob, seed=seed)
            pos = nx.circular_layout(G, dim=2, scale=1.0, center=None)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*size, font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        elif option['model'] == 'binomial':
            if prob <= 0.2:
                G = nx.fast_gnp_random_graph(size, prob, seed=seed)
            else:
                G = nx.gnp_random_graph(size, prob, seed=seed)  
            
            if type(size) == list:
                size = sum(size)
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*size, font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        elif option['model'] == 'power':
            G = nx.powerlaw_cluster_graph(n=size[0], m=size[1], p=prob, seed=seed)
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*size, font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")


        elif option['model'] == 'grid':
            if type(size) == int:
                size = [size, size]
            G = nx.grid_2d_graph(m=size[0], n=size[1])
            #pos = nx.spring_layout(G) #nx.fruchterman_reingold_layout(G) #nx.nx_pydot.graphviz_layout(G)
            pos = dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*sum(size), font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
        
        elif option['model'] == 'tree':
            try:
                gamma = option['gamma']
            except KeyError:
                gamma = 3
            tries = 10000
            G = nx.random_powerlaw_tree(n=size, gamma=gamma, seed=seed, tries=tries)
            pos = nx.circular_layout(G, dim=2, scale=1.0, center=None) #nx.shell_layout(G)#nx.spring_layout(G) #nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*size, font_color='w')
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
    
        G_out = nx.Graph()
        # node initialization 
        G_out.add_nodes_from(G.nodes(), attributes=np.ones((node_dim,)).T)
        G_out.add_edges_from(G.edges())
        nx.set_edge_attributes(G_out, 'weight', 1) #np.random.rand(1))
        # assign weight values 
        #for u, v, e in G_out.edges(data=True):
        #    e['weight'] = np.random.rand(1)

        return G_out


   #=============================================================================

    def get_node_attributes(self, G):
        n = len(G)
        d = len(G.node[G.nodes()[0]]['attributes'])
        X = np.ndarray((n,d))
        for i, d in G.nodes_iter(data=True):
            X[i,:] = d['attributes']

        nodeIdx = [{'node': idx, 'loc' : i} for i, idx in enumerate(G.nodes())]
        return [X, nodeIdx]


    def compute_laplacian_smoothness(self, G):
        total_diff = 0
        m = G.size()
        sum_weight = 0
        for node1, node2, d in G.edges_iter(data=True):
            total_diff = total_diff + d['weight']*np.linalg.norm(G.node[node1]['attributes']- G.node[node2]['attributes'])**2
            sum_weight = sum_weight + d['weight']
        return total_diff/sum_weight

    def compute_total_variation(self, G):
        total_variation = 0
        count = 0
        for i, idx in enumerate(G.nodes()):
            if len(G[idx]) != 0:
                temp_vec = sum([G.node[neighbor]['attributes']*G[idx][neighbor]['weight'] for neighbor in G[idx]])/sum([G[idx][neighbor]['weight'] for neighbor in G[idx]])
                total_variation = total_variation + np.linalg.norm(G.node[idx]['attributes'] - temp_vec)
                count = count + 1
        if count > 0:
            return total_variation/count
        else:
            return int(1e10)

    def get_edgelist(self, G):
        return G.edges()    



    def get_pos_coordinate(self, pos):
        return np.array([[pos[key][0], pos[key][1]] for key in pos])


    def plot_node_3d(self, pos_coordinates, edge_list,  node_values, view_angle, nodeIdx, columnIdx=0, figIdx=0, save_fig=False):  
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        [elev, azim] = view_angle
        ax.view_init(elev, azim)

        x = pos_coordinates[:,0]
        y = pos_coordinates[:,1]
        z = node_values[:,columnIdx]
        # plot a zero-plane
        Sx= np.linspace(min(x), max(x), 100)
        Sy= np.linspace(min(y), max(y), 100)
        SurfaceX, SurfaceY = np.meshgrid(Sx, Sy)
        SurfaceZ= np.zeros((len(Sx),))
        ax.plot_surface(SurfaceX,SurfaceY, SurfaceZ, rstride=8, cstride=8, alpha=0.05) 


        for edge in edge_list:
            n0_idx = next((item['loc'] for item in nodeIdx if item['node']== edge[0])) 
            n1_idx = next((item['loc'] for item in nodeIdx if item['node']== edge[1]))
            lx = np.array((x[n0_idx], x[n1_idx]))
            ly = np.array((y[n0_idx], y[n1_idx]))
            line=art3d.Line3D(lx, ly, np.zeros((2,)), marker='o', markevery=(0, 1), markerfacecolor='r', color='k', linewidth=0.5)
            ax.add_line(line)
        # plot node and node-attributes in stem plot
        for xi, yi, zi in zip(x, y, z):        
            line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='D', markevery=(1, 1), markerfacecolor='b', color='b',alpha=1)
            ax.add_line(line)
            #line=art3d.Line3D(*zip((xi, yi, zi), (xi, yi, 0)), marker='o', markevery=(1, 1), markerfacecolor='r', color='b', linewidth=1)
            #ax.add_line(line)
            #line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1, 1), color='b')
            #ax.add_line(line)
         


        ax.set_xlim3d(min(x), max(x))
        ax.set_ylim3d(min(y), max(y))
        ax.set_zlim3d(min([min(z),-1]), max([max(z), 1])) 
        plt.show()
        #filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netNode"+ str(figIdx) +".eps"
        #fig.savefig(filename, format='eps', dpi=1000)
        if save_fig == True:
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netNode"+ str(figIdx) +".png"
            fig.savefig(filename)
