# -*- coding: utf-8 -*-

import numpy as np
#from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime

class latent_signal_network:

   
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
            self.model_name = newman

        try:
            self.k_u = option['cutoff_freq']
        except KeyError:
            self.k_u = 10

        self.prob = prob
        self.size = size
        # if already written 
        self.ifwrite = False

        # random graph generate
        if self.model_name == 'partition':
            [self.p_in, self.p_out] = prob
            self.n = sum(self.size)
            G = nx.random_partition_graph(sizes=self.size, p_in=self.p_in, p_out=self.p_out, seed=self.seed, directed=False)
            self.pos = nx.nx_pydot.graphviz_layout(G)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.size[0]+['b']*self.size[1], font_color='w')

        elif self.model_name == 'newman':
            self.k = option['k-NN']
            self.n = self.size
            G=  nx.newman_watts_strogatz_graph(self.size, k=self.k, p=self.prob, seed=self.seed)
            self.pos = nx.circular_layout(G, dim=2, scale=1.0, center=None)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.n, font_color='w')

        elif self.model_name == 'binomial':
            self.n = self.size
            if self.prob < 0.2:
                G = nx.fast_gnp_random_graph(self.size, self.prob, seed=self.seed)
            else:
                G = nx.gnp_random_graph(self.size, self.prob, seed=self.seed)  
            self.pos = nx.nx_pydot.graphviz_layout(G)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.n, font_color='w')

        elif self.model_name == 'power':
            self.n = self.size[0]
            G = nx.powerlaw_cluster_graph(n=self.size[0], m=self.size[1], p=self.prob, seed=self.seed)
            self.pos = nx.nx_pydot.graphviz_layout(G)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.n, font_color='w')
        
        elif self.model_name == 'grid':
            self.n = self.size[0]*self.size[1]
            G = nx.grid_2d_graph(m=self.size[0], n=self.size[1])
            self.pos = nx.nx_pydot.graphviz_layout(G)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.n, font_color='w')
            
        elif self.model_name == 'tree':
            try:
                self.gamma = option['gamma']
            except KeyError:
                self.gamma = 3
            tries = 10000
            self.n = self.size
            G = nx.random_powerlaw_tree(n=self.size, gamma=self.gamma, seed= self.seed, tries=tries)
            self.pos = nx.circular_layout(G, dim=2, scale=1.0, center=None) #nx.shell_layout(G) #nx.spring_layout(G) #nx.nx_pydot.graphviz_layout(G)
            nx.draw(G, pos=self.pos, arrows=False, with_labels=True, fontsize= 10, node_color=['r']*self.n, font_color='w')
        
        G_out = nx.Graph()
        # node initialization 
        G_out.add_nodes_from(G.nodes(), attributes=np.zeros((self.node_dim,)).T)
        G_out.add_edges_from(G.edges())
        # assign weight values
        for u, v, e in G_out.edges(data=True):
            e['weight'] = 1
        
        self.G = G_out
        self.X = np.ndarray((self.n, self.node_dim))
        # the normalized graph laplacian and its eigen decomposition 
        self.L = nx.normalized_laplacian_matrix(self.G)
        self.L_eig, self.U = eigsh(self.L, k=self.k_u, which='SM')
        self.L_eig = self.L_eig.real
        self.L_eig = self.L_eig[1:self.k_u]
        self.U     = self.U[:,1:self.k_u]


    def graph_build(self, size, prob, option, save_fig=False):
        '''
            build graph with two random partition. prob = [p_in, p_out], within-cluster edge prob and between-cluster edge prob. 

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
            pos = nx.nx_pydot.graphviz_layout(G)
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
        G_out.add_nodes_from(G.nodes(), attributes=np.zeros((node_dim,)).T)
        G_out.add_edges_from(G.edges())
        # assign weight values 
        for u, v, e in G_out.edges(data=True):
            e['weight'] = 1

        return G_out


#    def clear_graph(self, G):
#        self.G.clear()
#        self.model_name = "clear"
#        self.pos = None
#        self.size = None
#        self.prob = 0
#        self.seed= None
#        self.node_dim = 0


    def smooth_gsignal_generate(self, G, T, sigma, alpha=0, show_plot=False, overwrite=False):
        '''
           generate the node attributes in the graph associated with the network topology.
         G = Graph with "attributes" data for each node 
         T = maximum loops of smoothing 
         sigma = initialized noise-level 

        '''
    
        #initialization
        n = len(G)
        d = len(G.node[G.nodes()[0]]['attributes'])
        X = np.ndarray((n,d))

        if alpha > 1 :
            alpha = 1- np.exp(-alpha)
        elif alpha < 0:
            alpha = np.exp(alpha)

        for i, idx in enumerate(G.nodes()):
            d = len(G.node[idx]['attributes'])
            np.random.seed((i*13+7)%100)
            G.node[idx]['attributes'] = sigma*np.random.randn(d)
            X[i,:] = G.node[idx]['attributes']
        
        #tempG2 = G.copy()
        #for i, idx in enumerate(G.nodes()):   
        #    G.node[idx]['attributes'] = tempG2.node[idx]['attributes'] - sum([tempG2.node[allnodes]['attributes'] for allnodes in tempG2])/n
        #    X[i,:] = G.node[idx]['attributes']
        #tempG2.clear()    

        if show_plot == True:    
            plt.figure(1)
            plt.subplot("511")
            plt.stem(np.arange(X.shape[0]),X[:,0])   
    
        sepT = int(T/5)
        hist_tv = np.zeros((T,))
        # run a smoothing process on network
        for t in range(T):
            tempG = G.copy()
            for i, idx in enumerate(G.nodes()):
                if len(tempG[idx]) != 0:
                    G.node[idx]['attributes'] = (1-alpha)*sum([tempG.node[neighbor]['attributes'] for neighbor in tempG[idx]])/len(tempG[idx]) + alpha*tempG.node[idx]['attributes']
            
            # avoid all one features
            tempG2 = G.copy()
            for i, idx in enumerate(G.nodes()):
                G.node[idx]['attributes'] = tempG2.node[idx]['attributes'] - sum([tempG2.node[allnodes]['attributes'] for allnodes in tempG2])/n

    
            for i, d in G.nodes_iter(data=True):
                X[i,:] = d['attributes']
            
            hist_tv[t] = self.compute_total_variation(G)

            if t%sepT == 0 and show_plot == True:
                plt.figure(1)
                print("Iteration: " + str(t+1))
                plt.subplot(str(int(T/sepT)+1)+"1"+str(int(t/sepT)+2))
                plt.stem(np.arange(X.shape[0]),X[:,0])
                plt.axis([0, 40, -6, 8])
    
        if show_plot == True:        
            plt.show()        

        if overwrite == True:
            self.G = G.copy()
            self.X = np.copy(X)
            self.hist_tv = np.copy(hist_tv)       
            self.ifwrite = True
        else:
            self.ifwrite = False 

        return [G, X, hist_tv]


    def graph_fourier_transform(self, X, show_fig=False, save_fig=False, overwrite=False):
        U_k = self.U
        gft = np.dot(U_k.T, X)
        if overwrite: self.X_f = gft
        if show_fig:
            fig = plt.figure(figsize=(15, 6))
            ax1  = fig.add_subplot(1,2,1)
            val  = X[:,0]
            markerline, stemlines, baseline = plt.stem(np.arange(len(val)), val,'b')
            plt.xlim([-0.1, len(val)+0.1])
            plt.xlabel('node index')
            plt.ylabel('graph signal')

            ax2 =  fig.add_subplot(1,2,2)
            val =  gft[:,0]
            markerline2, stemlines2, baseline2 = plt.stem(np.arange(len(val)),val,'b')
            plt.xlim([-0.1, len(val)+0.1])
            plt.xlabel('graph freq. comp')
            plt.ylabel('magnitude of graph Fourier transform')

            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_gft.eps"
            if save_fig :
                fig.savefig(filename, format="eps")
                 

        return gft


    def inference_hidden_graph_regul(self, sigma, init_X=None):
        # find the latent variables in factor analysis model
        #   h = arg min 0.5*|| x - U*h||**2 + (0.5/sigma**2)*h.T*Lambda*h 
        if init_X is not None:
            X = init_X
        else: 
            if self.ifwrite:
                X = self.X
            else:
                X = np.random.randn(self.n, self.node_dim)
    

        U_k = self.U#[:, 0:self.k_u]
        lambda_k = self.L_eig#[0:self.k_u]
        shinkage = lambda_k/sigma**2 + np.ones(lambda_k.shape)

        h = (np.dot(U_k.T, X).T/shinkage).T
        return h


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
        