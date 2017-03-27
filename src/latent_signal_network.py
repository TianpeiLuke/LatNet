# -*- coding: utf-8 -*-

import numpy as np
#from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse,  csr_matrix, isspmatrix, isspmatrix_csr
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime
import time
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve, average_precision_score, auc



def edge_list_convert(Laplacian):
    n = Laplacian.shape[0]
    edge_list_string = []
    for i in range(n):
        base_idx = np.arange(i+1,n)
        row = np.squeeze(np.asarray(Laplacian[np.ix_([i],base_idx)]))
        idx_nonzero = np.nonzero(row)[0]
        #print(idx_nonzero)
        edge_list_string = edge_list_string+["(%d,%d)"%(i, base_idx[x]) for x in idx_nonzero]
    return edge_list_string

def jaccard_sim(edge_list_g1, edge_list_g2):
    edge_set_g1 = set(edge_list_g1)
    edge_set_g2 = set(edge_list_g2)
    return len(edge_set_g1.intersection(edge_set_g2))/len(edge_set_g1.union(edge_set_g2))

def jaccard_dist(edge_list_g1, edge_list_g2):
    edge_set_g1 = set(edge_list_g1)
    edge_set_g2 = set(edge_list_g2)
    return 1- len(edge_set_g1.intersection(edge_set_g2))/len(edge_set_g1.union(edge_set_g2))


def graph_diff(Laplacian, Laplacian_est):
    if np.linalg.norm(Laplacian_est-Laplacian_est.T, 'fro') > 1e-5:
        raise ValueError("Laplacian_estimate must be symmetric")
        
    n = Laplacian.shape[0]
    n_e = Laplacian_est.shape[0]
    if n != n_e:
        raise ValueError("Dimension of Laplacian must be equal to "+ str(n)+"x" + str(n)+".")
   
    Laplacian_binary = np.sign(abs(Laplacian))
    Laplacian_est_binary = np.sign(abs(Laplacian_est))
    diff = sum(sum(abs(np.triu(Laplacian_binary, 1) - np.triu(Laplacian_est_binary, 1))))
    total_triu = n*(n-1)/2
    return (diff, diff/total_triu)

def glasso_nonzero_ratio(precision):
        
    n = precision.shape[0]
    precision_binary = np.sign(abs(precision))
    nonzeros = sum(sum(abs(np.triu(precision_binary, 0))))
    total_triu = n*(n-1)/2
    return (nonzeros, float(nonzeros)/float(total_triu))

def graph_comparison_norm(Laplacian, Laplacian_est, norm_type='fro'):
    return np.linalg.norm(Laplacian - Laplacian_est, norm_type)/np.linalg.norm(Laplacian, norm_type)



def graph_precision_recall_curve(Laplacian, Laplacian_est):
    if np.linalg.norm(Laplacian_est-Laplacian_est.T, 'fro') > 1e-5:
        raise ValueError("Laplacian_estimate must be symmetric")
        
    n = Laplacian.shape[0]
    n_e = Laplacian_est.shape[0]
    if n != n_e:
        raise ValueError("Dimension of Laplacian must be equal to "+ str(n)+"x" + str(n)+".")
    from scipy.special import betainc
    Laplacian_binary = np.triu(np.sign(abs(Laplacian)))
    #Laplacian_est[np.where(abs(Laplacian_est) > 1)] /= abs(Laplacian_est[np.where(abs(Laplacian_est) > 1)])
    Laplacian_est_binary = 1- (1-abs(Laplacian_est))**2 #betainc(0.5, 0.5, np.sign(abs(Laplacian_est))) #1- (1-abs(Laplacian_est))**2

    # compare the off-diagnoal term
    y_true = np.squeeze(Laplacian_binary[np.triu_indices(n, k=1)])
    prob_pred = np.squeeze(Laplacian_est_binary[np.triu_indices(n, k=1)])
    #print(np.squeeze(y_true).shape)
    #print(prob_pred.shape)
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pred)
    average_precision = average_precision_score(y_true, prob_pred)
    auc_pre_recall = auc(recall, precision)
    return (precision, recall, average_precision, auc_pre_recall)


def extern_influence(G, node_lists, verbose=False):
    in_all_ratio_dict = dict()
    for v in G.nodes():
        neighbors = list(G[v].keys())
        num_in = len([s for s in neighbors if s in node_lists[0]])
        num_out = len([s for s in neighbors if s in node_lists[1]])
        in_all_ratio_dict[v] = num_in/(num_out+num_in)
        if verbose: print("internal degree / all degree = %.3f" % (num_in/num_out+num_in))

    return in_all_ratio_dict

def extern_graph_draw(G, pos, intern_all_degree_ratio, save_fig=False):
    node_set_in_out = [intern_all_degree_ratio[v] for v in G.nodes()]
    fig1 = plt.figure(1)
    nodes = nx.draw_networkx_nodes(G, pos=pos, cmap=plt.get_cmap('seismic'), node_color=node_set_in_out)
    edges = nx.draw_networkx_edges(G, pos)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
    filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_intern_all_net.eps"
    if save_fig == True:
        fig1.savefig(filename, format="eps")


#===============================================================
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



    def graph_from_sparse_adjmat(self, adjMat, node_range=None, subset_range=None):
        if not isspmatrix(adjMat):
            adjMat = csr_matrix(adjMat)
        if node_range is None:
            G = nx.from_scipy_sparse_matrix(adjMat) 
            G.remove_nodes_from(nx.isolates(G)) #remove isolates
        else:
            G0 = nx.from_scipy_sparse_matrix(adjMat[np.ix_(node_range, node_range)])
            G0.remove_nodes_from(nx.isolates(G0))
            G = sorted(nx.connected_component_subgraphs(G0), key = len, reverse=True)[0]

        if subset_range is not None:
            # Find the largest connected components subgraph 
            #G1=sorted(nx.connected_component_subgraphs(G.subgraph(subset_range)), key = len, reverse=True)[0]
            G1 = nx.Graph(G.subgraph(subset_range))
            #print(len(G1))
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]

        if subset_range is None:
            return G 
        else:
            return (G, G1, node_lists, node_sets)



    def graph_build(self, size, prob, option, subset=None,  node_color=None, write_graph=False, save_fig=False):
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
 
                  ='tree', a tree whose degree follows power-law distribution
                          size = number of total nodes
                          option['gamma'] = power ratio

                  ='balanced_tree', a balanced tree with r branches and h depth
                          option['r'] = branches for each node
                          option['h'] = depth of the tree

                  ='random_bipartite_binomial', a bipartite version of the binomial (Erdős-Rényi) graph.
                          size = [n, m], n and m are node size of two clusters 
                          prob = probability of adding an edge between two clusters
                          must import nx.algorithm.bipartite
                          should be a connected graph

                  ='random_bipartite_uniform', a bipartite version of the binomial (Erdős-Rényi) graph.
                          size = [n, m], n and m are node size of two clusters 
                          option['num_edges'] = number of edges selected that connect two clusters
                          must import nx.algorithm.bipartite
                          should be a connected graph
                  
                  ='balanced_tree_add', a balanced tree with r branches and h depth and then add edges from latent vertices to a subset 
                          option['r'] = branches for each node
                          option['h'] = depth of the tree
                          option['add_edges_how']  #how to add edges
                                = "concentrate"  for each latent vertex, connects it with all other non-neighboring vertices in option['add_edges_to'] with fixed prob
                                = "uniform"      for each latent vertex, connects it with any other non-neighboring vertices in option['add_edges_to'] with i.i.d. prob
                          option['add_edges_to']  #the subset where to add edges
                                = "latent"  only add edges between two latent vertices
                                = "boundary" only add edges between latent and boundary vertices
                                = "observed" add edges between any latent and any observed vertices

                  ='balanced_tree_add_fixed', a balanced tree with r branches and h depth and then add fixed number of edges from latent vertices to a subset 
                          option['r'] = branches for each node
                          option['h'] = depth of the tree
                          option['add_edges_num'] = number of edges added for each latent vertices to a subset in option['add_edges_to']
                          option['add_edges_to']  #the subset where to add edges
                                = "latent"  only add edges between two latent vertices
                                = "boundary" only add edges between latent and boundary vertices
                                = "observed" add edges between any latent and any observed vertices
 
         option['seed'] for random seed 
         option['node_dim'] for the dimension of node attributes
        '''
        seed = option['seed']
        node_dim = option['node_dim']
        #===========================================================================
        if option['model'] == 'partition':
            if subset is not None: 
                G, G1, pos, node_lists, node_sets = self._graph_build_partition(size, prob, option, subset,  node_color, save_fig)
            else:
                G, pos = self._graph_build_partition(size, prob, option, subset,  node_color, save_fig)

        elif option['model'] == 'newman':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_newman(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_newman(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'binomial':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_binomial(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_binomial(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'power':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_power(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_power(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'grid':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_grid(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_grid(size, prob, option, subset, node_color, save_fig)
        
        elif option['model'] == 'tree':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_tree(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_tree(size, prob, option, subset, node_color, save_fig)
    
        elif option['model'] == 'balanced_tree':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_balanced_tree(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_balanced_tree(size, prob, option, subset, node_color, save_fig)
        
        elif option['model'] == 'bipartite_binomial':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_bipartite_binomial(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_bipartite_binomial(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'bipartite_uniform':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_bipartite_uniform(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_bipartite_uniform(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'balanced_tree_add':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_balanced_tree_add_edges(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_balanced_tree_add_edges(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'binomial_add':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_binomial_add_edges(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_binomial_add_edges(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'grid_add':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_grid_add_edges(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_grid_add_edges(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'balanced_tree_add_fixed':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_balanced_tree_add_fixed_edges(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_balanced_tree_add_fixed_edges(size, prob, option, subset, node_color, save_fig)

        elif option['model'] == 'grid_add_fixed':
            if subset is not None:
                G, G1, pos, node_lists, node_sets = self._graph_build_grid_add_fixed_edges(size, prob, option, subset, node_color, save_fig)
            else:
                G, pos = self._graph_build_grid_add_fixed_edges(size, prob, option, subset, node_color, save_fig)

        #===========================================================================
        G_out = nx.Graph()
        # node initialization 
        G_out.add_nodes_from(G.nodes(), attributes=np.zeros((node_dim,)).T)
        G_out.add_edges_from(G.edges())
        # assign weight values 
        for u, v, e in G_out.edges(data=True):
            e['weight'] = 1

        if subset is not None:
            G1_out = nx.Graph()
             # node initialization 
            G1_out.add_nodes_from(G1.nodes(), attributes=np.zeros((node_dim,)).T)
            G1_out.add_edges_from(G1.edges())
            # assign weight values 
            for u, v, e in G1_out.edges(data=True):
                e['weight'] = 1


        if write_graph:
            self.G = G_out
            if subset is not None:
                self.G1= G1_out
            self.option = option.copy()
        if subset is None:
            return (G_out, pos)
        else: 
            return (G_out, G1_out, pos, node_lists, node_sets)


    def _graph_build_partition(self, size, prob, option, subset=None,  node_color=None, save_fig=False):
        seed = option['seed']
        if len(prob) < 2:
            p_in = prob
            p_out = 1 - p_in
        
        [p_in, p_out] = prob
        G = nx.random_partition_graph(sizes=size, p_in=p_in, p_out=p_out, seed=seed, directed=False)
        pos = nx.nx_pydot.graphviz_layout(G)
        # choose sub-network
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]

        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size[0]+['b']*size[1]
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)

    
    def _graph_build_newman(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        try:
            k = option['k-NN']
        except KeyError:
            k = 2

        if type(size) == list:
            size = sum(size)

        G=  nx.newman_watts_strogatz_graph(size, k=k, p=prob, seed=seed)
        pos = nx.circular_layout(G, dim=2, scale=1.0, center=None)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]

        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'

        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_binomial(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        if prob <= 0.2:
            G = nx.fast_gnp_random_graph(size, prob, seed=seed)
        else:
            G = nx.gnp_random_graph(size, prob, seed=seed)  
        
        if not nx.is_connected(G): #must be connected
            raise ValueError("Not connected. Please increase the edge probability.")

        if type(size) == list:
            size = sum(size)
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)

    def _graph_build_power(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        G = nx.powerlaw_cluster_graph(n=size[0], m=size[1], p=prob, seed=seed)
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_grid(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        if type(size) == int:
            size = [size, size]
        G = nx.grid_2d_graph(m=size[0], n=size[1])
        pos = dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
        if subset is not None:
            xpos, ypos = np.meshgrid(subset[0], subset[1])
            xpos= xpos.reshape((xpos.size,))
            ypos= ypos.reshape((ypos.size,))
            node_subset = list(zip(xpos, ypos))
            #node_subset = [G.nodes()[i] for i in subset]
            #print(node_subset)
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*sum(size)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='k')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)

    
    def _graph_build_tree(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        try:
            gamma = option['gamma']
        except KeyError:
            gamma = 3
        tries = 10000
        G = nx.random_powerlaw_tree(n=size, gamma=gamma, seed=seed, tries=tries)
        pos = nx.circular_layout(G, dim=2, scale=1.0, center=None) #nx.shell_layout(G)#nx.spring_layout(G) #nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = subset #[G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_balanced_tree(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        try:
            r = option['r']
        except KeyError:
            r = 2
        try:
            h = option['h']
        except KeyError:
            h = 3
        G = nx.balanced_tree(r=r, h=h, create_using=nx.Graph())
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*len(G)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize= 8, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_bipartite_binomial(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        if type(size) == int:
            size = [10, 10]
        G = nx.algorithms.bipartite.random_graph(size[0], size[1], prob, seed=seed, directed=False)
        if subset is None:
            node_sets = nx.algorithms.bipartite.sets(G)
            G1 = nx.subgraph(node_sets[0])
        else:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
        node_lists = [list(node_sets[0]), list(node_sets[1])]
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        if not nx.is_connected(G): #must be connected
            raise ValueError("Not connected. Please increase the edge probability.")
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size[0]+['b']*size[1]
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize= 8, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)

    def _graph_build_bipartite_uniform(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        if type(size) == int:
            size = [10, 10]

        try:
            num_edges = option['num_edges']
        except KeyError:
            num_edges = size[0]*size[1]
 
        if num_edges > size[0]*size[1]:
            print("too many edges. reduce to " + str(size[0]*size[1]))
            num_edges = size[0]*size[1]

        G = nx.algorithms.bipartite.gnmk_random_graph(size[0], size[1], num_edges, seed=seed, directed=False)
        if subset is None:
            node_sets = nx.algorithms.bipartite.sets(G)
            G1 = nx.subgraph(node_sets[0])
        else:
            node_subset = [G.nodes()[i] for i in subset]
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
        node_lists = [list(node_sets[0]), list(node_sets[1])]
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        if not nx.is_connected(G): #must be connected
            raise ValueError("Not connected. Please increase the edge probability.")
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size[0]+['b']*size[1]
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize= 8, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_balanced_tree_add_edges(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        try:
            r = option['r']
        except KeyError:
            r = 2
        try:
            h = option['h']
        except KeyError:
            h = 3
        try:
            choice_add_edges_how = option['add_edges_how']  #where to add edges
        except KeyError:
            choice_add_edges_how = 'uniform'

        try:
            choice_add_edges_to = option['add_edges_to']  #how to add edges
        except KeyError:
            choice_add_edges_to = 'latent'


        G = nx.balanced_tree(r=r, h=h, create_using=nx.Graph())
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G_tmp=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            latent_vertices = list(set(G.nodes()).difference(G_tmp.nodes()))
            boundary_vertices = [v for v in G_tmp.nodes() if set(list(G[v].keys())).intersection(set(latent_vertices))]
            np.random.seed(seed)
            dices = np.random.rand(len(G), len(G))
            for i, v in enumerate(latent_vertices):
                neighbors = G[v]
                latent_not_neighbors = [s for s in latent_vertices if s not in neighbors]
                observed_not_neighbors = [s for s in node_subset if s not in neighbors]
                boundary_not_neighbors = [s for s in boundary_vertices if s not in neighbors]
                if choice_add_edges_to == 'observed':
                    for j, u in enumerate(observed_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'latent':
                    for j, u in enumerate(latent_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'boundary':
                    for j, u in enumerate(boundary_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*len(G)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize= 8, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_binomial_add_edges(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        if prob <= 0.2:
            G = nx.fast_gnp_random_graph(size, prob, seed=seed)
        else:
            G = nx.gnp_random_graph(size, prob, seed=seed)  
        
        if not nx.is_connected(G): #must be connected
            raise ValueError("Not connected. Please increase the edge probability.")
        try:
            choice_add_edges_how = option['add_edges_how']   #how to add edges
        except KeyError:
            choice_add_edges_how = 'uniform'
        try:
            choice_add_edges_to = option['add_edges_to']   #where to add edges
        except KeyError:
            choice_add_edges_to = 'latent'

        if type(size) == list:
            size = sum(size)
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            prob2 = option['latent_edges_prob']

            node_subset = [G.nodes()[i] for i in subset]
            G_tmp=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            latent_vertices = list(set(G.nodes()).difference(G_tmp.nodes()))
            boundary_vertices = [v for v in G_tmp.nodes() if set(list(G[v].keys())).intersection(set(latent_vertices))]
            #latent_vertices = list(set(G.nodes()).difference(node_subset))
            np.random.seed(seed)
            dices = np.random.rand(len(G), len(G))
            for i, v in enumerate(latent_vertices):
                neighbors = G[v]
                latent_not_neighbors = [s for s in latent_vertices if s not in neighbors]
                observed_not_neighbors = [s for s in node_subset if s not in neighbors]
                boundary_not_neighbors = [s for s in boundary_vertices if s not in neighbors]
                if choice_add_edges_to == 'observed':
                    for j, u in enumerate(observed_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'latent':
                    for j, u in enumerate(latent_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'boundary':
                    for j, u in enumerate(boundary_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob2:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*size
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)


    def _graph_build_grid_add_edges(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        try:
            choice_add_edges_how = option['add_edges_how'] # how to add edges
        except KeyError:
            choice_add_edges_how = 'uniform'
        try:
            choice_add_edges_to = option['add_edges_to']  #where to add edges
        except KeyError:
            choice_add_edges_to = 'latent'

        if type(size) == int:
            size = [size, size]
        G = nx.grid_2d_graph(m=size[0], n=size[1])
        pos = dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
        if subset is not None:
            #node_subset = subset #[G.nodes()[i] for i in subset]
            xpos, ypos = np.meshgrid(subset[0], subset[1])
            xpos= xpos.reshape((xpos.size,))
            ypos= ypos.reshape((ypos.size,))
            node_subset = list(zip(xpos, ypos))
            G_tmp=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            latent_vertices = list(set(G.nodes()).difference(G_tmp.nodes()))
            boundary_vertices = [v for v in G_tmp.nodes() if set(list(G[v].keys())).intersection(set(latent_vertices))]
            #latent_vertices = list(set(G.nodes()).difference(node_subset))
            np.random.seed(seed)
            dices = np.random.rand(len(G), len(G))
            for i, v in enumerate(latent_vertices):
                neighbors = G[v]
                latent_not_neighbors = [s for s in latent_vertices if s not in neighbors]
                observed_not_neighbors = [s for s in node_subset if s not in neighbors]
                boundary_not_neighbors = [s for s in boundary_vertices if s not in neighbors]
                if choice_add_edges_to == 'observed':
                    for j, u in enumerate(observed_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'latent':
                    for j, u in enumerate(latent_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                elif choice_add_edges_to == 'boundary':
                    for j, u in enumerate(boundary_not_neighbors):
                        if choice_add_edges_how == 'uniform':
                            if dices[i][j] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
                        elif choice_add_edges_how == 'concentrate':
                            if dices[i][0] < prob:
                                G.add_edge(v,u)
                                G.add_edge(u,v)
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*sum(size)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='k')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)



    def _graph_build_balanced_tree_add_fixed_edges(self, size, prob, option, subset=None, node_color=None, save_fig=False):  
        seed = option['seed']
        try:
            r = option['r']
        except KeyError:
            r = 2
        try:
            h = option['h']
        except KeyError:
            h = 3

        try:
            choice_add_edges_to = option['add_edges_to']  #how to add edges
        except KeyError:
            choice_add_edges_to = 'latent'

        try:
            choice_add_edges_num = option['add_edges_num']
        except KeyError:
            choice_add_edges_num = 3


        G = nx.balanced_tree(r=r, h=h, create_using=nx.Graph())
        pos = nx.nx_pydot.graphviz_layout(G)
        if subset is not None:
            node_subset = [G.nodes()[i] for i in subset]
            G_tmp=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            latent_vertices = list(set(G.nodes()).difference(G_tmp.nodes()))
            boundary_vertices = [v for v in G_tmp.nodes() if set(list(G[v].keys())).intersection(set(latent_vertices))]
            if choice_add_edges_num != -1:
                for i, v in enumerate(latent_vertices):
                    neighbors = G[v]
                    latent_not_neighbors = [s for s in latent_vertices if s not in neighbors]
                    observed_not_neighbors = [s for s in node_subset if s not in neighbors]
                    boundary_not_neighbors = [s for s in boundary_vertices if s not in neighbors]
                    if choice_add_edges_to == 'observed':
                        count = min(choice_add_edges_num, len(observed_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        observed_shuffle = list(observed_not_neighbors)
                        np.random.shuffle(observed_shuffle)
                        for u in observed_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
                    elif choice_add_edges_to == 'latent':
                        count = min(choice_add_edges_num, len(latent_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        latent_shuffle = list(latent_not_neighbors)
                        np.random.shuffle(latent_shuffle)
                        for u in latent_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
                    elif choice_add_edges_to == 'boundary':
                        count = min(choice_add_edges_num, len(boundary_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        boundary_shuffle = list(boundary_not_neighbors)
                        np.random.shuffle(boundary_shuffle)
                        for u in boundary_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*len(G)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize= 8, node_color=node_color, font_color='w')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)



    def _graph_build_grid_add_fixed_edges(self, size, prob, option, subset=None, node_color=None, save_fig=False):
        seed = option['seed']
        try:
            choice_add_edges_to = option['add_edges_to']  #where to add edges
        except KeyError:
            choice_add_edges_to = 'latent'
        try:
            choice_add_edges_num = option['add_edges_num']
        except KeyError:
            choice_add_edges_num = 3

        if type(size) == int:
            size = [size, size]
        G = nx.grid_2d_graph(m=size[0], n=size[1])
        pos = dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
        if subset is not None:
            #node_subset = subset #[G.nodes()[i] for i in subset]
            xpos, ypos = np.meshgrid(subset[0], subset[1])
            xpos= xpos.reshape((xpos.size,))
            ypos= ypos.reshape((ypos.size,))
            node_subset = list(zip(xpos, ypos))
            G_tmp=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            latent_vertices = list(set(G.nodes()).difference(G_tmp.nodes()))
            boundary_vertices = [v for v in G_tmp.nodes() if set(list(G[v].keys())).intersection(set(latent_vertices))]
            #latent_vertices = list(set(G.nodes()).difference(node_subset))
            if choice_add_edges_num != -1:
                for i, v in enumerate(latent_vertices):
                    neighbors = G[v]
                    latent_not_neighbors = [s for s in latent_vertices if s not in neighbors]
                    observed_not_neighbors = [s for s in node_subset if s not in neighbors]
                    boundary_not_neighbors = [s for s in boundary_vertices if s not in neighbors]
                    if choice_add_edges_to == 'observed':
                        count = min(choice_add_edges_num, len(observed_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        observed_shuffle = list(observed_not_neighbors)
                        np.random.shuffle(observed_shuffle)
                        for u in observed_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
                    elif choice_add_edges_to == 'latent':
                        count = min(choice_add_edges_num, len(latent_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        latent_shuffle = list(latent_not_neighbors)
                        np.random.shuffle(latent_shuffle)
                        for u in latent_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
                    elif choice_add_edges_to == 'boundary':
                        count = min(choice_add_edges_num, len(boundary_not_neighbors))
                        if count == 0: continue
                        np.random.seed(seed)
                        boundary_shuffle = list(boundary_not_neighbors)
                        np.random.shuffle(boundary_shuffle)
                        for u in boundary_shuffle[:count]:
                            G.add_edge(v,u)
                            G.add_edge(u,v)
                        seed +=1
            G1=sorted(nx.connected_component_subgraphs(G.subgraph(node_subset)), key = len, reverse=True)[0]
            node_sets = []
            node_sets.append(set(G1.nodes()))
            node_sets.append(set(G.nodes()).difference(set(G1.nodes())))
            node_lists = [list(node_sets[0]), list(node_sets[1])]
        fig1 = plt.figure(1)
        if node_color is None and subset is None:
            node_color = ['r']*sum(size)
        elif subset is not None:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'
        nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize= 10, node_color=node_color, font_color='k')
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
        if save_fig == True:
            fig1.savefig(filename, format="eps")
        if subset is not None:
            return (G, G1, pos, node_lists, node_sets)
        else:
            return (G, pos)



#==========================================================================================================================================
    def latent_graph_signal_generate(self, G, G1, node_lists, node_sets, option, nodeIdx=None, sigma=1, show_plot=False, save_fig=False, write_data=False):
        '''
            generate observed data on sub-graph G1 from latent vertices

             X1 = A*X2 + z
             
             where 
                  Z, [n_observed, dim], \sim N(0, S*-1) where S*-1 = L_sub, define the local conditional structure
                  X2, [n_latent, dim],  \sim N(0, sigma*I), independent random vector on latent variables
                  A, [n_observed, n_latent], define the transformation from Z to X2

                  choose A[i,j] = exp(-kernel_sigma*shortest_path_length(u_i, v_j))
              

        '''
        seed = option['seed']
        dim  = option['node_dim']
        try:
            kernel_sigma = option['exp_kernel_sigma']
        except KeyError:
            kernel_sigma = 0.5

        Laplacian = nx.normalized_laplacian_matrix(G, weight=None).todense()
        Adjacency = nx.adjacency_matrix(G, weight=None).todense()

        observed_vertex_set = node_sets[0]
        latent_vertex_set = node_sets[1]
        if nodeIdx is None:
            _, nodeIdx = self.get_node_attributes(G)
        observed_idx = [item['loc'] for item in nodeIdx if item['node'] in node_sets[0]]
        hidden_idx = [item['loc'] for item in nodeIdx if item['node'] in node_sets[1]]
        
        Laplacian_o = Laplacian[np.ix_(observed_idx, observed_idx)] #- np.diag([len([u for u in G[v] if u in latent_vertex_set]) for v in observed_vertex_set ])
        eigvals_L1, eigvecs_L1 = np.linalg.eigh(Laplacian_o)
        eigvecs_L1 = np.asarray(eigvecs_L1)
        try:
            eps = option['eps']
        except KeyError:
            eps = 1e-4

        nonzeros_indices = np.argwhere(eigvals_L1> 1e-4)
        transformed_eigvals_L1 = eigvals_L1.copy()
            #print(eigval[nonzeros_indices])
        transformed_eigvals_L1[nonzeros_indices] = 1/(eps + transformed_eigvals_L1[nonzeros_indices])
        # covariance of Z 

        Covariance_z = np.dot(transformed_eigvals_L1*eigvecs_L1, eigvecs_L1.T)
        #print(Covariance_z.shape)
        # find all-pair shortest path length 
        shortest_path_all = nx.all_pairs_dijkstra_path_length(G)
        K = np.zeros((len(G), len(G)))
        for item in nodeIdx:
            v = item['node']
            ii = item['loc']
            shortest_path_v = shortest_path_all[v]
            for u, shortest_path_uv in shortest_path_v.items():
                jj = min([item2['loc'] for item2 in nodeIdx if item2['node'] == u ])
                K[ii, jj] = np.exp(- kernel_sigma*shortest_path_uv)
        # define the transformation matrix from latent to observed data
        Transform_matrix = np.ascontiguousarray(K[np.ix_(observed_idx, hidden_idx)])
        # generate observed data
        if seed is not None:
            np.random.seed(seed)
        X2 = sigma*np.random.randn(len(hidden_idx), dim)
        if seed is not None:
            np.random.seed((seed+1000)%91)
        Z =  np.random.multivariate_normal(np.zeros((len(observed_idx),)), Covariance_z, dim).T
        #print(Z.shape)
        X1 = np.dot(Transform_matrix, X2) + Z

        X = np.zeros((len(G), dim))

        # write to graph
        for item in nodeIdx:
            v = item['node']
            ii = item['loc']
            if ii in hidden_idx:
                jj = hidden_idx.index(ii)
                X[ii,:] = X2[jj, :]
                G.node[v]['attributes'] = X[ii,:]
            elif ii in observed_idx:
                jj = observed_idx.index(ii)
                X[ii,:] = X1[jj, :]
                G.node[v]['attributes'] = X[ii,:]
        # check 
        if np.linalg.norm(X[observed_idx,:] - X1) > 1e-3:
            raise ValueError("Index not correct!")

        if np.linalg.norm(X[hidden_idx,:] - X2) > 1e-3:
            raise ValueError("Index not correct!")



        if write_data:
            self.G = G.copy()
            self.X = np.copy(X)
            self.ifwrite = True
        else:
            self.ifwrite = False 


        return (G, X, X1, X2)
        




    def smooth_gsignal_generate(self, G, T, sigma, alpha=0, seed=1000, add_noise=False, write_data=False, option=None, show_plot=False):
        '''
           generate the node attributes in the graph associated with the network topology.
         G = Graph with "attributes" data for each node 
         T = maximum loops of smoothing 
         sigma = initialized noise-level 

        '''
    
        #initialization
        n = len(G)
        dim = len(G.node[G.nodes()[0]]['attributes'])
        X = np.ndarray((n,dim))

        if alpha > 1 :
            alpha = 1- np.exp(-alpha)
        elif alpha < 0:
            alpha = np.exp(alpha)
          
        np.random.seed(seed)
        X_r = np.zeros((n,dim)) #sigma*np.random.randn(n, dim)/np.sqrt(n)
        #X_r = normalize(X_r, norm='l1', axis=0)

        for i, idx in enumerate(G.nodes()):
            #dim = len(G.node[idx]['attributes'])
            G.node[idx]['attributes'] = X_r[i,:]
            G.node[idx]['attributes'][0] = 1/np.sqrt(n)
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
        
        if T > 5:
            sepT = int(T/5)
        else:
            sepT = 1
        hist_tv = np.zeros((T,))
        # run a smoothing process on network
        for t in range(T+dim):
            tempG = G.copy()
            for i, idx in enumerate(G.nodes()):
                if len(tempG[idx]) != 0:
                    if t <= T:
                         G.node[idx]['attributes'][0] = (1-alpha)*sum([tempG.node[neighbor]['attributes'][0]/len(tempG[neighbor]) for neighbor in tempG[idx]]) + alpha*tempG.node[idx]['attributes'][0]
                         if add_noise:
                             G.node[idx]['attributes'][0] += sigma*np.random.randn(1)/np.sqrt(n) 
                    else:
                         G.node[idx]['attributes'][t-T] = (1-alpha)*sum([tempG.node[neighbor]['attributes'][t-T-1]/len(tempG[neighbor]) for neighbor in tempG[idx]]) + alpha*tempG.node[idx]['attributes'][t-T-1]
                         
            
            # avoid all one features
            tempG2 = G.copy()
            for i, idx in enumerate(G.nodes()):
                if t <= T:
                    G.node[idx]['attributes'][0] = tempG2.node[idx]['attributes'][0] - (sum([tempG2.node[allnodes]['attributes'][0] for allnodes in tempG2])/n)
                else:
                    G.node[idx]['attributes'][t-T] = tempG2.node[idx]['attributes'][t-T] - (sum([tempG2.node[allnodes]['attributes'][t-T] for allnodes in tempG2])/n)

    
            for i, d in G.nodes_iter(data=True):
                X[i,:] = d['attributes']

            #X = normalize(X, norm='l1', axis=0)          
            hist_tv[t] = self.compute_total_variation(G)

            if t%sepT == 0 and show_plot == True:
                plt.figure(1)
                print("Iteration: " + str(t+1))
                plt.subplot(str(int(T/sepT)+1)+"1"+str(int(t/sepT)+2))
                plt.stem(np.arange(X.shape[0]),X[:,0])
                plt.axis([0, 40, -6, 8])
    
        if show_plot == True:        
            plt.show()        

        if write_data == True:
            if option is None:
                raise ValueError('Option for graph generator must not be none.')
            self.G = G.copy()
            self.X = X.copy()
            self.option = option.copy()
            self.hist_tv = np.copy(hist_tv)
            self.ifwrite = True
        else:
            self.ifwrite = False 

        return [G, X, hist_tv]



    def smooth_gsignal_filter(self, G, option, sigma, seed=1000, add_noise=False, show_plot=False, save_fig=False, write_data =False):
        '''
             Apply graph filtering to initial random graph signal X0
                    X = T*X0
             where T is a linear operator based on either adjacency matrix or graph laplacian matrix

             T = U*g(Sigma)*U.T for eigendecomposition of matrix basis = U*Sigma*U.T 

             option['mat']
                       = 'adjacency_matrix': use Adjacency matrix as the matrix basis
                       = 'laplacian_matrix': use Normalized Graph Laplacian matrix as the matrix basis

             option['method']
                       = 'l0_threshold': enforce all eigval < option['threshold'] to be zero
                       = 'l1_threshold': apply soft-thresholding to all eigval < option['threshold']
                                         i.e. new_eigval = [eigval - threshold]_{+}
                       = 'polynomial':  
                                  define coeffs = [coeff0, coeff1, coeff2, ..., coeffd] for degree d polynomial
                                        coff0 + coeff1*x + coeff2*x**2 + ... + coffd*x**d
                                  apply polynomial to each eigval 
                       = 'sigmoid':
                                   eigval*1/(1+exp(rate(i-b)))-bias

                       = 'l0_renomalize': apply l0_threshold then add sum(residual_energy) to each remaining eigval
                       = 'rescale': new_eigval =  option['weights'] * eigval
                       = 'sigmoid_threshold':  eigma_new = soft_theshold(eigma_old*1/(1+exp(rate*(i+1-shift))), bias)
                       = 'inverse_sqrt': eigma_new = 1/sqrt(eigma_old+ epsilon) for eigma_old
                       = 'inverse_poly'
                                        coff0 + coeff1*x**-1 + coeff2*x**-2 + ... + coffd*x**-d

             return Graph G, transformed data X, and initial data X_r

        '''
        n = len(G)
        dim = len(G.node[G.nodes()[0]]['attributes'])
        X = np.ndarray((n,dim))
          
        np.random.seed(seed)
        X_r = sigma*np.random.randn(n, dim)/np.sqrt(n)
        #X_r = normalize(X_r, norm='l1', axis=0)
        for i, idx in enumerate(G.nodes()):
            #dim = len(G.node[idx]['attributes'])
            G.node[idx]['attributes'] = X_r[i,:]
            X[i,:] = G.node[idx]['attributes']
        
        #tempG2 = G.copy()
        #for i, idx in enumerate(G.nodes()):   
        #    G.node[idx]['attributes'] = tempG2.node[idx]['attributes'] - sum([tempG2.node[allnodes]['attributes'] for allnodes in tempG2])/n
        #    X[i,:] = G.node[idx]['attributes']
        #tempG2.clear()    

        transformed_eigvals, eigvecs, _ = self.graph_filter(G, X_r, option, show_plot, save_fig)
        #X = np.asarray(np.dot(np.dot(eigvecs, np.diag(transformed_eigvals)), np.dot(eigvecs.T, X)))
        X = np.dot(transformed_eigvals*eigvecs, np.dot(eigvecs.T, X))
        for i, idx in enumerate(G.nodes()):
            #dim = len(G.node[idx]['attributes'])
            G.node[idx]['attributes'] = X[i,:]

        if write_data:
            self.G = G.copy()
            self.X = np.copy(X)
            self.ifwrite = True
        else:
            self.ifwrite = False 
        return (G, X, X_r)



    def graph_filter(self, G, X0, option, show_plot=False, save_fig=False):
        '''
             Apply graph filtering to initial random graph signal X0
                    X = T*X0
             where T is a linear operator based on either adjacency matrix or graph laplacian matrix

             T = U*g(Sigma)*U.T for eigendecomposition of matrix basis = U*Sigma*U.T 

             option['mat']
                       = 'adjacency_matrix': use Adjacency matrix as the matrix basis
                       = 'laplacian_matrix': use Normalized Graph Laplacian matrix as the matrix basis

             option['method']
                       = 'l0_threshold': enforce all eigval < option['threshold'] to be zero
                       = 'l1_threshold': apply soft-thresholding to all eigval < option['threshold']
                                         i.e. new_eigval = [eigval - threshold]_{+}
                       = 'polynomial':  
                                  define coeffs = [coeff0, coeff1, coeff2, ..., coeffd] for degree d polynomial
                                        coff0 + coeff1*x + coeff2*x**2 + ... + coffd*x**d
                                  apply polynomial to each eigval 
                       = 'sigmoid':
                                   eigval*1/(1+exp(rate(i-b)))-bias

                       = 'l0_renomalize': apply l0_threshold then add sum(residual_energy) to each remaining eigval
                       = 'rescale': new_eigval =  option['weights'] * eigval
                       = 'sigmoid_threshold':  eigma_new = soft_theshold(eigma_old*1/(1+exp(rate*(i+1-shift))), bias)
                       = 'inverse_sqrt': eigma_new = 1/sqrt(eigma_old+ epsilon) for eigma_old
                       = 'inverse_poly'
                                        coff0 + coeff1*x**-1 + coeff2*x**-2 + ... + coffd*x**-d

        '''
        if option['mat'] == 'adjacency_matrix':
            Mat = nx.adjacency_matrix(G, weight=None).todense()
            eigval_, eigvec_ = np.linalg.eigh(Mat)
            #for adjacency matrix in decreasing order
            eig_index = np.argsort(abs(eigval_))[::-1]            
            eigval = eigval_[eig_index]
            eigvec = eigvec_[:, eig_index]

        elif option['mat'] == 'laplacian_matrix':
            Mat = nx.normalized_laplacian_matrix(G, weight=None).todense()
            eigval_, eigvec_ = np.linalg.eigh(Mat)
            #for laplacian matrix in increasing order
            eig_index = np.argsort(eigval_)
            eigval = eigval_[eig_index]
            # find the inverse of laplacian
            #eigval[1:len(eigval)] = 1/eigval[1:len(eigval)]
            eigvec = eigvec_[:, eig_index]
            

        n = len(G)
        dim = X0.shape[1]#len(G.node[G.nodes()[0]]['attributes'])

        if X0.shape[0] != n: #or X0.shape[1] != dim:
            X0 = np.random.randn(n, dim)
        
        if option['method'] == 'l0_threshold':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            import pywt
            transformed_eigval = pywt.threshold(eigval, tau, 'hard')

        elif option['method'] == 'l1_threshold':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            import pywt
            transformed_eigval = pywt.threshold(eigval, tau, 'soft')

        elif option['method'] == 'polynomial':
            try:
                coeffs = option['coeffs']
            except KeyError:
                coeffs = [0,1]
            def poly_fit(coeffs, sig):
                return sum([p*(sig**i) for i, p in enumerate(coeffs)])
            transformed_eigval = poly_fit(coeffs, eigval)

        elif option['method'] == 'sigmoid_threshold':
            try:
                rate = option['rate']
            except:
                rate = 1
           
            try:
                shift = option['shift']
            except:
                shift = 0

            try:
                bias = option['bias']
            except:
                bias = 0

            import pywt
            def sigmoid(x, rate, shift):
                return 1/(1+np.exp(rate*(x - shift)))
            
            transformed_eigval = pywt.threshold(eigval*sigmoid(1+np.arange(len(eigval)), rate, shift), bias, 'soft')

          
        elif option['method'] == 'l0_renormalize':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            index_less_l0 = np.where(eigval < tau)
            index_more_l0 = np.where(eigval >= tau)
            res = sum(eigval[index_less_l0])/len(eigval[index_more_l0])
            transformed_eigval = eigval.copy()
            transformed_eigval[index_less_l0] = 0
            transformed_eigval[index_more_l0] += res

        elif option['method'] == 'rescale':
            try:
               weights = option['weights']            
            except KeyError:
               weights = np.ones((len(eigval),))

            transformed_eigval = weights * eigval
        
        elif option['method'] == 'inverse_sqrt':
            try:
               eps = option['eps']
            except KeyError:
               eps = 1e-4

            nonzeros_indices = np.argwhere(eigval> 1e-4)
            transformed_eigval = eigval.copy()
            #print(eigval[nonzeros_indices])
            transformed_eigval[nonzeros_indices] = 1/np.sqrt(eps + transformed_eigval[nonzeros_indices])
            #transformed_eigval[eigval.argmin()] = 0

        elif option['method'] == 'inverse_poly':
            try:
                coeffs = option['coeffs']
            except KeyError:
                coeffs = [0,1]
            def poly_fit(coeffs, sig):
                return sum([p*(sig**i) for i, p in enumerate(coeffs)])
            nonzeros_indices = np.argwhere(eigval> 1e-4)
            transformed_eigval = eigval.copy()
            #print(eigval[nonzeros_indices])
            transformed_eigval[nonzeros_indices] = 1/transformed_eigval[nonzeros_indices]
            transformed_eigval = poly_fit(coeffs, transformed_eigval)

        if show_plot:
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(121)
            (markerline, stemlines, baseline) = plt.stem(np.arange(len(eigval)), eigval, 'b', basefmt='k-')
#            plt.plot(np.arange(len(eigval)), np.ones((len(eigval, ))), '-r')
#            if option['mat'] == 'adjacency_matrix':
#               plt.plot(np.arange(len(eigval)), -np.ones((len(eigval, ))), '-r')
            plt.xlabel('rank of eigenvalue')
            plt.ylabel('eigenvalue')
            ax1.grid(True)

            ax2 = fig.add_subplot(122)
            (markerline, stemlines, baseline) = plt.stem(np.arange(len(transformed_eigval)), transformed_eigval, 'b', basefmt='k-')
#            plt.plot(np.arange(len(eigval)), np.ones((len(eigval, ))), '-r')
            plt.xlabel('rank of eigenvalue')
            plt.ylabel('eigenvalue')
            ax2.grid(True)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_transform.eps"
            if save_fig : fig.savefig(filename)

        return (transformed_eigval, np.asarray(eigvec), eigval) 


    def eigen_transform(self, Mat, option, show_plot=False, save_fig=False):
        eigval_, eigvec_ = np.linalg.eigh(Mat)
            #for laplacian matrix in increasing order
        eig_index = np.argsort(eigval_)
        eigval = eigval_[eig_index]
            # find the inverse of laplacian
            #eigval[1:len(eigval)] = 1/eigval[1:len(eigval)]
        eigvec = eigvec_[:, eig_index]
            

        n = Mat.shape[0]
        
        if option['method'] == 'l0_threshold':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            import pywt
            transformed_eigval = pywt.threshold(eigval, tau, 'hard')

        elif option['method'] == 'l1_threshold':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            import pywt
            transformed_eigval = pywt.threshold(eigval, tau, 'soft')

        elif option['method'] == 'polynomial':
            try:
                coeffs = option['coeffs']
            except KeyError:
                coeffs = [0,1]
            def poly_fit(coeffs, sig):
                return sum([p*(sig**i) for i, p in enumerate(coeffs)])
            transformed_eigval = poly_fit(coeffs, eigval)

        elif option['method'] == 'sigmoid_threshold':
            try:
                rate = option['rate']
            except:
                rate = 1
           
            try:
                shift = option['shift']
            except:
                shift = 0

            try:
                bias = option['bias']
            except:
                bias = 0

            import pywt
            def sigmoid(x, rate, shift):
                return 1/(1+np.exp(rate*(x - shift)))
            
            transformed_eigval = pywt.threshold(eigval*sigmoid(1+np.arange(len(eigval)), rate, shift), bias, 'soft')

          
        elif option['method'] == 'l0_renormalize':
            try:
                tau = option['threshold'] 
            except KeyError:
                tau = 1
            index_less_l0 = np.where(eigval < tau)
            index_more_l0 = np.where(eigval >= tau)
            res = sum(eigval[index_less_l0])/len(eigval[index_more_l0])
            transformed_eigval = eigval.copy()
            transformed_eigval[index_less_l0] = 0
            transformed_eigval[index_more_l0] += res

        elif option['method'] == 'rescale':
            try:
               weights = option['weights']            
            except KeyError:
               weights = np.ones((len(eigval),))

            transformed_eigval = weights * eigval
        
        elif option['method'] == 'inverse_sqrt':
            try:
               eps = option['eps']
            except KeyError:
               eps = 1e-4

            nonzeros_indices = np.argwhere(eigval> 1e-4)
            transformed_eigval = eigval.copy()
            #print(eigval[nonzeros_indices])
            transformed_eigval[nonzeros_indices] = 1/np.sqrt(eps + transformed_eigval[nonzeros_indices])
            #transformed_eigval[eigval.argmin()] = 0

        elif option['method'] == 'inverse_poly':
            try:
                coeffs = option['coeffs']
            except KeyError:
                coeffs = [0,1]
            def poly_fit(coeffs, sig):
                return sum([p*(sig**i) for i, p in enumerate(coeffs)])
            nonzeros_indices = np.argwhere(eigval> 1e-4)
            transformed_eigval = eigval.copy()
            #print(eigval[nonzeros_indices])
            transformed_eigval[nonzeros_indices] = 1/transformed_eigval[nonzeros_indices]
            transformed_eigval = poly_fit(coeffs, transformed_eigval)

        if show_plot:
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(121)
            (markerline, stemlines, baseline) = plt.stem(np.arange(len(eigval)), eigval, 'b', basefmt='k-')
#            plt.plot(np.arange(len(eigval)), np.ones((len(eigval, ))), '-r')
#            if option['mat'] == 'adjacency_matrix':
#               plt.plot(np.arange(len(eigval)), -np.ones((len(eigval, ))), '-r')
            plt.xlabel('rank of eigenvalue')
            plt.ylabel('eigenvalue')
            ax1.grid(True)

            ax2 = fig.add_subplot(122)
            (markerline, stemlines, baseline) = plt.stem(np.arange(len(transformed_eigval)), transformed_eigval, 'b', basefmt='k-')
#            plt.plot(np.arange(len(eigval)), np.ones((len(eigval, ))), '-r')
            plt.xlabel('rank of eigenvalue')
            plt.ylabel('eigenvalue')
            ax2.grid(True)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_transform.eps"
            if save_fig : fig.savefig(filename)

        return (transformed_eigval, np.asarray(eigvec), eigval) 


    def get_normalized_laplacian(self, G, weight=None):
        laplacian = nx.normalized_laplacian_matrix(G, weight=weight).todense()
        return np.asarray(laplacian)


    def eigsh_laplacian(self, G, k=5):
        if k > len(G):
            k = len(G)
        laplacian = nx.normalized_laplacian_matrix(G)
        eigval, eigvec = eigsh(laplacian, k, which='SM')
        return (eigval, eigvec)#[:, [1:len(eigval)]])

    def graph_fourier_transform(self, G, X, show_fig=False, save_fig=False, overwrite=False):
        
        _, U_k = self.eigsh_laplacian(G, self.k_u)
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






    def inference_hidden_graph_regul(self, G,  sigma, init_X=None):
        # find the latent variables in factor analysis model
        #   h = arg min 0.5*|| x - U*h||**2 + (0.5/sigma**2)*h.T*Lambda*h 
        if init_X is not None:
            X = init_X
        else: 
            if self.ifwrite:
                X = self.X
            else:
                X = np.random.randn(len(G), self.node_dim)
    

        lambda_k, U_k = self.eigsh_laplacian(G, self.k_u)#[:, 0:self.k_u]
        shinkage = lambda_k/sigma**2 + np.ones(lambda_k.shape)

        h = (np.dot(U_k.T, X).T/shinkage).T
        return h


    def get_node_attributes(self, G):
        n = len(G)
        dim = len(G.node[G.nodes()[0]]['attributes'])
        X = np.zeros((n,dim))
        idx_x = 0
        for i, data in G.nodes_iter(data=True):
            X[idx_x,:] = data['attributes']
            idx_x += 1

        nodeIdx = [{'node': idx, 'loc' : i} for i, idx in enumerate(G.nodes())]
        return [X, nodeIdx]


    def compute_laplacian_smoothness(self, G):
        total_diff = 0
        m = G.size()
        sum_weight = 0
        for node1, node2, data in G.edges_iter(data=True):
            total_diff = total_diff + data['weight']*np.linalg.norm(G.node[node1]['attributes']- G.node[node2]['attributes'])**2
            sum_weight = sum_weight + data['weight']
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



    def draw(self, G, option, node_lists=None, pos=None, pos_init=None, with_labels=False, fontsize=8, font_color = 'k', node_size = 250, save_fig = False):
        seed = option['seed']
        node_dim = option['node_dim']
        try:
            node_color = option['draw_node_color']
        except KeyError:
            node_color = ['r']*len(G)

        if node_lists is None:
            node_color = ['r']*len(G)
        else:
            node_color = ['r']*len(G)
            indices = np.argsort(node_lists[0]+node_lists[1])
            for ii, node in enumerate(G.nodes_iter()):
                if ii >= len(G): 
                    print("here") 
                    continue
                if node in node_lists[1]:
                    node_color[ii] = 'b'

        try:
            scale = option['draw_scale']
        except KeyError:
            scale = 100
        #===========================================================================
        if option['model'] == 'partition': 
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
             

        elif option['model'] == 'newman':
            pos = nx.circular_layout(G, dim=2, scale=scale, center=None)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        elif option['model'] == 'binomial':
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        elif option['model'] == 'power':
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")


        elif option['model'] == 'grid':
            pos = dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
        
        elif option['model'] == 'tree':
            pos = nx.circular_layout(G, dim=2, scale=scale, center=None) #nx.shell_layout(G)#nx.spring_layout(G) #nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=False, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")
    
        elif option['model'] == 'balanced_tree':
            pos = nx.nx_pydot.graphviz_layout(G)
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")


        elif option['model'] == 'bipartite_binomial' or option['model'] == 'bipartite_uniform':
            node_sets = nx.algorithms.bipartite.sets(G)
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
            if not nx.is_connected(G): #must be connected
                raise ValueError("Not connected. Please increase the edge probability.")
            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, arrows=True, with_labels=True, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        else: 
            if pos is None:
                pos = nx.spring_layout(G, pos=pos_init, scale=scale, iterations=100) #nx.nx_pydot.graphviz_layout(G)

            fig1 = plt.figure(1)
            nx.draw(G, pos=pos, with_labels=with_labels, fontsize=fontsize, node_color=node_color, font_color=font_color, node_size=node_size)
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netTop.eps"
            if save_fig == True:
                fig1.savefig(filename, format="eps")

        return pos


    def draw_degree_rank(G, save_fig=False):
        degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
        #print "Degree sequence", degree_sequence
        dmax=max(degree_sequence)
        fig3 = plt.figure(3) 
        plt.loglog(degree_sequence,'b-',marker='o')
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")
        
        # draw graph in inset
        plt.axes([0.45,0.45,0.45,0.45])
        Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
        pos=nx.spring_layout(Gcc)
        plt.axis('off')
        nx.draw_networkx_nodes(Gcc,pos,node_size=20)
        nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
        
        filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_degree_rank_plot.eps"
        #filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_eigenvalue_adjMat.eps"
        if save_fig : fig3.savefig(filename)
        plt.show()





    def get_pos_graph(self, G, choice):
        if choice == 'newman' or choice == 'tree':
            pos=nx.circular_layout(G, dim=2, scale=1.0, center=None)
        elif choice == 'grid':
            pos=dict(zip(G.nodes(), [np.asarray(u) for u in G.nodes()]))
        elif 'bipartite' in choice:
            pos=nx.nx_pydot.graphviz_layout(G, prog='dot')
        else:
            pos=nx.nx_pydot.graphviz_layout(G, prog='neato')
        return pos



    def get_pos_coordinate(self, pos, nodeIdx=None):
        if nodeIdx is None:
            return np.array([[pos[key][0], pos[key][1]] for key in pos])
        else:
            return np.array([[pos[it['node']][0], pos[it['node']][1]] for it in nodeIdx])


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
            #print(lx)
            #print(ly)
            #print('----')
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
        