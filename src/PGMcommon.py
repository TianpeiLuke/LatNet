
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import FancyArrowPatch
import os
from time import gmtime, strftime
#import pymc3 as pm3
import pymc  as pm
from pylab import Inf
from math import exp, log



#===================================================================================
#        Sigmoid Belief Network
#
#
class SigBeliefNet(pm.MCMC):
    '''
          Sigmoid Belief Net --- Directed Bayesian Network (DBN)

          For each binary v, 

          P(v|Pa(v)) = sigmoid(np.dot(w.T,Pa(v)))

    '''
    def __init__(self, G=nx.balanced_tree(2,3, create_using=nx.DiGraph())):
        self.G = G
        self.n = len(G)
        roots = [n for n,d in G.in_degree().items() if d==0]
        # root random variables
        self.roots   = [pm.Bernoulli(str(v), 0.5, value=0) for v in roots]

        self.node_top_sort  = nx.topological_sort(G)
        node_variables = dict(zip(roots, self.roots))
        self.others = []
        
        for node in self.node_top_sort:
            if node not in roots:
                Pa_v = [node_variables[key] for key in node_variables.keys() if key in G.predecessors(node)]
                weight = [G[key][node]['weight'] for key in G.predecessors(node) ] 
                bias =  G.node[node]['bias']

                # energy =  w*Pa(v) + bias
                energy = pm.Lambda('energy_%d'% (node), lambda Pa_v=Pa_v, weight=weight, bias=bias: pm.LinearCombination('weighted_sum', x=Pa_v, y=weight) + bias )
                
                # Pr(v=1) = sigmoid(energy)
                # use pymc.Lambda deterministic function 
                sigmoid_cond_prob = pm.Lambda('sigmoid_%d_cond'% (node), lambda energy=energy: 1/(1+ exp(-energy)))
                node_variables[node] = pm.Bernoulli(str(node), sigmoid_cond_prob, value=0)
                
                self.others.append(node_variables[node])

        self.node_variables = node_variables
        pm.MCMC.__init__(self, [self.roots, self.others])


    def draw_data_prepare(self, G):
        pos_coordinates = self.plot_get_pos(G)
        trace_all = []
        bin_transform = lambda x: 2*x-1
        nodeIdx = []
        for i, v in enumerate(G.nodes_iter()):
            nodeIdx.append({'node': v, 'loc': i})
            trace_all.append(np.vectorize(bin_transform)(self.trace(str(v))[:].astype(int)))
        trace_all = np.asarray(trace_all)
        edge_list = G.edges()
        return (pos_coordinates, edge_list, trace_all, nodeIdx)

    def plot_get_pos(self, G):
        pos = nx.nx_pydot.graphviz_layout(G)
        return np.array([[pos[key][0], pos[key][1]] for key in pos])

    
    def make_animate(self, G, view_angle, save_anime=False):
        pos_coordinates, edge_list, trace_all, nodeIdx = self.draw_data_prepare(G)
   
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        [elev, azim] = view_angle
        ax.view_init(elev, azim)

        x = pos_coordinates[:,0]
        y = pos_coordinates[:,1]
        z = trace_all[:,0]
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
            #line=ax.quiver(x[n0_idx], y[n0_idx], 0, (x[n1_idx]- x[n0_idx]), (y[n1_idx]- y[n0_idx]), 0 , marker='o', markevery=(0, 1), markerfacecolor='r', color='k', linewidth=0.5)
            line=art3d.Line3D(lx, ly, np.zeros((2,)), marker='o', markevery=(0, 1), markerfacecolor='r', color='k', linewidth=0.5)
            ax.add_line(line)
        # plot node and node-attributes in stem plot
        lines = [] #record line objects
        for xi, yi, zi in zip(x, y, z):        
            line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='D', markevery=(1, 1), markerfacecolor='b', color='b',alpha=1)
            ax.add_line(line)
            lines.append(line)

        ax.set_xlim3d(min(x), max(x))
        ax.set_ylim3d(min(y), max(y))
        ax.set_zlim3d(min([min(z),-1]), max([max(z), 1])) 

        def animate(i):
            z = trace_all[:,i]

            for idx, line in enumerate(lines):
                ax.lines[len(edge_list)].remove()
 
            for idx, line in enumerate(lines):
                line=art3d.Line3D((x[idx], x[idx]), (y[idx], y[idx]), (0, z[idx]), marker='D', markevery=(1, 1), markerfacecolor='b', color='b',alpha=1)
                ax.add_line(line)
            return lines 

        def init():
            for line in lines:
                line.set_data([],[])
            return lines 

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=trace_all.shape[1], interval=30, blit=True)
        if save_anime == True:
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netAnime" +".mp4"
            anim.save(filename, fps=10, extra_args=['-vcodec', 'libx264'])








#==============================================================================================================================
#
#
#
#
#
#===============================================================================================================================
 




#class IsingModel(pm.MCMC):
#    '''
#        #================================================================
#        #         A Ising Model using Gibbs sampler
#        #
#
#    '''
#    def __init__(self, G=nx.grid_2d_graph(m=20, n=20), size=[20,20], node_potential=1, edge_potential=1):
#        self.G = G
#        self.size = size
#        self.param = [node_potential, edge_potential]
#        self.X   = [pm.Bernoulli(str(v), 0.5, value=0) for v in G.nodes_iter()]
#        self.X_idx = dict([(n, i) for i, n in enumerate(G.nodes_iter())])
#        self.psi = [self.edgePotential(v, G[v]) for v in G.nodes_iter()]
#        pm.MCMC.__init__(self, [self.X, self.psi])
#
#
#    def edgePotential(self, v, N_v):
#
#        def potential_logp(v, neighbors):
#            partial_energy = 0
#            for u in neighbors:
#                s_v = v
#                partial_energy += s_v
#            return -self.param[1]*partial_energy
#        
#        return pm.Potential(logp= potential_logp, name = "N_(%d, %d)" % (v[0], v[1]), parents = {'v': self.X[self.X_idx[v]], 'neighbors': [self.X[self.X_idx[w]] for w in N_v]}, doc="edge potential"  )
#
#
#    def draw_data_prepare(self, G):
#        pos_coordinates = self.plot_get_pos(G)
#        #names = [str(v) for v in G.nodes_iter()]
#        trace_all = []
#        bin_transform = lambda x: 2*x-1
#        nodeIdx = []
#        for i, v in enumerate(G.nodes_iter()):
#            nodeIdx.append({'node': v, 'loc': i})
#            trace_all.append(np.vectorize(bin_transform)(self.trace(str(v))[:].astype(int)))
#        trace_all = np.asarray(trace_all)
#        edge_list = G.edges()
#        return (pos_coordinates, edge_list, trace_all, nodeIdx)
#
#    def plot_get_pos(self, G):
#        pos = nx.nx_pydot.graphviz_layout(G)
#        return np.array([[pos[key][0], pos[key][1]] for key in pos])
#
#
#    def animate(self, G, save_anime=False):
#        hist = self.draw_data_prepare(G)
#        fig = plt.figure(4)
#        ax = fig.add_subplot(111)
#        cax = ax.matshow(hist[0], vmin=-1, vmax=1)
#    
#        def animate(i):
#            #print(i)
#            cax.set_data(hist[i])  # update the data
#            return cax,
#    
#    
#        # Init only required for blitting to give a clean slate.
#        def init():
#            cax.set_data(hist[0]) 
#            return cax,
#        # call the animator.  blit=True means only re-draw the parts that have changed.
#        anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                   frames=len(hist), interval=10, blit=True)
#    
#        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
#        # installed.  The extra_args ensure that the x264 codec is used, so that
#        # the video can be embedded in html5.  You may need to adjust this for
#        # your system: for more information, see
#        # http://matplotlib.sourceforge.net/api/animation_api.html
#        if save_anime: 
#            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_isingState"+".mp4"
#            anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264']) 
#    
#        plt.show()


class IndepSetMC(pm.MCMC):
    '''
        #=================================================================
        #   A markov random field with independent subgraph S 
        #    Pr[X = 1_{S}] = 1/Z * exp(\beta * |S|)
        #
        #   inherited from pymc.MCMC class
    '''
    def __init__(self, G=nx.cycle_graph(9), beta=0.0):
        self.G = G, self.beta = G, beta
        self.x   = [pm.Bernoulli(str(v), 0.5, value=0) for v in G.nodes_iter()]
        self.psi = [self.IndepSetPotential(v, G[v]) for v in G.nodes_iter()]
        pm.MCMC.__init__(self, [self.x, self.psi])

    def IndepSetPotential(self, v, N_v):
        '''
            N_v is neighbors of v
            
            see descption in 
                https://pymc-devs.github.io/pymc/modelbuilding.html#the-potential-classhttps://pymc-devs.github.io/pymc/modelbuilding.html#the-potential-class
        '''

        def potential_logp(v, N_v):
            if v + max(N_v) > 1:
                return -Inf
            else:
                return self.beta*v
        return pm.Potential(logp = potential_logp, name= "N_%d" % v, parents = {'v': self.x[v], 'N_v': [self.x[w] for w in N_v]}, doc = 'vertex potential term'  )


    def animate(self, G, view_angle, save_anime=False):
        pos_coordinates, edge_list, trace_all, nodeIdx = self.draw_data_prepare(G)
   
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        [elev, azim] = view_angle
        ax.view_init(elev, azim)

        x = pos_coordinates[:,0]
        y = pos_coordinates[:,1]
        z = trace_all[:,0]
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
        lines = [] #record line objects
        for xi, yi, zi in zip(x, y, z):        
            line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='D', markevery=(1, 1), markerfacecolor='b', color='b',alpha=1)
            ax.add_line(line)
            lines.append(line)

        ax.set_xlim3d(min(x), max(x))
        ax.set_ylim3d(min(y), max(y))
        ax.set_zlim3d(min([min(z),-1]), max([max(z), 1])) 

        def animate(i):
            z = trace_all[:,i]

            for idx, line in enumerate(lines):
                ax.lines[len(edge_list)].remove()
 
            for idx, line in enumerate(lines):
                line=art3d.Line3D((x[idx], x[idx]), (y[idx], y[idx]), (0, z[idx]), marker='D', markevery=(1, 1), markerfacecolor='b', color='b',alpha=1)
                ax.add_line(line)
            return lines 

        def init():
            for line in lines:
                line.set_data([],[])
            return lines 

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=trace_all.shape[1], interval=30, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        if save_anime == True:
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netAnime" +".mp4"
            anim.save(filename, fps=10, extra_args=['-vcodec', 'libx264'])




    def draw(self, G, view_angle, t=0, save_fig=False):
        #pos_coordinates = self.plot_get_pos(G)
        #names = [str(v) for v in G.nodes_iter()]
 #        trace_all = []
 #        bin_transform = lambda x: 2*x-1
 #        nodeIdx = []
 #        for i, v in enumerate(G.nodes_iter()):
 #            nodeIdx.append({'node': v, 'loc': i})
 #            trace_all.append(np.vectorize(bin_transform)(self.trace(str(v))[:].astype(int)))
 #        trace_all = np.asarray(trace_all)
 #        edge_list = G.edges()
        pos_coordinates, edge_list, trace_all, nodeIdx = self.draw_data_prepare(G)
        val = trace_all[:,t]
        if type(t) is int:
            self.plot_node_3d(pos_coordinates, edge_list, val, view_angle, nodeIdx, 1, save_fig)


    def draw_mean(self, G, view_angle, t=0, save_fig=False):
        #pos_coordinates = self.plot_get_pos(G)
        #names = [str(v) for v in G.nodes_iter()]
 #        trace_all = []
 #        bin_transform = lambda x: 2*x-1
 #        nodeIdx = []
 #        for i, v in enumerate(G.nodes_iter()):
 #            nodeIdx.append({'node': v, 'loc': i})
 #            trace_all.append(np.vectorize(bin_transform)(self.trace(str(v))[:].astype(int)))
 #        trace_all = np.asarray(trace_all)
 #        edge_list = G.edges()
        pos_coordinates, edge_list, trace_all, nodeIdx = self.draw_data_prepare(G)
        mean_trace = np.mean(trace_all, axis=1)
        if type(t) is int:
            self.plot_node_3d(pos_coordinates, edge_list, mean_trace, view_angle, nodeIdx, 1, save_fig)

    def draw_data_prepare(self, G):
        pos_coordinates = self.plot_get_pos(G)
        #names = [str(v) for v in G.nodes_iter()]
        trace_all = []
        bin_transform = lambda x: 2*x-1
        nodeIdx = []
        for i, v in enumerate(G.nodes_iter()):
            nodeIdx.append({'node': v, 'loc': i})
            trace_all.append(np.vectorize(bin_transform)(self.trace(str(v))[:].astype(int)))
        trace_all = np.asarray(trace_all)
        edge_list = G.edges()
        return (pos_coordinates, edge_list, trace_all, nodeIdx)

    def plot_get_pos(self, G):
        pos = nx.nx_pydot.graphviz_layout(G)
        return np.array([[pos[key][0], pos[key][1]] for key in pos])

    def plot_node_3d(self, pos_coordinates, edge_list,  node_values, view_angle, nodeIdx, figIdx=0, save_fig=False):  
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        [elev, azim] = view_angle
        ax.view_init(elev, azim)

        x = pos_coordinates[:,0]
        y = pos_coordinates[:,1]
        z = node_values
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
         


        ax.set_xlim3d(min(x), max(x))
        ax.set_ylim3d(min(y), max(y))
        ax.set_zlim3d(min([min(z),-1]), max([max(z), 1])) 
        plt.show()
        #filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netNode"+ str(figIdx) +".eps"
        #fig.savefig(filename, format='eps', dpi=1000)
        if save_fig == True:
            filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_netNode"+ str(figIdx) +".png"
            fig.savefig(filename)


#=======================================================================================
#                
#               A naive direct implementation, 
#
#=======================================================================================
def ising_2dmodel(G, m, n, Tmax):
    '''
        # ===========================================================================
        #                       Ising model        
        #
        #  p(s_(0,0), s_(0,1), ... , s_(1,0), ..., s_(m,n) ) \proto exp(-E(s,w))
        #         where E = \sum_{((i,j), (u,v))} weight*s_((i,j))*s_((u,v)) + \sum_{(i,j)}s_(i,j)

        #  p(s_(i,j)|s_rest) = \frac{exp(-(attributes_(i,j)+\sum_{(u,v) in G[(i,j)]}weight*s_{(u,v)})(s_(i,j) + 1)  ) }
        #                           {1 + exp(-2(attributes_(i,j) + \sum_{(u,v) in G[(i,j)]}weight*s_{(u,v)})}
    '''
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

def ising_show_animate(hist, save_anime=False):
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    cax = ax.matshow(hist[0], vmin=-1, vmax=1)

    def animate(i):
        #print(i)
        cax.set_data(hist[i])  # update the data
        return cax,


    # Init only required for blitting to give a clean slate.
    def init():
        cax.set_data(hist[0]) 
        return cax,
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(hist), interval=10, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save_anime: 
        filename = "../figures/" +  strftime("%d%m%Y_%H%M%S", gmtime()) + "_isingState"+".mp4"
        anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264']) 

    plt.show()
