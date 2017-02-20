# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime
import scipy as sp


class Gaussian_Random_Field(object): 
    '''
         A Gaussian random field with measurement X on undirected graph G 
    '''
    def __init__ (self, X, G, n, option):
        self.X = X, self.G = G, self.n = n
        

        