# -*- coding: utf-8 -*-

import scipy as sp
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import os 
import sys


def load_mat_data(filename, variable_name):
    mat_info = loadmat(filename)
    data = mat_info[variable_name]
    return (data, mat_info)

def load_mat_data_sparse(filename, variable_name):
    mat_info = loadmat(filename)
    data = csr_matrix(mat_info[variable_name])
    return (data, mat_info)

def sparse_adjmat_oneshot(adjMat_stack, time):
    adjMat = csr_matrix(adjMat_stack[:,:,time])
    return adjMat

def detect_isolates(adjMat):
    n= adjMat.size[1]
    if_exist_isolates = False
    isolates = []
    for i in range(n):
        if(all(v ==0 for v in adjMat[i,:].todense())):
            if_exist_isolates = True
            isolates.append(i)

    return (if_exist_isolates, isolates)