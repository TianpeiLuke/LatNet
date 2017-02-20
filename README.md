#Latent graph signal estimation from noisy network

by Tianpei (Luke) Xie. 

This package includes __python__ codes that implements the _latent graph signal analysis_ in last chapter of my thesis. It contains the following sections: 

## Probabilistic graphical model 

The scipt `PGMcommon.py` contains several classes of probabilistic grapical models, including 

  1. Sigmoid Belief Network

   A directed Baysian network on [0,1]^{d} where $d$ is the dimension of features and equals to the number of nodes in graph. A Sigmoid Belief Network defines each conditional probability as $$p(s_i = 1 | Pa(s_i)) = \sigma\left(\mathbf{w}^{T}Pa(s_i) + \beta_{i} \right)  $$




## Generate probablistic graphical model


