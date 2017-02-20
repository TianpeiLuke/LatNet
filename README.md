#Latent graph signal estimation from noisy network

by Tianpei (Luke) Xie. 

This package includes __python__ codes that implements the ___latent graph signal analysis___ in last chapter of my thesis. It contains the following sections: 

## Probabilistic graphical model 

The scipt `PGMcommon.py` contains several classes of probabilistic grapical models, including 

  1. Sigmoid Belief Network (`SigBeliefNet`)

   A directed Baysian network on $[0,1]^d$ where $d$ is the dimension of features and equals to the number of nodes in graph. A Sigmoid Belief Network defines each conditional probability as 

   $$
      p(s_{1}, s_{2}, \ldots, s_{n}) = \prod_{i} p(s_{i}|Pa(s_{i}))
   $$
   $$
      p(s_{i} = 1 | Pa(s_{i})) = \sigma\left(\mathbf{w}^{T}Pa(s_{i}) + \beta_{i} \right)  
   $$
   Use __pymc__ package


  2. Independent Set Markov Chain Monte Carlo (`IndepSetMC`)

   Implemented a Markov Random Field containing several independent sub-graph. Ech node is a binary variable and the __edge potential__ is defined as 

   $$
      \log p(v | Pa(v)) \propto \left\\{\\begin{array}{cc} -\infty& v+ \max(Pa(v))>1 \\\\ \beta\, v &\text{o.w.}\\end{array} \right. 
   $$
  


## Generate probablistic graphical model


