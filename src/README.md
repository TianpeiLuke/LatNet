# Probabilistic Graphical Models and stochastic graph signal analysis

Consider a probabilistic graph signal model, $P(\mathbf{X}|\mathcal{G})$, where $\mathcal{G}:= (\mathcal{V}, \mathcal{E})$ with vertex set $\mathcal{V}$ and $\mathcal{E}$. Each verterx $v\in \mathcal{V}$ is associated with a $m$-dimensional measurement $\mathbf{x}\_{v}:= (x\_{v}^{(1)}, \\ldots, x\_{v}^{(m)}) \in \mathbb{R}^{m}$. Let $\mathbf{X}:= [\mathbf{x}\_{1}, \\ldots, \mathbf{x}\_{N}]^{T} \in \mathbb{R}^{N \times m}$ 

Assume that each column of $\mathbf{X}$ is independent so that they are i.i.d. samples of a stochastic function $x: \mathcal{V} \times \Omega \rightarrow \mathbb{R}$. 

<!--
$$
   x\_{v} = f(h\_{1}, \\ldots, h\_{v}, \\ldots, h\_{N}), \quad v \in \mathcal{V}
$$ with 
$$
    p(h\_{1}, h_2, \\ldots h_{N} ) = \mathcal{N}(\mathbf{0}, \mathbf{J}\_{h,h})
$$ where $\mathbf{J}\_{h,h}$ is inverse covariance matrix of the hidden variables. 
-->

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

   Implemented a Markov Random Field containing several independent sub-graph. Each node is a binary variable and the __edge potential__ is defined as 

   $$
      \log p(v | Pa(v)) \propto \left\\{\\begin{array}{cc} -\infty& v+ \max(Pa(v))>1 \\\\ \beta\, v &\text{o.w.}\\end{array} \right. 
   $$
  

## Gaussian Markov Random Field

Let $\mathbf{x}:= (x_1, \\ldots, x_N)$ be a random graph signal 
$$
    p(x_1, x_2, \\ldots x_N ) = \mathcal{N}(\mathbf{0}, \mathbf{J}\_{x,x})
$$ where $\mathbf{J}\_{x,x}$ is inverse covariance matrix of the observed variables. 

A graph constraint is superimposed on the inverse convariance structure so that the ___support___ of $\mathbf{J}\_{x,x}$ equals to the ___support___ of the adjacency matrix of the given graph $\mathcal{G}$. That is, the inverse covariance graph is equal to the given network. 

In other words, the stochastic graph signal $x: \mathcal{V} \rightarrow \mathbb{R}$ evaluated at two non-adjacency vertices are conditional independent given all the neigboring nodes. 

$$ x\_{v_1} \bot x\_{v_2} | \mathbf{x}\_{\mathcal{N}(v_1)}, \quad  v_2 \not\in \mathcal{N}(v_1)  $$

We can compare with the _sparse inverse covariance estimation_ using e.g. __gLasso__. The sparse inverse covariance estimation find the maximum likelihood estimator of inverse covariance estimation under sparsity constraint. That is, 
$$
   \widehat{\mathbf{J}\_{xx}} = \arg\min\_{\mathbf{J} \succeq \mathbf{0}} -\log\left( \det |\mathbf{J}| \right) + \text{tr}\left(\mathbf{S}\,\mathbf{J}\right) + \alpha\,\\|\mathbf{J} \\|\_{1} 
$$ 
where $\mathbf{S} := \mathbf{X}\mathbf{X}^{T}/m$



## Latent variable Gaussian Graphical Model

If the data contains hidden variables, we can assume that the Gaussian network on observed data is the marginal distribution of Gaussian network on (observed, latent). Assume that the observed sample covariance matrix is $\widehat{\mathbf{\Sigma}}\_{o}$. We can find its inverse covariance by solving 

$$(\mathbf{S}^{\*}, \mathbf{L}^{\*})= \arg\min\_{\mathbf{L}\;, \;\mathbf{S}} -\frac{m}{2}\log\det\left(\mathbf{S} - \mathbf{L}\right)+ \frac{m}{2}\text{tr}\left(\widehat{\mathbf{\Sigma}}\_{o}\left( \mathbf{S} - \mathbf{L}\right)\right) + \alpha\_{m} \left(\lambda \|\mathbf{S}\|\_{1}  + \|\mathbf{L}\|\_{\*} \right)$$
$$\text{s.t. }\phantom{===}\; \mathbf{S}- \mathbf{L}\succeq \mathbf{0}$$
$$\phantom{===}\; \mathbf{L}\succeq \mathbf{0}$$

The estimated precision matrix $\widehat{\mathbf{\Theta}}\_{o}= \mathbf{S} - \mathbf{L}$. Note that $\widehat{\mathbf{\Theta}}\_{o}$ is not sparse itself, since after marginalization, the conditional independence of data is lost. 

### Implementation of lasso with adaptive $\alpha$

We need to implement an adaptive version of graph Lasso
$$
   \widehat{\mathbf{J}\_{xx}} = \arg\min\_{\mathbf{J} \succeq \mathbf{0}} -\log\left( \det |\mathbf{J}| \right) + \text{tr}\left(\mathbf{S}\,\mathbf{J}\right) + \alpha\,\\|\mathbf{P}\odot \mathbf{J} \\|\_{1} 
$$ 
where $\mathbf{P}$ is a mask matrix with all zeros but a few elements and $\odot$ is the pointwise product.
  
Use the `skggm` package in the `download\_packages` directory. 

[] (I modified the `lasso_path` according to `sklearn.linear_model` packages in`adaptive_lasso`. Within `adaptive_lasso`, import a __Cython__ script named `cd_fast_adaptive.pyx`, which _Lasso_ with _coordinate descent_ algorithm in c. It allows for a different $\alpha\_i$ for each feature. See `cd_fast_adaptive.pyx`. The old implementation of _Lasso_ is wrapped in `cd_fast_fixed.pyx`.) 

[](I write a `setup.py` file using `Configuration` instance in `numpy.distutils.misc_util`. See `setup.py` and `setup.example`. The latter complie a test function in `test.pyx`, which define a 2D-convolution function.

To complie it, use the following command:

`python3 setup.py build_ext --inplace`

Then load module package into python directly. )
