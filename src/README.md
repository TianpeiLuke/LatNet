# Gaussian random field and stochastic graph signal analysis

Consider a probabilistic graph signal model, $P(\mathbf{X}|\mathcal{G})$, where $\mathcal{G}:= (\mathcal{V}, \mathcal{E})$ with vertex set $\mathcal{V}$ and $\mathcal{E}$. Each verterx $v\in \mathcal{V}$ is associated with a $p$-dimensional measurement $\mathbf{x}\_{v}:= (x\_{v}^{(1)}, \\ldots, x\_{v}^{(p)}) \in \mathbb{R}^{p}$. Let $\mathbf{X}:= [\mathbf{x}\_{1}, \\ldots, \mathbf{x}\_{N}]^{T} \in \mathbb{R}^{N \times p}$ 

Assume that each column of $\mathbf{X}$ is independent so that they are i.i.d. samples of a stochastic function $x: \mathcal{V} \times \Omega \rightarrow \mathbb{R}$. 

<!--
$$
   x\_{v} = f(h\_{1}, \\ldots, h\_{v}, \\ldots, h\_{N}), \quad v \in \mathcal{V}
$$ with 
$$
    p(h\_{1}, h_2, \\ldots h_{N} ) = \mathcal{N}(\mathbf{0}, \mathbf{J}\_{h,h})
$$ where $\mathbf{J}\_{h,h}$ is inverse covariance matrix of the hidden variables. 
-->

## Gaussian Markov Random Field

Let $\mathbf{x}:= (x_1, \\ldots, x_N)$ be a random graph signal 
$$
    p(x_1, x_2, \\ldots x_N ) = \mathcal{N}(\mathbf{0}, \mathbf{J}\_{x,x})
$$ where $\mathbf{J}\_{x,x}$ is inverse covariance matrix of the observed variables. 

A graph constraint is imposed so that the _support_ of $\mathbf{J}\_{x,x}$ equals to the _support_ of the adjacency matrix of the given graph $\mathcal{G}$. That is, the inverse covariance graph is equal to the given network. 

In other words, the stochastic graph signal $x: \mathcal{V} \rightarrow \mathbb{R}$ evaluated at two non-adjacency vertices are conditional independence. 


