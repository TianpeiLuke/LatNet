# Gaussian random field and latent graph signal analysis

Consider a probabilistic graph signal model, $P(\mathbf{X}|\mathcal{G})$, where $\mathcal{G}:= (\mathcal{V}, \mathcal{E})$ with vertex set $\mathcal{V}$ and $\mathcal{E}$. Each verterx $v\in \mathcal{V}$ is associated with a $p$-dimensional measurement $\mathbf{x}\_{v} \in \mathbb{R}^{p}$. Let $\mathbf{X}:= [\mathbf{x}\_{1}, \\ldots, \mathbf{x}\_{N}]^{T} \in \mathbb{R}^{N \times p}$ 

Assume that each column of $\mathbf{X}$ is independent and 
$$
   x\_{v} = f(h\_{1}, \\ldots, h\_{v}, \\ldots, h\_{N})
$$ with 
$$
    p(h\_{1}, h_2, \\ldots h_{N} ) = \mathcal{N}(\mathbf{0}, \mathbf{J}_{h,h})
$$ where $\mathbf{J}\_{h,h}$ is inverse covariance matrix of the hidden variables. 