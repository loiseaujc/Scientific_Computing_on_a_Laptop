---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
```

# Vectorized implementation

In the previous section, we've illustrated how one could go from a simple implementation taking close to 2 seconds to compute the acceleration of 500 particles down to less than 100 milliseconds using **Numba** and its *just-in-time* compiling capabilities.
Doing so did not require any modification of the original code except for the decorator `@numba.jit()`.
Even though this 20x-25x speed-up might be sufficient, we can do even better.
We'll however need to take a step back from the code and reframe the problem slightly differently.

## Computing the pairwise distance of $n$ points in $\mathbb{R}^m$

The most intensive part of our simulation is the computation of the pairwise distance (in the $\ell_2$ norm) between the i<sup>th</sup> and j<sup>th</sup> particles.
Given the positions vectors $\mathbf{x}_i$ and $\mathbf{x}_j$, this (squared) distance can be expressed as

$$
\| \mathbf{x}_i - \mathbf{x}_j \|^2 = \left( \mathbf{x}_i - \mathbf{x}_j \right)^T \left( \mathbf{x}_i - \mathbf{x}_j \right).
$$

The set of all pairwise distances forms the (squared) distance matrix $\mathbf{D}$.
Its $ij$ entry is given by

$$
D_{ij} = \| \mathbf{x}_i - \mathbf{x}_j \|^2.
$$

The name of the game is thus: **How to compute this distance matrix efficiently?**

The answer to this question requires only a tiny bit of linear algebra.
Developping the expression for $D_{ij}$ yields

$$
\begin{aligned}
    D_{ij} & = \| \mathbf{x}_i - \mathbf{x}_j \|^2 \\
    & = \left( \mathbf{x}_i - \mathbf{x}_j \right)^T \left( \mathbf{x}_i - \mathbf{x}_j \right) \\
    & = \mathbf{x}_i^T \mathbf{x}_i - 2 \mathbf{x}_i^T \mathbf{x}_j + \mathbf{x}_j^T \mathbf{x}_j.
\end{aligned}
$$

The last expression can be rewritten as

$$
D_{ij} = \| \mathbf{x}_i \|^2 - 2 \mathbf{x}_i^T \mathbf{x}_j + \| \mathbf{x}_j \|^2.
$$

In matrix notation, $\mathbf{D}$ can thus be expressed as

$$
\mathbf{D} = \textrm{diag}(\mathbf{G})^T \mathbf{1}_n - 2 \mathbf{G} + \mathbf{1}_n^T \textrm{diag}(\mathbf{G})
$$

where $\mathbf{G}$ is the $n \times n$ Gram matrix given by $\mathbf{G} = \mathbf{XX}^T$ and $\mathbf{1}_n$ is an $n \times 1$ column vector of ones.
Computing the distance matrix like this does not require `for` loops and involves only matrix-vector or matrix-matrix products.
Under the hood, NumPy relies on [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) to perform such products.
This will be incredibly faster than using `for` loops in Python.

The last bit we need to compute is the distance vector $\Delta \mathbf{x} = \mathbf{x}_j - \mathbf{x}_i$.
To avoid using slow `for` loops, we can rely on some NumPy mechanics, namely **broadcasting**, to compute all the pairwise distance vectors in one go.
For more details about broadcasting, please refer to the dedicated [NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).

## Back to the code

We now have everything we need to write an incredibly efficient `pairwise_interactions` function.
The corresponding code is shown below.

**<center>Algorithm 2: Vectorized + broadcasting implementation</center>**

```{code-cell} ipython3
def pairwise_interactions(X):
    # --> Number of particles.
    n = len(X)
    
    # --> Gram matrix.
    d2 = -2 * X @ X.T
    
    # --> Squared pairwise distances.
    diag = -0.5 * np.einsum('ii->i', d2)
    d2 += diag + diag[:, None]
    
    # --> Prevent division by 0.
    np.einsum('ii->i', d2)[...] = 1
    
    # --> Net forces.
    F = np.nansum( (X[:, None, :] - X) * d2[..., None]**-1.5, axis=0)
    
    return F
```

The calls to `np.einseum('ii->i', d2)` creates a **view** of the diagonal of the matrix `d2`.
This is more efficient memory-wise than actually extracting the diagonal with `np.diagonal`.
Let's now benchmark this implementation using again 500 particles randomly located in $\mathbb{R}^3$.

```{code-cell} ipython3
:tags: [hide-input]
X = np.random.randn(500, 3)
```

```{code-cell} ipython3
%%timeit
pairwise_interactions(X)
```
It now takes only approximately 15 to 20 milliseconds!
This is another 3x or 4x speed-up compared to the `numba` implementation and a 80x to 100x speed-up compared to our original implementation.
This piece of code can moreover compute the pairwise distance in an arbitrary number of dimensions, as long as you use the Eucliden $\ell_2$ norm to measure distances.
These increased performances however come with a piece of code which might be harder to decipher, particularly if you don't known the linear algebra trick or what broadcasting is.
This will often be the case.
Improving the performances of a piece of code most often relies on mathematical tricks or programming mechanics which might obscure what the code is doing.
This is one of the main reasons why you need to document your code, particularly if you intend to share it with others!
