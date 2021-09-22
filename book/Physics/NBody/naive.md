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

# Implementation based on nested `for` loops

Let us start with the implementation students are most likely to come up with.
As usual, this implementation often involves nested `for` loops and possibly redundant computations.
We'll assume that `numpy` has already been imported as `np`.
Such a naïve implementation is shown below.

**<center>Algorithm 1a: Nested `for` loops</center>**

```{code-cell} ipython3
def pairwise_interactions(X):

    # --> Number of particles.
    n = len(X)
    
    # --> Initialize the force array.
    F = np.zeros_like(X)
    
    # --> Loop through all the particles.
    for i in range(n):
        for j in range(n):
            
            if i == j:
                continue
                
            # --> Compute the difference vector.
            Δx = X[j] - X[i]
            
            # --> Add the ij contribution to the net force.
            F[i] += Δx / np.linalg.norm(Δx)**3
            
    return F
```

The code is self-explanatory.
It loops through each particle.
For each particle, it then compute the signed distance vector with respect to all the other particles and add the resulting gravitational attraction to the net force experienced by the i<sup>th</sup> particle.
Pretty simple.
Let us now benchmark this implementation.
For that purpose, we'll consider 500 particles with random initial positions and velocities in $\mathbb{R}^3$.
The `%%timeit` magic is a useful command to know for that.

```{code-cell} ipython3
:tags: [hide-input]
# --> Generate particles with random initial positions.
n = 500
X = np.random.randn(n, 3)
```

```{code-cell} ipython3
%%timeit
pairwise_interactions(X)
```
On my laptop, this computation takes a bit more than 2 seconds.
An actual simulation may requires thousands if not millions of time steps, with each time step necessitating the evaluation of these pairwise interactions.
Despite its simplicity, there is no way than one could simulate even a moderately large number of particles in a reasonnable time using this code snippet.

Even though the code is slow, it can run twice as fast by leveraging existing symmetries in the equations.
You can easily convince yourself that

$$
\mathbf{F}_{ij} = -\mathbf{F}_{ji}.
$$

Hence, the inner loop in **Algorithm 1a** does not need to run from `j=0` to `i=n-1` but only from `j=i+1` to `j=n-1`.
As soon as the contribution $\mathbf{F}_{ij}$ has been computed, $\mathbf{F}_{ji}$ comes for free.
The corresponding code is shown below.


**<center>Algorithm 1b: Revisiting nested `for` loops</center>**

```{code-cell} ipython3
def pairwise_interactions(X):

    # --> Number of particles.
    n = len(X)
    
    # --> Initialize the force array.
    F = np.zeros_like(X)
    
    # --> Loop through all the particles.
    for i in range(n):
        for j in range(i+1, n):
                
            # --> Compute the difference vector.
            Δx = X[j] - X[i]
            
            # --> Add the ij contribution to the net force.
            F[i] += Δx / np.linalg.norm(Δx)**3
            
            # --> Add the ji contribution to the net force.
            F[j] -= F[i]
            
    return F
```

Let us once again benchmark this implementation using the same 500 particles.

```{code-cell} ipython3
%%timeit
pairwise_interactions(X)
```

As expected, the computational time is roughly divided by 2.
Yet, the code is still extremely slow and cannot be realistically used in a production run.
The primary reason for this slowness is the use of `for` loops.
Such loops are incredibly slow in Python compared to other compiled languages as Fortran or C.
There are technical reasons for why `for` loops are slow in Python but discussing these would bring us down the rabbit hole.
For now, you only need to remember that, when its comes to scientific computing, `for` loops in Python need to be avoided like the plague.

In many cases, `for` loops can be avoided by taking a step back from code and reframing the problem slightly differently.
This will be illustrated in the next section where we'll use some neat `numpy` tricks such as **vectorization** and **broadcasting**.
It may happen however that getting rid of `for` loops is not possible.
It could also severely reduce the code readability or simply not be obvious at all.


<img src="https://numba.pydata.org/_static/numba-blue-horizontal-rgb.svg" width="192px" align="left" margin="16px" />

In these cases, Python's scientific computing ecosystem has a few extremely useful packages.
One of them is [**Numba**](https://numba.pydata.org/), an open source *just-in-time* compiler that can translate a subset of Python or NumPy code into fast machine code.
From Numba's webiste

> Numba translates Python functions to optimized machine code at runtime using the industry-standard [LLVM](https://llvm.org/) compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or Fortran.

What Numba does basically is to take your naïve Python implementation and generates on-the-fly specialized machine code for different array data types and layouts to optimize performance.

```{admonition} Be careful though...
:class: danger
At first, Numba may seem kind of magic and you may be tempted to use it on all your different pieces of code.
It may not however bring the speed-up you expected it would!
For more details about when to use or not Numba JIT capabilities, please read [Numba's documentation](https://numba.readthedocs.io/en/stable/user/5minguide.html).
```

Despite what we've just said, our function `pairwise_interactions` is luckily an excellent candidate to illustrate the massive speed-up Numba can bring.
After having imported `numba`, using its JIT capabilities is as simple as adding the decorator `@numba.jit()` to your function.
This is illustrated below.

**<center>Algorithm 1c: Numba + nested `for` loops</center>**

```{code-cell} ipython3
import numba

@numba.jit(nopython=True)
def pairwise_interactions(X):

    # --> Number of particles.
    n = len(X)
    
    # --> Initialize the force array.
    F = np.zeros_like(X)
    
    # --> Loop through all the particles.
    for i in range(n):
        for j in range(i+1, n):
                
            # --> Compute the difference vector.
            Δx = X[j] - X[i]
            
            # --> Add the ij contribution to the net force.
            F[i] += Δx / np.linalg.norm(Δx)**3
            
            # --> Add the ji contribution to the net force.
            F[j] -= F[i]
            
    return F
```

As you can see, nothing has changed in the implementation of `pairwise_interactions`.
We simply added a decorator to it.
Let's now see how it behaves using the same set of 500 particles.

```{code-cell} ipython3
%%timeit
pairwise_interactions(X)
```

Suddenly, the execution time went from close to 2 seconds down to less than 100 ms!
This is a 20x to 25x speed-up and it only required adding a single line of code.
As we'll see, we can do even better.
Yet, this massive performance boost can sometime be sufficient to use this code in a production run.
It certainly is the case for most demonstration codes intended for students.
