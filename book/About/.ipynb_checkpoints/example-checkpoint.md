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
:tags: [remove-input]

import numpy as np
```

# An Example : Conway's Game of Life

Here is an exemple of what this book aims to achieve.
Consider [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), a [cellular automaton](https://en.wikipedia.org/wiki/Cellular_automaton) devised by the British mathematician [John Horton Conway](https://en.wikipedia.org/wiki/John_Horton_Conway) in 1970.
Although the mathematical rules are pretty simple, the corresponding computational model can be implemented in several different ways, from a naïve yet inefficient implementation based on nested `for` loops to the highly efficient yet more abstract [HashLife](https://en.wikipedia.org/wiki/Hashlife) algorithm relying on [quadtrees](https://en.wikipedia.org/wiki/Quadtree) and [hash tables](https://en.wikipedia.org/wiki/Hash_table).

Using only standard functions from [**NumPy**](https://numpy.org/) and [**SciPy**](https://www.scipy.org/), as well as a few mathematical tricks, our objective is to enable beginners to move away from the naïve and inefficient implementation based on nested `for` loops and to come up with a more efficient implementation.

```{important}
Experienced users might argue that even more efficient implementations exist.
This is certainly the case.
Our goal is however not to make students experts in a particular problem and associated implementation.
Rather, we aim at increasing their overall coding skills and literacy.
```

Before discussing the implementation details of Game of Life, let us first quickly explain **what is Game of Life**!

## Rules

...


## Implementing Game of Life

...

```{code-cell} ipython3
def initial_condition(nrows, ncols):
    return np.random.randint(0, 2, (nrows, ncols))
```

### A naïve implementation

...

```{code-cell} ipython3
def naive_update(state):
    # --> Size of the lattice.
    m, n = state.shape

    # --> New state.
    new_state = state.copy()

    # --> Loop through all the grid points.
    for i in range(m):
        for j in range(n):
            # --> Compute the number of neighbours.
            neighbours = (state[(i+1)%m, (j-1)%n] + state[(i+1)%m, j] + state[(i+1)%m, (j+1)%n]
                        + state[i, (j-1)%n]                           + state[i, (j+1)%n]
                        + state[(i-1)%m, (j-1)%n] + state[(i-1)%m, j] + state[(i-1)%m, (j+1)%n])

            # --> Apply rules of Game of Life.
            if state[i, j] == 1 and (neighbours < 2 or neighbours > 3):
                new_state[i, j] = 0
            elif state[i, j] == 0 and neighbours == 3:
                new_state[i, j] = 1

    return new_state
```

```{code-cell} ipython3
# -->
nrows, ncols = 256, 256
state = initial_condition(nrows, ncols)
```

```{code-cell} ipython3
%%timeit
naive_update(state)
```

### Using convolution and NumPy's tricks

...


### Comparing the performances

...


---


## To go further

...
