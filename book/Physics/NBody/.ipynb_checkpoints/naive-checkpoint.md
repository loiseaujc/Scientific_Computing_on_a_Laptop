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

# A naïve implementation

Let us start with the implementation students are most likely to come up with.
As usual, this implementation often involves nested `for` loops and possibly redundant computations.
We'll assume that `numpy` has already been imported as `np`.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
```

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
