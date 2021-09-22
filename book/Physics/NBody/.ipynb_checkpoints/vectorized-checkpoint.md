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

# Implementation using `numpy` vectorization and broadcasting

```{code-cell} ipython3
def pairwise_interactions(X):
    # --> Number of particles.
    n = len(X)
    
    # --> Gram matrix.
    d2 = -2 * X @ X.T
    
    # --> Squared pairwise distances.
    diag = -0.5 * np.einsum('ii->i', d2)
    d2 += diag + diag[:, None]
    np.einsum('ii->i', d2)[...] = 1 # Prevent division by 0.
    
    # --> Net forces.
    F = np.nansum( (X[:, None, :] - X) * d2[..., None]**-1.5, axis=0)
    
    return F
```

```{code-cell} ipython3
X = np.random.randn(500, 3)
```

```{code-cell} ipython3
%%timeit
pairwise_interactions(X)
```
