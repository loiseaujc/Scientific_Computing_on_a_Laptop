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

# Full code for an N-Body simulation

We now have every thing we need to implement a fairly efficient N-Body solver.
For the sake of simplicity, the equations of motions will be integrated forward in time using the `solve_ivp` function from `scipy.integrate`.
It gives access to fairly standard yet efficient time-integrators (e.g. high-order Runge-Kutta schemes).
Note that, for the problem we consider, energy-preserving schemes (e.g. symplectic time-integrators) might be better.
It however requires other packages that we won't cover herein.
If you want to know more, you can check for instance [diffeqpy](https://github.com/SciML/diffeqpy), a Python wrapper for the incredible Julia package [DifferentialEquations.jl](https://diffeq.sciml.ai/dev) providing such symplectic integrators.

**<center> N-Body simulator using only NumPy and SciPy</center>**

```{code-cell} ipython3
# --> Import needed packages/functions.
import numpy as np
from scipy.integrate import solve_ivp

def pairwise_interactions(X):
    """Computation of the net forces resulting from pairwise interactions."""
    # --> Number of particles.
    n = len(X)
    
    # --> Gram matrix.
    d2 = -2 * X @ X.T
    
    # --> Squared pairwise distances.
    diag = -0.5 * np.einsum('ii->i', d2)
    d2 += diag + diag[:, None]
    
    # --> Prevent division by zero.
    np.einsum('ii->i', d2)[...] = 1
    
    # --> Net forces.
    F = np.nansum( (X[:, None, :] - X) * d2[..., None]**-1.5, axis=0)
    
    return F

def nbody(t, u, ndim):
    """Right-hand side function for the N-Body simulation."""
    # --> Number of particles.
    n = len(u) // (2*ndim)
    
    # --> Initialize output vector.
    du = np.zeros_like(u)
    
    # --> Extract the positions and velocities.
    x, dx = u[:ndim*n], u[ndim*n:]
    
    # --> Compute the acceleration.
    ddx = pairwise_interactions(x.reshape(n, ndim))

    # --> Return the time-derivatives for the ODE solver.
    du[:ndim*n] = dx
    du[ndim*n:] = ddx.ravel()
       
    return du
    
def simulation(x, dx, t):
    
    # --> Number of dimensions/Number of particles.
    assert x.shape == dx.shape
    n, ndim = x.shape

    # --> Initial condition for the simulation.
    u = np.r_[x.flatten(), dx.flatten()]
    
    # --> Parameters for the ODE solver.
    tspan = (t.min(), t.max())
    t_eval = t
    method = "DOP853"
    atol = rtol = 1e-13
    
    # --> Run the simulation.
    output = solve_ivp(
        lambda t, u : nbody(t, u, ndim),
        tspan,
        u,
        t_eval=t_eval,
        method=method,
        atol=atol,
        rtol=rtol
    )
    
    # --> Extract the positions and velocities of the n particles.
    x = output["y"][:n*ndim]
    dx = output["y"][n*ndim:]

    return x, dx
```

```{code-cell} ipython3
n, ndim = 4, 2

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
dx = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])

t = np.linspace(0, 3.0, 1024)

x, dx = simulation(x, dx, t)
```

```{code-cell} ipython3
X = x.reshape((n, ndim, -1))
dX = dx.reshape((n, ndim, -1))
```

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

for x in X:
    ax.plot(x[0], x[1], color="lightgray")

ln, = ax.plot([], [], "o")
    
ax.axis(False);
ax.set_aspect("equal")

def update(i):
    ln.set_data(X[:, 0, i], X[:, 1, i])
    return ln

anim = FuncAnimation(fig, update, frames=len(t), interval=10);
```

```{code-cell} ipython3
---
render:
  image:
    align: center
tags: [remove-input, full-width]
---
HTML(anim.to_html5_video())
```

```{code-cell} ipython3

```
