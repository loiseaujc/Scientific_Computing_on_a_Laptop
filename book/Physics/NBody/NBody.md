# N-Body simulations

An [**N-Body simulation**](https://en.wikipedia.org/wiki/N-body_simulation) is a simulation of a dynamical system made of a large number of particles.
It is widely used in astrophysics to predict the individual motions of a group of celestial objects interacting with each other, usually gravitationally.
It is also used in electrostatics problems where the [Coulomb potential](https://en.wikipedia.org/wiki/Electric_potential#Electric_potential_due_to_a_point_charge) has the same form as the gravitational one, with the exception that charges may be positive or negative.
To date, the largest N-body cosmological simulation ever performed is the [Millennium-XXL Simulation](http://galformod.mpa-garching.mpg.de/mxxlbrowser/) involving more than 300 billion particles!

```{figure} https://upload.wikimedia.org/wikipedia/commons/b/b0/Galaxy_cluster_sim.png
---
width: 1024px
---
N-body simulation of the cosmological formation of a cluster of galaxies in an expanding universe. From [Wikipedia](https://en.wikipedia.org/wiki/N-body_simulation#/media/File:Galaxy_cluster_sim.png).
```

In this section, we'll implement a small-scale analog of the Millenium XXL simulation, involving only up to a few dozen particles (maybe up to a few hundred if you have a good laptop).
While the two-body problem can be solved analytically, various schemes have been proposed in the literature to solve numerically the N-body one, or at least to solve it approximately but efficiently.
For the sake of simplicity, and owing to the limited number of particles we'll consider, the **particle-particle method** will be used in what follows.
In this method, the interaction between every pairs of particles will be computed.
As we'll see, coming up with an efficient implementation of the pairwise interactions can be already challenging enough.
