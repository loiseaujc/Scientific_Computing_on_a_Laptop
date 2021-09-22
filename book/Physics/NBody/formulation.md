# General formulation

Consider $n$ point particles in an inertial reference frame in three dimensional space $\mathbb{R}^3$.
The particles move under the influence of mutual gravitational attraction.
For the sake of simplicity, we'll consider that all particles have unit mass.
We'll assume furthermore that the gravitational constant $G$ is unity.
Each particle has a position vector $\mathbf{x}_i$ given by

$$
\mathbf{x}_i = \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}
$$

where $x_i$, $y_i$ and $z_i$ are the i<sup>th</sup> particle's coordinates in $\mathbb{R}^3$.
It also has a velocity vector $\mathbf{v}_i$ given by

$$
\mathbf{v}_i = \dot{\mathbf{x}}_i =  \begin{bmatrix} \dot{x}_i \\ \dot{y}_i \\ \dot{z}_i \end{bmatrix}
$$

where $\dot{}$ is Newton's notation for differentiation with respect to time.
Given we assumed unit masses, Newton's second law simply reads

$$
\ddot{\mathbf{x}}_i = \sum_{\substack{j = 1 \\ j \neq i}}^n \mathbf{F}_{ij}
$$

where $\mathbf{F}_{ij}$ is the force exerted by the j<sup>th</sup> particle onto the i<sup>th</sup> one.
Assuming it follows Newton's law of gravity, this force can be modelled as

$$
\mathbf{F}_{ij} = \dfrac{\mathbf{x}_j - \mathbf{x}_i}{\| \mathbf{x}_j - \mathbf{x}_i \|^3}
$$

where $\| \mathbf{x}_j - \mathbf{x}_i \|$ denotes the Euclidean distance between the two particles.
It should be obvious that computing the interaction between every pair of particles will be the bottleneck of our computational model.
Summing over all particles, the equations of motion for the i<sup>th</sup> particle are

$$
\ddot{\mathbf{x}}_i = \sum_{\substack{j=1 \\ j \neq i}}^n  \dfrac{\mathbf{x}_j - \mathbf{x}_i}{\| \mathbf{x}_j - \mathbf{x}_i \|^3}.
$$

This is a second-order nonlinear ordinary differential equation (ODE).
Each particle having 6 degrees of freedom (3 positions and 3 velocities), we thus have to solve numerically a system of $6n$ coupled ordinary differential equations.

## Useful physical properties

The N-Body problem has a number of symmetries which can be leveraged to validate one's numerical simulation.
These symmetries yield global [integrals of motion](https://en.wikipedia.org/wiki/Integral_of_motion) that need to be verified by our simulation.
Given our assumptions, the system exhibits three different symmetries.

**Translational symmetry** implies that the center of mass of the particles

$$
\mathbf{C} = \dfrac{\sum_{i=1}^n \mathbf{x}_i}{\sum_{i=1}^n m_i}
$$

moves with constant velocity so that $\mathbf{C} = \mathbf{L}_0 t + \mathbf{C}_0$.
Here, $\mathbf{L}_0$ is the linear velocity of the center of mass and $\mathbf{C}_0$ its initial position.
These two constants of motion thus represent six integrals of motion that need to be satisfied by our simulation.

**Rotational symmetry** results in the total angular momemtum

$$
\mathbf{A} = \sum_{i=1}^n \mathbf{x}_i \times \mathbf{p}_i
$$

being constant.
Here, $\mathbf{p}_i = m_i \mathbf{v}_i$ is the momentum vector of the i<sup>th</sup> particle, and $\times$ denotes the cross product.
Conservation of the total angular momentum thus provides us with three additional constants of motion.
Finally, the last constant of motion results from the **conservation of the total energy**

$$
E = T + U.
$$

In the expression above, $T$ is the kinetic energy given by

$$
T = \sum_{i=1}^n \dfrac{\| \mathbf{v}_i \|^2}{2}
$$

while

$$
U = - \sum_{1 \leq i < j \leq n} \dfrac{1}{\| \mathbf{x}_j - \mathbf{x}_i \|}
$$

is the *self-potential energy* of the system.
Some numerical methods explicitely take into account these constants of motion to ensure they are satisfied to machine precision.
This will not be considered in this example.
Hereafter, we'll simply simulate the system and verify *a posteriori* whether these theoretical constants of motion are indeed constant in our computational model.
