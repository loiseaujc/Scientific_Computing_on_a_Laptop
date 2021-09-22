# Computing the pairwise interactions

As shown in the previous page, the equation of motion for the i<sup>th</sup> particle reads

$$
\ddot{\mathbf{x}}_i = \sum_{\substack{j=1 \\ j \neq i}}^n \dfrac{\mathbf{x}_j - \mathbf{x}_i}{\| \mathbf{x}_j - \mathbf{x}_i \|^3}.
$$

The most intensive part of our simulation will thus be the computation of these pairwise interactions.
Depending on how this computation is implemented, simulating a moderately large number of particles can take from a few seconds to a few hours.
But before discussing the nitty-gritty implementation details, let us define exactly what our function should do, as well as its input and output arguments.

**What the pairwise interaction kernel should look like?**

The function computing the pairwise interactions should take as input the positions of the $n$ particles and return the net force each of them experiences.
We'll suppose that the positions are stored in an $n \times 3$ array of the form

$$
\mathbf{X}
= 
\begin{bmatrix}
    x_1 & y_1 & z_1 \\
    x_2 & y_2 & z_2 \\
    & \vdots & \\
    x_{n-1} & y_{n-1} & z_{n-1} \\
    x_n & y_n & z_n
\end{bmatrix}
$$

so that the i<sup>th</sup> row vector contains the position of the i<sup>th</sup> particle.
Similarly, the net forces should be returned as a $n \times 3$ array for the form

$$
\mathbf{F}
=
\begin{bmatrix}
    F_1^{(x)} & F_1^{(y)} & F_1^{(z)} \\
    F_2^{(x)} & F_2^{(y)} & F_2^{(z)} \\
    & \vdots & \\
    F_{n-1}^{(x)} & F_{n-1}^{(y)} & F_{n-1}^{(z)} \\
    F_n^{(x)} & F_n^{(y)} & F_n^{(z)} \\
\end{bmatrix}.
$$

The function should thus have the following interface

```python
def pairwise_interactions(X):
    """
    Computation of net force on each particle resulting
    from all the pairwise interactions.
    
    INPUT
    -----
    
    X : numpy array, shape (n, 3)
        Positions of the n particle in R^3.
        
    RETURN
    ------
    
    F : numpy array, shape (n, 3)
        Net force applied to each particle.
    """
    
    # DO SOME COMPUTATIONS
    
    return F
```

In what follows, we'll explore three different implementations of this function.
