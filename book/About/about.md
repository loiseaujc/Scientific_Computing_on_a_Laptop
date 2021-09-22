# What is this book?

For the better part of the last decade, I have been teaching a variety of applied mathematics courses for engineers.
In all of them, scientific computing played a major role.
In some courses (e.g. *Introduction to nonlinear dynamics*), it may be used to gain intuition about a particular phenomenon before turning to mathematics to prove (or disprove) what our intuition suggested.
In others (e.g. *Numerical methods for incompressible fluid dynamics*), scientific computing IS the course.
Yet, students most often lacked basic coding literacy despite having the mathematical skills, even at Master level.
If they worked at all, their implementations were often sloppy and terribly inefficient.
Moreover, given that their only computational resources are laptops, these inefficient codes prevent them from doing parametric studies or exploring different scenarios in a reasonnable time (say over a couple of hours).
Being unable to explore a phenomenom by yourself because of limited coding skills is, I believe, a massive blow to one's scientific education.

This book has been created with the desire to collect in one place all the bits and pieces about scientific computing I had to teach during my classes where it was merely a tool and not the subject of the class.
It is the book I wish I had written 10 years ago when I first started teaching.
Had I done so, my experience would have been more enjoyable: I would have had to spend less time on basic coding issues and focused more on what I actually wanted to teach.
More importantly, it would have been more profitable for students!
More time to focus on the physics/engineering aspects of what we were studying rather than on debugging poorly written pieces of code and stressing out because the assignment is due in a couple of days and the simulation takes ages to run!
So, what is this book and who is it for ?

This is not, per say, a text book about Scientific Computing, nor is it on Numerical Analysis.
There already are plenty of such text books out there.
In the rest, we will thus assume that the reader already has a basic education in numerical analysis (e.g. familiar with finite differences, gradient descent or numerical integration to name just a few).
We won't reinvent the wheel either (e.g. implementing from scratch a high performance SVD solver).
Instead, we will focus on *how to implement high-level methods efficiently* using existing tools such as **NumPy** and **SciPy**.
This book moreover uses a *teach-by-example* methodology.
For each example, we'll start with the naïve implementation students might come up with, explain what are the performance bottlenecks and show how it can be improved.
This will either come from coding tricks or by exploiting some mathematical structure underlying the problem at hand.
Because we want the book to be useful to as many students as possible, examples come from a large variety of scientific fields, from stochastic systems in biology to N-body simulations in astrophysics, the Ising model of phase transition in physics, all the way to partial differential equations such as the Kuramoto-Sivashinsky or the incompressible Navier-Stokes equations.
