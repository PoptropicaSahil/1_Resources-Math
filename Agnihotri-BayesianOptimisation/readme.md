# Notes from Apoorv Agnihotri's Blog titled 'Exploring Bayesian Optimization' available at <https://distill.pub/2020/bayesian-optimization/>

> All credits to him and Nipun Batra sir!

## Intro

Assume we want to mine gold in an unknown land. Mining is costly so we cannot mine everywhere. Suppose the gold distribution $f(x)$ looks something like the function below. It is bi-modal (i.e. two maximas/minimas), with a maximum value around $x=5$

<img src="readme-images/intro-graph.png" alt="drawing" width="350"/>

- **Problem 1: Best Estimate of Gold Distribution (Active Learning)** - Drill at locations providing high information about the gold distribution
- **Problem 2: Location of Maximum Gold (Bayesian Optimization)** - Drill at locations showing high promise about the gold content

## Active Learning

While there are various methods in active learning literature, we look at **uncertainty reduction**. This method proposes labeling (drilling) *the point whose model uncertainty is the highest*. Often, the variance acts as a measure of uncertainty.

Since we only know the true value of our function at a few points, we need a *surrogate model* for the values our function takes elsewhere. This surrogate should be flexible enough to model the true function.

> Using a **Gaussian Process (GP)** is a common choice, both because of its flexibility and its ability to give us uncertainty estimates.

Our surrogate model starts with a $\textrm{prior}$ of $f(x)$ — in the case of gold, we pick a $\textrm{prior}$ assuming that it's smoothly distributed. As we evaluate points (drilling), we get more data for our surrogate to learn from, updating it according to Bayes’ rule.

<img src="readme-images/surogate.png" alt="drawing" width="500"/>

We started with uniform uncertainty. But after our first update, the $\textrm{posterior}$ is certain near $x=0.5$ and uncertain away from it. We should choose the next query point *smartly*. Although there are many ways, we will be picking the most uncertain one.

1. Choose and add the point with the highest uncertainty to the training set (by querying/labeling that point)
1. Train on the new training set
1. Go to #1 till convergence or budget elapsed

|   |   |    |     |
|---|---|--- | --- |
| ![alt text](readme-images/posterior1.png) | ![Figure 2](readme-images/posterior2.png) | ![Figure 3](readme-images/posterior4.png) | ![Figure 4](readme-images/posterior4.png) |

> The visualization shows that one can estimate the true distribution in a few iterations. **Furthermore, the most uncertain positions are often the farthest points from the current evaluation points.** At every iteration, active learning **explores the domain** to make the estimates better.

## Bayesian Optimisation

The above excercise may seem wasteful — why should we use evaluations improving our estimates of regions where the function expects low gold content when we only care about the maximum?

> This is the core question in Bayesian Optimization: "Based on what we know so far, which point should we evaluate next?"

In active learning, *we picked the most uncertain point, exploring the function*. But in Bayesian Optimization, we need to balance

- *exploring* uncertain regions, which might unexpectedly have high gold content
- *exploitating* regions we already know have higher gold content

> **Acquisition Functions**: Heuristics for how desirable it is to evaluate a point, based on our present model. They are inexpensive to calculate. These are meant to balance exploration and exploitation.

***V.IMP!*** <br>
At every step, we determine what the best point to evaluate next is according to the acquisition function by optimizing it. We maintain a model describing our estimates and uncertainty at each point. The model is updated according to Bayes' rule. Repeat this process to determine the next point to evaluate. Our acquisition functions are based on this model.

## Formalising Bayesian Optimisation

Goal: Find location $ x \in ℝ^d$ corresponding to global max (or min) of function $f : ℝ^d ↦ ℝ $

| **General Constraint** | **Constraints in Gold Mining Example** |
|------------------------|----------------------------------------|
| $f$'s feasible set $A$ is simple, e.g., box constraints | Our domain in the gold mining problem is a single-dimensional box constraint: $0 ≤ x ≤ 6$ |
| $f$ is continuous but lacks special structure, e.g., concavity, that would make it easy to optimise | Our function is neither a convex nor a concave function, but bi-modal |
| $f$ is derivative-free, i.e., evaluations do not give gradient information. | Our evaluation (by drilling) of the amount of gold content at a location did not give us any gradient information |
| $f$ is expensive to evaluate | Drilling is costly |
| $f$ may be noisy. If noise, we assume it is independent and normally distributed, with common but unknown variance | We assume noiseless measurements in our modeling (though, it is easy to incorporate normally distributed noise for GP regression). |

Algorithm:

1. Choose a surrogate model for modeling the true function $f$ and define its **prior**.
1. Given the set of **observations** (function evaluations), use Bayes rule to obtain the **posterior**.
1. Use an acquisition function $α(x)$, which is a function of the posterior, to decide the next sample point $x_t = \textrm{argmax}_x α(x)$
1. Add newly sampled data to the set of **observations** and goto step #2 till convergence or budget elapses

## Acquisition Functions

### 1. Probability of Improvement (PI)

This chooses the next query point as the one which has the *highest probability of improvement* over the current $\textrm{max} f(x^+)$ (improvement of *atleast* $ϵ$). Selection of next point is -

$$ x_{t+1} = \textrm{argmax}(α_{PI}(x)) = \textrm{argmax}(P(f(x)≥(f(x^+)+ϵ))) $$

> <img src="readme-images/similarity-haha.png" alt="drawing" width="350"/>

Where $x^+ = \textrm{argmax}_{x_i \in x_{1:t}}f(x_i)$ where $x_i$ is the location at $i_{th}$ time step


### 2. Expected Improvement (EI)

### 3. Thompson Sampling

### 4. Random
