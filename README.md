# `generalized-newtons-method`

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/generalized-newtons-method/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/generalized-newtons-method/graph/badge.svg?token=ZWTBITN49T)](https://codecov.io/gh/dscamiss/generalized-newtons-method)

A PyTorch implementation of the generalized Newton's method, first proposed in [1].

# Brief background

The generalized Newton's method is a learning rate scheduler that uses second-order derivative data.

As a concrete example, suppose that our objective is to minimize the loss function $L: \Theta \to \mathbf{R}$ using
vanilla SGD and a static learning rate $\alpha$.  One gradient descent iteration is $\theta_{t+1} \leftarrow \theta_t - \alpha \nabla_\theta L(\theta_t)$.
For this iteration, introduce the "loss per learning rate" function 

$$g(\alpha) = L(\theta_t - \alpha \nabla_\theta L(\theta_t)).$$  

Towards the objective of minimizing $L$, we can attempt to choose $\alpha$ such that 
$g$ is (approximately) minimized.  Provided that $g$ is well-approximated 
near the origin by its second-order Taylor polynomial, and
that this polynomial is strictly convex, the generalized Newton's method chooses

$$\alpha_t = \frac{d_\theta L(\theta_t) \cdot \nabla_\theta L(\theta_t)}{d_\theta^2 L(\theta_t) \cdot (\nabla_\theta L(\theta_t), \nabla_\theta L(\theta_t))}.$$

This choice of $\alpha_t$ minimizes the second-order Taylor polynomial, and therefore approximately minimizes $g$.

More theory and implementation notes can be found in [this blog post](https://dscamiss.github.io/blog/posts/generalized_newtons_method).

# Caveats

Currently only the "exact version" of the method is implemented. A future version will implement the "approximate 
version" of the method as well.  The difference between the two versions is that the "approximate version" trades off 
accuracy for efficiency, since it does not materialize the required Hessian-vector products.

# Installation

```bash
git clone https://github.com/dscamiss/generalized-newtons-method
pip install generalized-newtons-method
```

# Usage

## Setup

```python
import generalized_newtons_method as gen
model = MyModel()
criterion = MyLossCriterion()
```

* Call `make_gen_optimizer()` to make a wrapped version of your desired optimizer:

```python
optimizer = gen.make_gen_optimizer(torch.optim.AdamW, model.parameters())
```

* Create the learning rate scheduler:

```python
lr_min, lr_max = 0.0, 1e-3  # Clamp learning rate between `lr_min` and `lr_max`
scheduler = gen.ExactGen(optimizer, model, criterion, lr_min, lr_max)
```

## Training

* Run standard training loop:

```python
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    scheduler.step(x, y)  # <-- Note additional arguments
    optimizer.step()
```

# TODO

- [x] Add test cases to verify second-order coefficients
- [ ] Add "approximate version"
- [ ] Add shallow CNN training example

# References

[1] Zi Bu and Shiyun Xu, Automatic gradient descent with generalized Newton’s method, [arXiv:2407.02772](https://arxiv.org/abs/2407.02772)
