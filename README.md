## generalized-newtons-method (WIP)

![Build](https://github.com/dscamiss/generalized-newtons-method/actions/workflows/python-package.yml/badge.svg)

A PyTorch implementation of the "generalized Newton's method" for learning rate selection, proposed in [1].

Theory and implementation notes can be found in [this blog post](https://dscamiss.github.io/blog/posts/generalized_newtons_method).

Currently only the "exact version" of the method is implemented. A future version will implement the "approximate 
version" of the method as well (the difference between the two versions is that the "approximate version" trades off 
accuracy for efficiency, since it does not materialize Hessian-vector products).

## Installation

```bash
git clone https://github.com/dscamiss/generalized-newtons-method
pip install generalized-newtons-method
```

## Usage

* Make necessary imports:

```python
import generalized_newtons_method as gen
```

* Call `make_gen_optimizer()` to make a wrapped version of your favorite optimizer:

```python
optimizer = gen.make_gen_optimizer(torch.optim.Adam, model.parameters(), lr=1e-5, ...)
```

* Create the learning rate scheduler:

```python
scheduler = gen.ExactGen(optimizer, -1, model, criterion, lr_min, lr_max)
```

* Run standard training loop:

```python
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    scheduler.step(x, y)  # <-- Note additional arguments
    optimizer.step()
```

## References

[1] Zi Bu and Shiyun Xu, Automatic gradient descent with generalized Newton’s method, [arXiv:2407.02772](https://arxiv.org/abs/2407.02772)
