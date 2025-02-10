## generalized-newtons-method

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/generalized-newtons-method/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/generalized-newtons-method/graph/badge.svg?token=ZWTBITN49T)](https://codecov.io/gh/dscamiss/generalized-newtons-method)

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

* Make your model and loss criterion:

```
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

* Run standard training loop:

```python
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    scheduler.step(x, y)  # <-- Note additional arguments
    optimizer.step()
```

## TODO

- [x] Add test cases to verify second-order coefficients
- [ ] Add "approximate version"
- [ ] Add shallow CNN training example

## References

[1] Zi Bu and Shiyun Xu, Automatic gradient descent with generalized Newton’s method, [arXiv:2407.02772](https://arxiv.org/abs/2407.02772)
