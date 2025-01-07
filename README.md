## generalized-newtons-method

![Build](https://github.com/dscamiss/generalized-newtons-method/actions/workflows/python-package.yml/badge.svg)

A PyTorch implementation of the "generalized Newton's method" for learning rate selection, proposed in [1].

Theory and implementation notes can be found in [this blog post](https://dscamiss.github.io/blog/posts/generalized_newtons_method).

Currently only the exact version of the method, which materializes Hessian-vector products, is implemented for SGD.
A future version will implement the exact and approximate versions of the method for all standard PyTorch optimizers.

## Installation

```bash
git clone https://github.com/dscamiss/generalized-newtons-method
pip install generalized-newtons-method
```
## References

[1] Zi Bu and Shiyun Xu, Automatic gradient descent with generalized Newton’s method, [arXiv:2407.02772](https://arxiv.org/abs/2407.02772)
