## generalized-newtons-method

A PyTorch implementation of the generalized Newton's method proposed in [1].

Currently only the exact version of the method, which materializes Hessian-vector products, is implemented for SGD.
A future version will implement the exact and approximate versions of the method for all standard PyTorch optimizers.

## Installation

```bash
git clone https://github.com/dscamiss/generalized-newtons-method
pip install generalized-newtons-method
```
## References

[1] Zi Bu and Shiyun Xu, Automatic gradient descent with generalized Newton’s method, [arXiv:2407.02772](https://arxiv.org/abs/2407.02772)
