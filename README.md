## About

Simple PyTorch utilities for studying the effects of different learning rates (still very much WIP).

## Features

* Compute loss values which would result from applying various learning rates.
  * For example, suppose that we have a model with parameters $\theta$ and a loss function
    $L$. One iteration of stochastic gradient descent (SGD) with learning rate $\alpha$ updates
    the parameters according to the rule $\theta_{t+1} \leftarrow L(\theta_t - \alpha \nabla_\theta L(\theta_t))$.
    Here, $\nabla_\theta L(\theta_t)$ is the gradient of $L$ with respect to $\theta$, evaluated at $\theta_t$.
    The function `loss_by_learning_rate()` computes the right-hand side $L(\theta_t - \alpha \nabla_\theta L(\theta_t))$
    for each $\alpha$ in a user-specified set of learning rates.  The implementation 
    supports arbitrary models, loss functions, and optimizers.

## Installation

```bash
git clone https://github.com/dscamiss/learning-rate-utils
pip install learning-rate-utils
```
## Examples

For more detail, see the [examples](https://github.com/dscamiss/learning-rate-utils/blob/master/examples/) directory.

### Fully-connected neural network

We can plot the loss per learning rate for varying input/output data fed to an untrained fully-connected neural network.

```python
batch_size = 32
input_dim = 4
hidden_layer_dims = [32]
output_dim = 8

learning_rates = list(np.linspace(0.0, 1.0, 100))
losses = np.ndarray((len(learning_rates), 10))

model = FullyConnected(input_dim, hidden_layer_dims, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters())

for i in range(losses.shape[-1]):
    x = 4.0 * torch.randn(batch_size, input_dim)  # Constant factor for larger errors
    y = torch.randn(batch_size, output_dim)
    losses[:, i] = loss_per_learning_rate(model, x, y, criterion, optimizer, learning_rates)
```

Plotting `losses` gives

![Fully-connected example](https://github.com/dscamiss/learning-rate-utils/blob/main/examples/figures/fully_connected_untrained.png)

### Shallow CNN

We can plot the loss per learning rate for varying input/output data fed to an untrained shallow CNN.

```python
learning_rates = list(np.linspace(0.0, 1.0, 100))
losses = np.ndarray((len(learning_rates), 10))

model = ShallowCNN()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters())

for i in range(losses.shape[-1]):
    x, y = next(iter(train_loader))
    losses[:, i] = loss_per_learning_rate(model, x, y, criterion, optimizer, learning_rates)
```

Plotting `losses` gives

![Shallow CNN example (untrained)](https://github.com/dscamiss/learning-rate-utils/blob/main/examples/figures/shallow_cnn_untrained.png)

Repeating the experiment after loading trained weights gives

![Shallow CNN example (trained)](https://github.com/dscamiss/learning-rate-utils/blob/main/examples/figures/shallow_cnn_trained.png)

In this case, the model was trained for MNIST digit classification.
