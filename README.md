# Gradient Checking 📊

Welcome to the Gradient Checking repository! This project demonstrates the concept of gradient checking in both 1-dimensional and N-dimensional contexts. Gradient checking is essential for verifying the correctness of the gradients computed by backpropagation algorithms in neural networks.

## Overview 🧠

Gradient checking ensures that the gradients calculated by your backpropagation implementation are correct by comparing them with numerical approximations. This technique helps identify errors in the gradient computations and is a crucial step in training deep learning models.

## 1-Dimensional Gradient Checking 🔢

This section includes functions for performing gradient checking in a simple 1-dimensional case.

### `forward_propagation`

Implements linear forward propagation to compute the cost function

$$ J(\theta) = \theta \cdot x $$

```python
import numpy as np

def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J
```

### `backward_propagation`

Computes the gradient of the cost function with respect to the parameter \( \theta \).

```python
def backward_propagation(x, theta):
    dtheta = x
    return dtheta
```

### `gradient_check`

Performs gradient checking by comparing the analytical gradient with the numerical gradient.

```python
import numpy as np

def gradient_check(x, theta, epsilon=1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = np.dot(thetaplus, x)
    J_minus = np.dot(thetaminus, x)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    grad = x
    numerator = np.linalg.norm(gradapprox - grad)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator / denominator
    
    if difference < 1e-7:
        print("The gradient is correct! ✅")
    else:
        print("The gradient is wrong! ❌")
    
    return difference
```

## N-Dimensional Gradient Checking 📐

This section extends gradient checking to more complex, multi-dimensional scenarios involving logistic regression.

### `forward_propagation_n`

Computes the logistic cost function for given inputs and parameters.

```python
import numpy as np

def forward_propagation_n(X, Y, parameters):
    # Forward propagation logic
    # ...
    return cost, cache
```

### `backward_propagation_n`

Calculates gradients of the cost function with respect to model parameters.

```python
def backward_propagation_n(X, Y, cache):
    # Backward propagation logic
    # ...
    return gradients
```

### `gradient_check_n`

Verifies the correctness of the gradients from `backward_propagation_n` by comparing them with numerically approximated gradients.

```python
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    # Gradient checking logic
    # ...
    return difference
```

## Results 🏆

- **1-Dimensional Gradient Checking**: The gradient is correct! ✅
- **N-Dimensional Gradient Checking**: Your backward propagation works perfectly fine! ✅

## Credits 🙏

This repository is inspired by the Deep Learning Specialization by [DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/). Special thanks to the course for providing the foundational concepts and techniques used in this project.
