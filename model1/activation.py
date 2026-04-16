"""Activation functions for Model 1.

Main owner:
- Member 3

Purpose:
- define the activation logic used after convolution/pooling stages
"""

import numpy as np


def relu(x):
    """Element-wise ReLU: max(0, x). Shape-agnostic."""
    return np.maximum(0, x)


class ReLU:
    """ReLU activation layer with a forward() interface."""

    def forward(self, x):
        """Apply ReLU element-wise to x (any shape)."""
        return np.maximum(0, x)
