"""Flattening logic for Model 1.

Main owner:
- Member 3

Purpose:
- convert extracted feature maps into vectors before classification
"""

import numpy as np


def flatten(x):
    """Flatten any ndarray to a 1-D vector (C-order)."""
    return x.flatten()


class Flatten:
    """Layer that flattens an ndarray of any shape to 1-D."""

    def forward(self, x):
        """Return x as a 1-D numpy array."""
        return x.flatten()
