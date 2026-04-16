"""Manual pooling layer for Model 1.

Main owner:
- Member 3

Purpose:
- implement pooling from scratch
- keep pooling separate from convolution for easier teamwork
"""

import numpy as np


class PoolingLayer:
    """2-D spatial pooling (MAX or AVERAGE) over (H, W, C) feature maps.

    Built from scratch with no deep-learning libraries.
    Mirrors the iterate_regions / forward pattern used in ConvLayer.

    Parameters
    ----------
    pool_size : int
        Side length of the square pooling window (default 2).
    mode : str
        'MAX' or 'AVERAGE' (case-insensitive).
    """

    def __init__(self, pool_size=2, mode='MAX'):
        self.pool_size = pool_size
        self.mode = mode.upper()
        if self.mode not in ('MAX', 'AVERAGE'):
            raise ValueError("mode must be 'MAX' or 'AVERAGE'")

    def iterate_regions(self, image):
        """Yield (region, i, j) for every non-overlapping pooling window.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, C)

        Yields
        ------
        region : np.ndarray  shape (pool_size, pool_size, C)
        i, j   : output row and column indices
        """
        h, w, _ = image.shape
        p = self.pool_size
        for i in range(h // p):
            for j in range(w // p):
                region = image[i*p:(i+1)*p, j*p:(j+1)*p, :]
                yield region, i, j

    def forward(self, x):
        """Apply pooling to x by iterating over every pooling window.

        Parameters
        ----------
        x : np.ndarray
            Shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Shape (H // pool_size, W // pool_size)  for 2-D input, or
            (H // pool_size, W // pool_size, C)      for 3-D input.
        """
        p = self.pool_size
        single_channel = x.ndim == 2
        if single_channel:
            x = x[:, :, np.newaxis]

        h, w, d = x.shape
        h_out = h // p
        w_out = w // p
        output = np.zeros((h_out, w_out, d))

        for region, i, j in self.iterate_regions(x):
            if self.mode == 'MAX':
                output[i, j] = np.max(region, axis=(0, 1))
            else:
                output[i, j] = np.mean(region, axis=(0, 1))

        return output[:, :, 0] if single_channel else output
