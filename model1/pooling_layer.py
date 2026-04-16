import numpy as np


class PoolingLayer:
    def __init__(self, pool_size=2, mode='MAX'):
        self.pool_size = pool_size
        self.mode = mode.upper()
        if self.mode not in ('MAX', 'AVERAGE'):
            raise ValueError("mode must be 'MAX' or 'AVERAGE'")

    def iterate_regions(self, image):
        h, w, _ = image.shape
        p = self.pool_size
        for i in range(h // p):
            for j in range(w // p):
                region = image[i*p:(i+1)*p, j*p:(j+1)*p, :]
                yield region, i, j

    def forward(self, x):
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
