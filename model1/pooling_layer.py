import numpy as np


class PoolingLayer:
    def __init__(self, size=2, mode="max"):
        self.size = size
        self.mode = mode

    def iterate_regions(self, image):
        h, w, d = image.shape

        new_h = h // self.size
        new_w = w // self.size

        for i in range(new_h):
            for j in range(new_w):
                region = image[
                    i*self.size:(i+1)*self.size,
                    j*self.size:(j+1)*self.size,
                    :
                ]
                yield region, i, j

    def forward(self, image):
        h, w, d = image.shape

        new_h = h // self.size
        new_w = w // self.size

        output = np.zeros((new_h, new_w, d))

        for region, i, j in self.iterate_regions(image):
            if self.mode == "max":
                output[i, j] = np.max(region, axis=(0, 1))
            else:
                output[i, j] = np.mean(region, axis=(0, 1))

        return output
