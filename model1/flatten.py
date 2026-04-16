import numpy as np


def flatten(x):
    return x.flatten()


class Flatten:
    def forward(self, x):
        return x.flatten()
