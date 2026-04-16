import numpy as np


class FeatureExtractor:
    def __init__(self, output_dim=128, seed=42):
        self.output_dim = output_dim
        self.seed = seed
        self._W = None

    def _build(self, input_dim):
        rng = np.random.RandomState(self.seed)
        self._W = rng.randn(input_dim, self.output_dim) / np.sqrt(self.output_dim)

    def transform(self, x):
        if self._W is None:
            self._build(x.shape[0])
        return x @ self._W

    def transform_batch(self, X):
        if self._W is None:
            self._build(X.shape[1])
        return X @ self._W
