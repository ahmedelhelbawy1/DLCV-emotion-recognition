"""Feature extraction pipeline for Model 1.

Main owner:
- Member 3

Purpose:
- connect convolution, pooling, activation, and flattening
- output features that can be passed to K-means
- downsample high-dimensional flat vector to 1x128 via random projection
"""

import numpy as np


class FeatureExtractor:
    """Projects a high-dimensional flat feature vector down to `output_dim`.

    Uses a fixed random Gaussian matrix (Johnson-Lindenstrauss style
    random projection).  The matrix is built lazily on the first call so
    the layer is fully shape-agnostic — the input dimensionality does not
    need to be known at construction time.

    The same seeded matrix is reused across all calls, so train/val/test
    features live in the same projected space.

    Parameters
    ----------
    output_dim : int
        Target feature dimensionality (default 128).
    seed : int
        Random seed for the projection matrix (default 42).
    """

    def __init__(self, output_dim=128, seed=42):
        self.output_dim = output_dim
        self.seed = seed
        self._W = None          # projection matrix, shape (input_dim, output_dim)

    def _build(self, input_dim):
        rng = np.random.RandomState(self.seed)
        # Columns are unit-variance Gaussian vectors, scaled by 1/sqrt(output_dim)
        # so that the expected squared norm is preserved.
        self._W = rng.randn(input_dim, self.output_dim) / np.sqrt(self.output_dim)

    def transform(self, x):
        """Project a single 1-D feature vector to output_dim dimensions.

        Parameters
        ----------
        x : np.ndarray
            1-D array of length input_dim.

        Returns
        -------
        np.ndarray
            1-D array of length output_dim.
        """
        if self._W is None:
            self._build(x.shape[0])
        return x @ self._W

    def transform_batch(self, X):
        """Project a batch of feature vectors.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, input_dim).

        Returns
        -------
        np.ndarray
            Shape (N, output_dim).
        """
        if self._W is None:
            self._build(X.shape[1])
        return X @ self._W
