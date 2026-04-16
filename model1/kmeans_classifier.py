"""K-means classifier for Model 1.

Main owner:
- Member 3

Purpose:
- train and use K-means on the extracted features
"""

import numpy as np


class KMeansClassifier:
    """K-means clustering used as a supervised classifier.

    Fit learns cluster centroids via Lloyd's algorithm with k-means++
    initialisation.  A majority-vote label map (cluster → class) is derived
    from the training labels so that predict() returns class labels instead
    of anonymous cluster indices.

    All computation is pure numpy — no scikit-learn or deep-learning libs.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (should match the number of emotion classes,
        default 4).
    max_iter : int
        Maximum Lloyd iterations (default 300).
    tol : float
        Convergence threshold on the maximum centroid shift (default 1e-4).
    seed : int
        Random seed for centroid initialisation (default 42).
    """

    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None           # shape (n_clusters, n_features)
        self.cluster_to_label = None    # dict: cluster index → class label

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _kmeans_plus_plus(self, X):
        """K-means++ seeding for better centroid initialisation."""
        rng = np.random.RandomState(self.seed)
        n = X.shape[0]
        first = rng.randint(0, n)
        centroids = [X[first].copy()]

        for _ in range(1, self.n_clusters):
            # Squared distance from each point to its nearest centroid
            C = np.array(centroids)                                      # (k, F)
            dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # (N, k)
            min_dists = dists.min(axis=1)                                # (N,)
            probs = min_dists / min_dists.sum()
            next_idx = rng.choice(n, p=probs)
            centroids.append(X[next_idx].copy())

        return np.array(centroids)                                       # (n_clusters, F)

    # ------------------------------------------------------------------ #
    # Assignment step                                                      #
    # ------------------------------------------------------------------ #

    def _assign(self, X):
        """Return the nearest centroid index for every row of X."""
        # (N, K) squared-Euclidean distance matrix, computed without a loop
        dists = np.sum(
            (X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2
        )
        return np.argmin(dists, axis=1)   # (N,)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """Fit K-means and build the cluster-to-label map.

        Parameters
        ----------
        X : np.ndarray  shape (N, n_features)
            Feature matrix.
        y : array-like  shape (N,)
            True class labels (int or str).

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.centroids = self._kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            assignments = self._assign(X)

            new_centroids = np.array([
                X[assignments == k].mean(axis=0)
                if np.any(assignments == k)
                else self.centroids[k]          # keep old centroid if cluster is empty
                for k in range(self.n_clusters)
            ])

            max_shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if max_shift < self.tol:
                break

        # Build majority-vote label map
        final_assignments = self._assign(X)
        self.cluster_to_label = {}
        for k in range(self.n_clusters):
            mask = final_assignments == k
            if mask.any():
                values, counts = np.unique(y[mask], return_counts=True)
                self.cluster_to_label[k] = values[np.argmax(counts)]
            else:
                self.cluster_to_label[k] = k    # fallback: use cluster index

        return self

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : np.ndarray  shape (N, n_features)

        Returns
        -------
        np.ndarray  shape (N,)
            Predicted class labels (same dtype as y passed to fit).
        """
        cluster_ids = self._assign(X)
        return np.array([self.cluster_to_label[c] for c in cluster_ids])

    def score(self, X, y):
        """Return fraction of correctly classified samples (accuracy)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
