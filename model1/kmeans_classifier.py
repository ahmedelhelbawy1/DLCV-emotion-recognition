import numpy as np


class KMeansClassifier:
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.cluster_to_label = None

    def _kmeans_plus_plus(self, X):
        rng = np.random.RandomState(self.seed)
        n = X.shape[0]
        first = rng.randint(0, n)
        centroids = [X[first].copy()]

        for _ in range(1, self.n_clusters):
            C = np.array(centroids)
            dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
            min_dists = dists.min(axis=1)
            probs = min_dists / min_dists.sum()
            next_idx = rng.choice(n, p=probs)
            centroids.append(X[next_idx].copy())

        return np.array(centroids)

    def _assign(self, X):
        dists = np.sum(
            (X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2
        )
        return np.argmin(dists, axis=1)

    def fit(self, X, y):
        y = np.asarray(y)
        self.centroids = self._kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            assignments = self._assign(X)

            new_centroids = np.array([
                X[assignments == k].mean(axis=0)
                if np.any(assignments == k)
                else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            max_shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if max_shift < self.tol:
                break

        final_assignments = self._assign(X)
        self.cluster_to_label = {}
        for k in range(self.n_clusters):
            mask = final_assignments == k
            if mask.any():
                values, counts = np.unique(y[mask], return_counts=True)
                self.cluster_to_label[k] = values[np.argmax(counts)]
            else:
                self.cluster_to_label[k] = k

        return self

    def predict(self, X):
        cluster_ids = self._assign(X)
        return np.array([self.cluster_to_label[c] for c in cluster_ids])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
