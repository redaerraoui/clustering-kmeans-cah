"""K-means implémenté à la main (initialisation K-means++).

- distance euclidienne : mise à jour des centres par la moyenne (K-means classique) ;
- distance de Manhattan (« city_block ») : mise à jour par la médiane (K-médianes), qui est
  le centre minimisant la somme des distances L1.
"""
import numpy as np


class KMeans:
    def __init__(self, k=3, distance_metric="euclidean", random_state=None, tol=1e-6):
        if distance_metric not in ("euclidean", "city_block"):
            raise ValueError("Distance non prise en charge : utilisez 'euclidean' ou 'city_block'.")
        self.k = k
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.inertia_history = []
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    def fit(self, X, max_iterations=200):
        X = np.asarray(X, dtype=float)
        if X.shape[0] < self.k:
            raise ValueError("Le nombre d'observations doit être au moins égal à k.")
        rng = np.random.default_rng(self.random_state)
        self.centroids = self.initialize_centroids(X, rng)
        self.inertia_history = []
        labels = None

        for it in range(max_iterations):
            distances = self.calculate_distances(X)              # forme (k, n)
            new_labels = np.argmin(distances, axis=0)
            # Critère minimisé : somme des distances au carré (euclidienne) ou somme des distances L1
            d_min = distances[new_labels, np.arange(X.shape[0])]
            inertia = np.sum(d_min ** 2) if self.distance_metric == "euclidean" else np.sum(d_min)
            self.inertia_history.append(float(inertia))

            converged = labels is not None and np.array_equal(labels, new_labels)
            if len(self.inertia_history) > 1:
                prev = self.inertia_history[-2]
                converged = converged or abs(prev - inertia) <= self.tol * max(prev, 1e-12)
            labels = new_labels
            self.n_iter_ = it + 1
            if converged:
                break
            self.centroids = self.update_centroids(X, labels, d_min)

        return labels, self.centroids, self.inertia_history

    # ------------------------------------------------------------------
    def calculate_distances(self, X):
        if self.distance_metric == "euclidean":
            return np.vstack([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.vstack([np.sum(np.abs(X - c), axis=1) for c in self.centroids])

    def update_centroids(self, X, labels, d_min):
        new = []
        for i in range(self.k):
            members = X[labels == i]
            if len(members) == 0:
                # Cluster vide : on le replace sur le point le plus mal représenté
                new.append(X[np.argmax(d_min)])
            elif self.distance_metric == "euclidean":
                new.append(members.mean(axis=0))
            else:
                new.append(np.median(members, axis=0))
        return np.array(new)

    def initialize_centroids(self, X, rng):
        """K-means++ : chaque nouveau centre est tiré avec une probabilité proportionnelle
        au carré de la distance au centre le plus proche déjà choisi."""
        centroids = [X[rng.integers(X.shape[0])]]
        d2 = np.sum((X - centroids[0]) ** 2, axis=1)
        for _ in range(1, self.k):
            total = d2.sum()
            idx = rng.integers(X.shape[0]) if total == 0 else rng.choice(X.shape[0], p=d2 / total)
            centroids.append(X[idx])
            d2 = np.minimum(d2, np.sum((X - X[idx]) ** 2, axis=1))
        return np.array(centroids, dtype=float)
