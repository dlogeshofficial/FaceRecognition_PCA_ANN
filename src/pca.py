import numpy as np

class PCAEigenfaces:
    def __init__(self, k):
        self.k = k
        self.mean_face = None
        self.eigenfaces = None

    def fit(self, X):
        # X shape: (mn, p)
        self.mean_face = np.mean(X, axis=1, keepdims=True)
        Delta = X - self.mean_face

        # Surrogate covariance
        C = np.dot(Delta.T, Delta)

        eigenvalues, eigenvectors = np.linalg.eig(C)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        Psi = eigenvectors[:, :self.k]

        # Generate eigenfaces
        Phi = np.dot(Delta, Psi)

        # Normalize
        Phi = Phi / np.linalg.norm(Phi, axis=0)

        self.eigenfaces = Phi
        return self

    def transform(self, X):
        Delta = X - self.mean_face
        Omega = np.dot(self.eigenfaces.T, Delta)
        return Omega