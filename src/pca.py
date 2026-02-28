import numpy as np

class PCAEigenfaces:
    def __init__(self, k):
        self.k = k
        self.mean_face = None
        self.eigenfaces = None

    def fit(self, X):
        # Step 2: Mean Calculation
        # X shape: (mn, p)
        self.mean_face = np.mean(X, axis=1, keepdims=True)
        
        # Step 3: Do mean Zero
        Delta = X - self.mean_face

        # Step 4: Calculate Surrogate Co-Variance
        # C (p, p) = Delta.T * Delta
        C = np.dot(Delta.T, Delta)

        # Step 5: Do eigenvalue and eigenvector decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Step 6: Find the best direction (Generation of feature vectors)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select k eigenvectors to generate Feature vector (Psi) p*k
        Psi = eigenvectors[:, :self.k]

        # Step 7: Generating Eigenfaces (Phi) k*mn
        # (Phi)k*mn = (Psi)t * (Delta)t
        # Actually (Phi)k*mn = (Delta * Psi).T
        self.eigenfaces = np.dot(Delta, Psi).T
        
        return self

    def transform(self, X):
        # Step 8: Generate Signature of Each Face (omega)
        # (omega)k*i = (Phi)k*mn * (Delta)mn*i
        Delta = X - self.mean_face
        Omega = np.dot(self.eigenfaces, Delta)
        return Omega