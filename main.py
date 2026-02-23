import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.train import load_dataset
from src.pca import PCAEigenfaces
from src.ann import ANNClassifier

# Load dataset
dataset_path = "dataset"
X, y, label_map = load_dataset(dataset_path)

k_values = [5, 10, 15, 20, 25, 30]
accuracies = []

for k in k_values:
    print(f"Running for k = {k}")

    # PCA
    pca = PCAEigenfaces(k)
    pca.fit(X)
    Omega = pca.transform(X)

    # Split 60/40
    X_train, X_test, y_train, y_test = train_test_split(
        Omega.T, y, test_size=0.4, random_state=42
    )

    # ANN
    ann = ANNClassifier()
    ann.train(X_train, y_train)

    acc = ann.accuracy(X_test, y_test)
    print("Accuracy:", acc)

    accuracies.append(acc)

# Plot Accuracy vs k
plt.plot(k_values, accuracies)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.savefig("results/accuracy_vs_k.png")
plt.show()

print("Finished Successfully!")