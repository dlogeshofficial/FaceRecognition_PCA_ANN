import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.train import load_dataset
from src.pca import PCAEigenfaces
from src.ann import ANNClassifier

def main():
    dataset_path = "dataset"
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 1. Generate the face database (Load images)
    # Exclude one person to act as an imposter later
    imposter_name = "Aamir" 
    print(f"Loading dataset, excluding {imposter_name} for imposter testing...")
    X, y, label_map = load_dataset(dataset_path, exclude_people=[imposter_name])
    
    # Load imposter data separately
    X_imposter, _, _ = load_dataset(dataset_path, exclude_people=[p for p in os.listdir(os.path.join(dataset_path, "faces")) if p != imposter_name])

    k_values = range(5, 51, 5)
    accuracies = []

    for k in k_values:
        print(f"\nEvaluating for k = {k}")

        # PCA Implementation (Steps 2-7)
        pca = PCAEigenfaces(k)
        pca.fit(X)
        
        # Step 8: Generate Signature of Each Face
        Omega = pca.transform(X)

        # Step 9: Apply ANN for training
        # Split 60% training and 40% test set
        X_train, X_test, y_train, y_test = train_test_split(
            Omega.T, y, test_size=0.4, random_state=42
        )

        ann = ANNClassifier()
        ann.train(X_train, y_train)

        # Evaluation
        acc = ann.accuracy(X_test, y_test)
        print(f"Classification Accuracy: {acc*100:.2f}%")
        accuracies.append(acc)

        # Imposter detection (Optional: only for the last k)
        if k == k_values[-1]:
            print("\nRunning Imposter Detection...")
            # Project imposter faces into the same eigenface space
            Omega_imposter = pca.transform(X_imposter)
            
            # Predict using ANN
            # We use predict_proba to see if the model is confident
            # If max probability is low, it might be an imposter
            probs = ann.predict_proba(Omega_imposter.T)
            max_probs = np.max(probs, axis=1)
            
            threshold = 0.5 # Confidence threshold
            recognized = max_probs > threshold
            imposter_count = np.sum(~recognized)
            
            print(f"Total imposter images tested: {len(X_imposter.T)}")
            print(f"Correctly identified as 'not enrolled person': {imposter_count}")
            print(f"Imposter detection rate: {(imposter_count/len(X_imposter.T))*100:.2f}%")

    # a) Plot a graph between accuracy and k value
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [a*100 for a in accuracies], marker='o', linestyle='-', color='b')
    plt.xlabel("k (Number of Eigenvectors)")
    plt.ylabel("Accuracy (%)")
    plt.title("Face Recognition Accuracy vs Number of Eigenvectors (k)")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "accuracy_vs_k.png"))
    print(f"\nPlot saved to {os.path.join(results_dir, 'accuracy_vs_k.png')}")

if __name__ == "__main__":
    main()
