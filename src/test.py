import numpy as np

def detect_imposter(model, X_test, threshold=0.6):
    probabilities = model.predict_proba(X_test)
    results = []

    for prob in probabilities:
        if max(prob) < threshold:
            results.append("Imposter")
        else:
            results.append("Known")

    return results