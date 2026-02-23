from sklearn.neural_network import MLPClassifier

class ANNClassifier:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100,),
                                   max_iter=500)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def accuracy(self, X, y):
        return self.model.score(X, y)