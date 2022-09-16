from re import L
import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, num_epochs=500, learning_rate=0.01, threshold=0.5):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
    
    def fit(self, X, y):
        X = self.prepare_features(X)
        self.parameters = np.zeros((X.shape[1], 1))
        num_of_samples = X.shape[0]
        
        for epoch in tqdm(range(self.num_epochs)):
            grad = np.zeros((X.shape[1], 1))
            for i in range(num_of_samples):
                x = X[i].reshape(-1, 1)
                grad += (self.sigmoid(self.parameters.T@x)-y[i])*x
            self.parameters -= self.learning_rate*grad

    def predict(self, X):
        X = self.prepare_features(X)
        y = np.zeros(X.shape[0])
        self._check_fitted()
        for i in range(len(X)):
            x = X[i]
            if self.sigmoid(self.parameters.T@x) >= self.threshold:
                y[i] = 1
            else:
                y[i] = 0
        return y

    def _check_fitted(self):
        if not hasattr(self, 'parameters'):
            raise AttributeError("Model must be trained before making predictions.")

    @staticmethod
    def prepare_features(x):
        return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
