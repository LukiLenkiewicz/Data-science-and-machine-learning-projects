import numpy as np
from tqdm import tqdm


class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def _prepare_features(x):
        return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    def _check_fitted(self):
        if not hasattr(self, 'parameters'):
            raise AttributeError("Model must be trained before making predictions.")

    def _check_dimensions(self, X, parameters):
        if X.shape[1]+1 != parameters.shape[0]:
            raise ValueError("Expected 2D array, use np.reshape() to get correct dimensions.")
            

class LogisticRegression(BaseModel):
    def __init__(self, num_epochs=10000, learning_rate=0.01, threshold=0.5):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
    
    def fit(self, X, y):
        X = self._prepare_features(X)
        self.parameters = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        
        for epoch in tqdm(range(self.num_epochs)):
            grad = X.T@(self.sigmoid(X@self.parameters) - y)
            self.parameters -= self.learning_rate*grad

    def predict(self, X):
        self._check_dimensions(X, self.parameters)
        X = self._prepare_features(X)
        y = np.zeros(X.shape[0])
        self._check_fitted()
        y = (self.sigmoid(X@self.parameters) >= self.threshold)

        return y.reshape(-1).astype(int)


class SotfmaxRegression(BaseModel):
    def __init__(self, num_epochs=500, learning_rate=0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = self._prepare_features(X)
        y = self.one_hot(y)
        self.parameters = np.ones((X.shape[1], y.shape[1]))

        for epoch in tqdm(range(self.num_epochs)):
            grad = X.T@(self.softmax(X@self.parameters) - y)
            self.parameters -= self.learning_rate*grad

    def predict(self, X):
        self._check_dimensions(X, self.parameters)
        X = self._prepare_features(X)
        y = np.zeros(X.shape[0])

        y = self.softmax(X@self.parameters)
        return np.argmax(y, axis=1)

    @staticmethod
    def softmax(x):
        return x/np.sum(x, axis=1, keepdims=True)

    @staticmethod
    def one_hot(y):
        num_of_classes = len(np.unique(y))
        y_one_hot = np.zeros((y.shape[0], num_of_classes))
        for i in range(len(y_one_hot)):
            y_one_hot[i, y[i]] = 1

        return y_one_hot
