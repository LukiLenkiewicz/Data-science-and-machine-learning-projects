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
    def __init__(self, num_epochs=500, learning_rate=0.01, threshold=0.5):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
    
    def fit(self, X, y):
        X = self._prepare_features(X)
        self.parameters = np.zeros((X.shape[1], 1))
        num_of_samples = X.shape[0]
        
        for epoch in tqdm(range(self.num_epochs)):
            grad = np.zeros((X.shape[1], 1))
            for i in range(num_of_samples):
                x = X[i].reshape(-1, 1)
                grad += (self.sigmoid(self.parameters.T@x)-y[i])*x
            self.parameters -= self.learning_rate*grad

    def predict(self, X):
        self._check_dimensions(X, self.parameters)
        X = self._prepare_features(X)
        y = np.zeros(X.shape[0])
        self._check_fitted()
        for i in range(len(X)):
            x = X[i]
            if self.sigmoid(self.parameters.T@x) >= self.threshold:
                y[i] = 1
            else:
                y[i] = 0
        return y


class SotfmaxRegression(BaseModel):
    def __init__(self, num_epochs=500, learning_rate=0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate


    def fit(self, X, y):
        X = self.prepare_features(X)
        self.labels = np.array(list(set(y)))
        self.parameters = np.zeros((X.shape[1], len(self.labels)))
        num_of_samples = X.shape[0]
        for epoch in tqdm(range(self.num_epochs)):
            for num_of_label, label in enumerate(self.labels):
                w = self.parameters[:, num_of_label] 
                w = w.reshape(-1, 1)

                grad = np.zeros((X.shape[1], 1))
                for i in range(num_of_samples):
                    y_k = 1 if y[i] == label else 0
                    x = X[i].reshape(-1, 1)
                    grad += (self.sigmoid(w.T@x)-y_k)*x 
                w -= self.learning_rate*grad
                self.parameters[:, num_of_label] = w.reshape(-1)

    def predict(self, X):
        self.check_dimensions(X, self.parameters)
        X = self.prepare_features(X)
        num_of_labels = len(self.labels)
        y = np.zeros(X.shape[0])

        for i in range(len(X)):
            predictions = []
            x = X[i]
            for label_params in range(num_of_labels):
                w = self.parameters[:, label_params]
                predictions.append(self.sigmoid(np.dot(w, x)))
            predictions = np.array(predictions)
            predictions = predictions/np.sum(predictions)
            y[i] = self.labels[np.argmax(predictions)]
        
        return y