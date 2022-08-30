import numpy as np

class LogisticRegression:
    def __init__(self, num_epochs=500, learning_rate=0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        self.w = np.zeros((X.shape[1], 1))
        num_of_samples = X.shape[0]
        
        for epoch in range(self.num_epochs):
            grad = np.zeros((X.shape[1], 1))
            for i in range(num_of_samples):
                x = X[i].reshape(-1, 1)
                grad += (self.sigmoid(self.w.T@x)-y[i])*x
            self.w -= self.learning_rate*grad

    def predict(self, X):
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
