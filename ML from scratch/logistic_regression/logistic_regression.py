import numpy as np

class LogisticRegression:
    def __init__(self, num_epochs=500, learning_rate=0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
