import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, tol = 1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.tol = tol

    def fit(self, X, y):
        # init the model parameters
        n_samples,n_feature = X.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0

        prev_loss = 0

        # loop over the number of iterations
        for i in range(self.n_iters):

            # Linear regression equation to calculate the y predict
            y_pred = np.dot(X,self.weights) + self.bias

            # Calculate the derivative if weight and bias
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)))
            db = (1 / n_samples) * (np.sum(y_pred - y))

            # update the weights and bias by gradient descent optimization
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

            # check if the different of loss not for the current and prev not equal the threshold
            current_loss = np.mean(np.square(y_pred - y))

            if abs(current_loss - prev_loss) < self.tol:
                break

            prev_loss = current_loss




    def predict(self,X):
        return np.dot(X, self.weights) + self.bias