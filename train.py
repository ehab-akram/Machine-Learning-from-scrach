import numpy as np
from Linear_Regression import LinearRegression  # Make sure to import your class

# Sample dataset (you can replace this with real data)
# Let's use a simple linear relationship: y = 2 * X1 + 3 * X2 + 4
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([11, 15, 19, 23, 27])

# Initialize and train the Linear Regression model
model = LinearRegression(lr=0.01, n_iters=1000, tol=1e-5)
model.fit(X, y)

# Predict using the trained model
y_pred = model.predict(X)

# Output the results
print("Predicted values:", y_pred)
print("Actual values:", y)
print("Model Weights:", model.weights)
print("Model Bias:", model.bias)
