import numpy as np
from LogisticRegression import LogisticRegression

X = np.array([[2, 3], [1, 1], [4, 5], [3, 4], [1, 2], [2, 1]])  # Training features
y = np.array([1, 0, 1, 1, 0, 0])  # Labels

model = LogisticRegression(lr=0.01, n_iters=1000)  # Initialize model
model.fit(X, y)  # Train model

# Predict using the trained model
y_pred = model.predict(X)

# Output the results
print("Predicted values:", y_pred)
print("Actual values:", y)
print("Model Weights:", model.weights)
print("Model Bias:", model.bias)
