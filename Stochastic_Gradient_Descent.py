import numpy as np

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize model parameters
theta = np.random.randn(2, 1)  # Random initial values for slope and intercept
learning_rate = 0.01
n_iterations = 1000
m = len(X)

# Stochastic Gradient Descent
for iteration in range(n_iterations):
    for i in range(m):
        random_index = np.random.randint(m)  # Randomly choose a data point
        xi = np.c_[1, X[random_index]]  # Add a bias term
        yi = y[random_index:random_index+1]
        gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta = theta - learning_rate * gradient

# Resulting 'theta' will be the optimized parameters for the linear regression model
print("Optimized theta (intercept and slope):", theta)
