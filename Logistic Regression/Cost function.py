#cost function for logistic regression 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Tumor sizes from 0 to 10 cm
y = (X.ravel() > 5).astype(int)  # 1 if tumor size > 5 cm, else 0

# Initialize parameters
theta0 = 0
theta1 = 1

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def cost_function(theta0, theta1, X, y):
    z = theta0 + theta1 * X
    h = sigmoid(z)
    epsilon = 1e-5  # to avoid log(0)
    cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# Plot cost function over a range of theta0 and theta1
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 3, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = cost_function(theta0, theta1, X, y)

# Contour plot of the cost function
Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)
plt.figure(figsize=(10, 6))
CS = plt.contour(Theta0, Theta1, J_vals.T, levels=np.logspace(-1, 2, 20), cmap='viridis')
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$\theta_0$ (Intercept)')
plt.ylabel(r'$\theta_1$ (Coefficient)')
plt.title('Cost Function Contours for Logistic Regression')
plt.show()

# Train logistic regression using sklearn
model = LogisticRegression()
model.fit(X, y)

# Get the learned parameters
theta1_learned = model.coef_[0][0]
theta0_learned = model.intercept_[0]
print(f"Learned parameters: theta0 = {theta0_learned:.4f}, theta1 = {theta1_learned:.4f}")

# Compute the cost with learned parameters
cost_learned = cost_function(theta0_learned, theta1_learned, X, y)
print(f"Cost with learned parameters: {cost_learned:.4f}")

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c=y, cmap='viridis', edgecolors='k', label='Data Points')
# Plot the decision boundary
x_vals = np.linspace(0, 10, 100)
z = theta0_learned + theta1_learned * x_vals
prob = sigmoid(z)
plt.plot(x_vals, prob, color='red', label='Sigmoid Curve')
# Decision boundary where probability = 0.5
decision_boundary = x_vals[np.argmin(np.abs(prob - 0.5))]
plt.axvline(x=decision_boundary, color='green', linestyle='--', label=f'Decision Boundary (x = {decision_boundary:.2f} cm)')
plt.xlabel('Tumor Size (cm)')
plt.ylabel('Probability of Malignancy')
plt.title('Logistic Regression: Tumor Size vs Malignancy')
plt.legend()
plt.grid(True)
plt.show()
