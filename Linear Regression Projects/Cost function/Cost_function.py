
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # assuming a linear relationship

# Define the hypothesis function
def hypothesis(theta_0, theta_1, x):
    return theta_0 + theta_1 * x

# Define the cost function
def cost_function(theta_0, theta_1, x, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum((hypothesis(theta_0, theta_1, x) - y) ** 2)

# Create meshgrid for theta_0 and theta_1 values
theta_0_vals = np.linspace(-10, 10, 100)
theta_1_vals = np.linspace(-2, 2, 100)
theta_0, theta_1 = np.meshgrid(theta_0_vals, theta_1_vals)

# Compute the cost for each combination of theta_0 and theta_1
cost_vals = np.zeros(theta_0.shape)
for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        cost_vals[j, i] = cost_function(theta_0_vals[i], theta_1_vals[j], x, y)

# Plot the cost function in 2D using contour plot
plt.figure(figsize=(8, 6))
cp = plt.contour(theta_0, theta_1, cost_vals, levels=50, cmap='jet')
plt.colorbar(cp)

plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.title('Cost Function Contour Plot')
plt.show()
