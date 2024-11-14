import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Integrator experiment data (frequency vs. amplitude)
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)  # Frequency data
y = np.array([20, 19, 18, 15, 11, 10, 8])  # Amplitude data

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Trying different polynomial degrees
degrees = [1, 2, 3, 4, 5]
train_mse = []
test_mse = []

best_degree = None
min_test_mse = float("inf")

for degree in degrees:
    # Fit a polynomial model of the current degree
    coeffs = np.polyfit(x_train.flatten(), y_train, degree)  # Flatten x_train for polyfit
    poly = np.poly1d(coeffs)
    
    # Predict y values for training and testing sets
    y_train_pred = poly(x_train)
    y_test_pred = poly(x_test)
    
    # Calculate MSE for this model
    train_mse_value = mean_squared_error(y_train, y_train_pred)
    test_mse_value = mean_squared_error(y_test, y_test_pred)
    train_mse.append(train_mse_value)
    test_mse.append(test_mse_value)
    
    # Print MSE for each degree on training and test data
    print(f"Degree {degree} - Train MSE: {train_mse_value}, Test MSE: {test_mse_value}")
    
    # Check if this is the best degree based on test MSE
    if test_mse_value < min_test_mse:
        min_test_mse = test_mse_value
        best_degree = degree

print(f"\nBest Degree: {best_degree} with Test MSE: {min_test_mse}")

# Fit the best model and plot the curve
best_coeffs = np.polyfit(x.flatten(), y, best_degree)  # Flatten x for polyfit
best_poly = np.poly1d(best_coeffs)

# Generate a smooth curve for plotting
x_curve = np.linspace(min(x), max(x), 100)
y_curve = best_poly(x_curve)

# Plotting the data points and the best polynomial fit curve
plt.scatter(x, y, color="red", label="Original Data")
plt.plot(x_curve, y_curve, color="blue", label=f"Best Fit (Degree {best_degree})")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Best Polynomial Fit for Integrator Experiment Data")
plt.legend()
plt.show()
