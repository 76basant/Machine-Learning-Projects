# ----- Import Libraries -----

import math
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----- Linear Regression with Scipy (Curve Fitting) -----

# Define the objective function for curve fitting
def objective(x, z):
    return z * x

# Load the dataset
t = [5, 9, 12, 15, 16, 18]  # Time (in seconds)
x = [11, 21, 33, 44, 56, 66]  # Values of x
y = [(i / 10)**2 for i in t]  # y = (t/10)^2

# Curve fitting to find the optimal slope (z)
popt, _ = curve_fit(objective, x, y)
z = popt[0]

# Calculate the value of g in CGS units
scgs = (4 * math.pi**2) / z

# Output the results
print("Slope (z):", z)
print("g in cgs:", scgs)

# Plot the data points (x vs y) and the fitted line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Data Points")
x_line = arange(0, max(x) + (x[-1] - x[-2]) / 3, 0.01)
y_line = objective(x_line, z)
plt.plot(x_line, y_line, '--', color='red', label="Fitted Line")

# Annotation
text = "$g$= " + str(round(scgs, 2)) + "$cm/sec^2$"
plt.annotate(text, xy=(20, 3), fontsize=20, color='blue',
             bbox=dict(boxstyle='round,pad=0.9', edgecolor='black', facecolor='white'))

# Axis labels and title
plt.xlabel('$L$ (cm)')
plt.ylabel('$T^2$ (sec$^2$)')
plt.title('Linear Regression by Scipy')
plt.legend()
plt.grid()
plt.show()

# ----- Linear Regression Manually (Calculating Slope and Intercept) -----

# Create a DataFrame from x and y
data = pd.DataFrame({'x': x, 'y': y})
cov = data.cov()
b = cov['x']['y'] / cov['x']['x']  # Slope
a = data['y'].mean() - b * data['x'].mean()  # Intercept

print("DataFrame:")
print(data)
print("Slope of the line =", b)
print("Intercept of the line =", a)

# Prediction using manual linear equation
def model_equation(x):
    return a + b * x

y_pred = model_equation(data['x'])
data2 = pd.DataFrame({'y_true': data['y'], 'y_predicted': y_pred})
print("Predicted DataFrame:", data2)

# Plot data points and linear regression line
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'], 'bo', markersize=10, markerfacecolor='w', label='Data Points')
plt.plot(data['x'], y_pred, 'ro-', label='Linear Regression')

# Calculate and annotate `scgs`
scgs = ((2 * np.pi) ** 2) / b
text = "$g$= " + str(round(scgs, 2)) + "$cm/sec^2$"
plt.annotate(text, xy=(30, 3), fontsize=18, color='blue',
             bbox=dict(boxstyle='round,pad=0.9', edgecolor='black', facecolor='white'))

# Axis labels and title
plt.xlabel('$L$ (cm)')
plt.ylabel('$T^2$ (sec$^2$)')
plt.title('Linear Regression Manual')
plt.legend()
plt.grid()
plt.show()

# ----- Linear Regression with Scikit Learn -----

# Convert DataFrame to numpy arrays
X = data[['x']].values
y = data['y'].values

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)
print("Model trained successfully")

# Output intercept and coefficients
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Prediction
y_pred_sk = model.predict(X)

# Plot data points and linear regression line (Scikit Learn)
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'], 'bo', markersize=10, markerfacecolor='w', label='Data Points')
plt.plot(data['x'], y_pred_sk, 'go-', label='Linear Regression (Scikit Learn)')

# Calculate and annotate `scgs`
scgs = (2 * np.pi) ** 2 / model.coef_[0]
text = "$g$= " + str(round(scgs, 2)) + "$cm/sec^2$"
plt.annotate(text, xy=(30, 3), fontsize=18, color='blue',
             bbox=dict(boxstyle='round,pad=0.9', edgecolor='black', facecolor='white'))

# Axis labels and title
plt.xlabel('$L$ (cm)')
plt.ylabel('$T^2$ (sec$^2$)')
plt.title('Linear Regression by Scikit Learn')
plt.legend()
plt.grid()
plt.show()

# ----- Train & Test Split with Scikit Learn -----

# Convert lists to numpy arrays and reshape
x = np.array(x).reshape(-1, 1)
y = np.array(y)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Prediction on test data
Y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
z = model.coef_[0]
g = (4 * math.pi**2) / z

# Output Results
print("----- Linear Regression Results -----")
print(f"Slope (z) = {z:.4f} sec²/cm")
print(f"Calculated g = {g:.2f} cm/s²")
print(f"Mean Squared Error (MSE) on Test Data = {mse:.4f} sec²")
print(f"R² Score on Test Data = {r2:.4f}")

# Visualization of Train-Test Split
x_line = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data', s=100)
plt.scatter(X_test, Y_test, color='green', label='Testing Data', s=100)
plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')

# Annotate with calculated g
plt.annotate(f"$g$ = {g:.2f} cm/s²", xy=(0.7, 0.9), xycoords='axes fraction',
             fontsize=14, color='purple',
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Axis labels and title
plt.xlabel('$L$ (cm)')
plt.ylabel('$T^2$ (sec²)')
plt.title('Pendulum: Linear Regression with Train-Test Split')
plt.legend()
plt.grid(True)
plt.show()




