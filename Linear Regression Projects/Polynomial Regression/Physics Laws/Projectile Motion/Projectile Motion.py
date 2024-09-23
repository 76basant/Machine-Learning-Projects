

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Step 1: Simulate Data for Projectile Motion
np.random.seed(42)
angles = np.linspace(0, 90, 100)  # Launch angles in degrees
g = 9.81  # Acceleration due to gravity (m/s^2)
initial_velocity = 50  # Initial velocity (m/s)

# Maximum height formula: H = (v^2 * sin^2(theta)) / (2 * g)
heights = (initial_velocity**2 * (np.sin(np.radians(angles))**2)) / (2 * g) + np.random.normal(0, 0.5, angles.shape)

# Create a DataFrame
data = pd.DataFrame({
    'Angle (degrees)': angles,
    'Max Height (m)': heights
})

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Prepare the Data for Polynomial Regression
X = data['Angle (degrees)'].values.reshape(-1, 1)  # Independent variable (Angle)
y = data['Max Height (m)'].values  # Dependent variable (Max Height)

# Step 3: Visualize the Data (Angle vs. Max Height)
plt.scatter(X, y, color='blue', alpha=0.5)
plt.xlabel('Angle (degrees)')
plt.ylabel('Max Height (m)')
plt.title('Scatter Plot of Angle vs. Max Height')
plt.grid(True)
plt.show()

# Step 4: Split the Data into Training and Testing Sets (50% for each)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Display training and testing set sizes
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Step 5: Create Polynomial Features
degree = 3  # Degree of the polynomial
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 6: Create and Train the Polynomial Regression Model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict the max heights on the test set
y_pred = model.predict(X_test_poly)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2_score = model.score(X_test_poly, y_test)  # Calculate R^2 score

# Display the evaluation metrics
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2_score:.4f}")

# Step 8: Visualize the Model's Predictions
plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(X_test, y_test, label='Testing Data', color='green', alpha=0.7)
plt.scatter(X_test, y_pred, label='Predicted Max Heights', color='red', marker='x')

# Plotting the fitted polynomial curve
x_range = np.linspace(0, 90, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)
plt.plot(x_range, y_range_pred, label='Polynomial Fit', color='orange', linewidth=2)

# Labels and grid
plt.xlabel('Angle (degrees)')
plt.ylabel('Max Height (m)')
plt.title('Angle vs. Max Height with Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()
