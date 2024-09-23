import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Simulate Data for Hooke's Law (F = k * x)
np.random.seed(42)  # Ensure reproducibility
displacement = np.linspace(0.1, 10, 100)  # Displacement (x) in meters
true_spring_constant = 15.0  # True spring constant (k) in N/m
force = true_spring_constant * displacement + np.random.normal(0, 5, displacement.shape)  # Force (F) with noise

# Create a DataFrame to store the simulated data
data = pd.DataFrame({
    'Displacement (m)': displacement,
    'Force (N)': force
})

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Prepare the Data for the Linear Regression Model
x = data['Displacement (m)'].values.reshape(-1, 1)  # Independent variable (Displacement)
y = data['Force (N)'].values  # Dependent variable (Force)

# Step 3: Visualize the Data (Displacement vs. Force)
plt.scatter(x, y, color='blue', alpha=0.5)
plt.xlabel('Displacement (m)')
plt.ylabel('Force (N)')
plt.title('Scatter Plot of Displacement vs. Force')
plt.grid(True)
plt.show()

# Step 4: Split the Data into Training and Testing Sets (50% for each)
split_index = int(len(x) * 0.5)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Display training and testing set sizes
print(f"Training set size: {len(x_train)}, Testing set size: {len(x_test)}")

# Step 5: Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict the force values on the test set
y_pred = model.predict(x_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2_score = model.score(x_test, y_test)  # Calculate R^2 score

# Display the evaluation metrics
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2_score:.4f}")
print(f"Estimated Spring Constant (Slope): {model.coef_[0]:.4f} N/m")
print(f"Force Offset (Intercept): {model.intercept_:.4f} N")

# Value of displacement for which we want to predict force (12 m)
displacement_value = np.array([[12]])

# Predict the force at 12 meters displacement
predicted_force = model.predict(displacement_value)

# Display the prediction
print(f"Predicted Force at 12m displacement: {predicted_force[0]:.4f} N")

# Step 7: Visualize the Model's Predictions
plt.scatter(x_train, y_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(x_test, y_test, label='Testing Data', color='green', alpha=0.7)
plt.plot(x_test, y_pred, label='Fitted Line', color='red', linewidth=2)

# Plot the predicted point (Displacement = 12 m, Predicted Force)
plt.scatter(displacement_value, predicted_force, color='purple', label=f'Prediction at 12m', s=100, zorder=5)

# Draw horizontal and vertical lines ending at the predicted point
plt.plot([displacement_value[0][0], displacement_value[0][0]], [0, predicted_force[0]], color='gray', linestyle='--')  # Vertical line
plt.plot([0, displacement_value[0][0]], [predicted_force[0], predicted_force[0]], color='gray', linestyle='--')  # Horizontal line

# Labels and grid
plt.xlabel('Displacement (m)')
plt.ylabel('Force (N)')
plt.title('Displacement vs. Force with Fitted Line and Predicted Value')
plt.legend()
plt.grid(True)
plt.show()
