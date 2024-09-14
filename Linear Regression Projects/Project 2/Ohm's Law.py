import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Simulate Data for Ohm's Law (V = IR)
np.random.seed(42)  # Ensure reproducibility
current = np.linspace(0.1, 10, 100)  # Current (I) in Amperes
true_resistance = 5.0  # True resistance in Ohms
voltage = true_resistance * current + np.random.normal(0, 0.5, current.shape)  # Voltage (V) with noise

# Create a DataFrame to store the simulated data
data = pd.DataFrame({
    'Current (A)': current,
    'Voltage (V)': voltage
})

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Prepare the Data for the Linear Regression Model
x = data['Current (A)'].values.reshape(-1, 1)  # Independent variable (Current)
y = data['Voltage (V)'].values  # Dependent variable (Voltage)

# Step 3: Visualize the Data (Current vs. Voltage)
plt.scatter(x, y, color='blue', alpha=0.5)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('Scatter Plot of Current vs. Voltage')
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

# Predict the voltage values on the test set
y_pred = model.predict(x_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2_score = model.score(x_test, y_test)  # Calculate R^2 score

# Display the evaluation metrics
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2_score:.4f}")
print(f"Estimated Resistance (Slope): {model.coef_[0]:.4f} Ohms")
print(f"Voltage Offset (Intercept): {model.intercept_:.4f} Volts")

# Value of current for which we want to predict voltage (12 A)
current_value = np.array([[12]])

# Predict the voltage at 12A
predicted_voltage = model.predict(current_value)

# Display the prediction
print(f"Predicted Voltage at 12A: {predicted_voltage[0]:.4f} Volts")

# Step 7: Visualize the Model's Predictions
plt.scatter(x_train, y_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(x_test, y_test, label='Testing Data', color='green', alpha=0.7)
plt.plot(x_test, y_pred, label='Fitted Line', color='red', linewidth=2)

# Plot the predicted point (Current = 12 A, Predicted Voltage)
plt.scatter(current_value, predicted_voltage, color='purple', label=f'Prediction at 12A', s=100, zorder=5)

plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('Current vs. Voltage with Fitted Line and Predicted Value')
plt.legend()
plt.grid(True)
plt.show()

