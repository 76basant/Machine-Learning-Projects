import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulate some data
np.random.seed(42)
current = np.random.uniform(0.1, 10, 50)  # Current in Amperes (I)
true_resistance = 5.0  # True resistance in Ohms
voltage = true_resistance * current + np.random.normal(0, 0.5, current.shape)  # Voltage in Volts (V)

# Create a DataFrame
data = pd.DataFrame({
    'Current (A)': current,
    'Voltage (V)': voltage
})

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 2: Prepare the Data
x = data['Current (A)'].values.reshape(-1, 1)  # Current (A) as independent variable
y = data['Voltage (V)'].values  # Voltage (V) as dependent variable

# Step 3: Visualize the Data
plt.scatter(x, y, color='blue', alpha=0.5)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('Scatter Plot of Current vs. Voltage')
plt.grid(True)
plt.show()

# Step 4: Split the Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Training set size: {len(x_train)}, Testing set size: {len(x_test)}")

# Step 5: Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict the voltage on the test set
y_pred = model.predict(x_test)

# Step 6: Evaluation
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")

# Calculate and print the R^2 score
r2_score = model.score(x_test, y_test)
print(f"R^2 Score: {r2_score:.4f}")

# Print the model's coefficient (slope) and intercept
print(f"Model Coefficient (Estimated Resistance): {model.coef_[0]:.4f} Ohms")
print(f"Model Intercept (Voltage Offset): {model.intercept_:.4f} Volts")

# Step 7: Visualize the Results
plt.scatter(x_train, y_train, label='Train Data', color='blue')
plt.scatter(x_test, y_test, label='Test Data', color='green')
plt.plot(x_test, y_pred, label='Fitted Line', color='red', linewidth=2)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.title('Predicted vs. Actual Voltage')
plt.legend()
plt.grid(True)
plt.show()
