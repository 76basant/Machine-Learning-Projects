import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
file_name = "Simple pendulum data.xlsx"
data = pd.read_excel(file_name)

# Display the data to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Step 2: Prepare the data
x = data.iloc[:, 0].values.reshape(-1, 1)  # Length (m)
y = (data.iloc[:, 1].values) ** 2  # Period (s) squared

# Calculate the correlation coefficient between Length and Period^2
correlation_coefficient = np.corrcoef(data.iloc[:, 0], y)[0, 1]
print(f"\nCorrelation Coefficient between Length and Period^2: {correlation_coefficient:.4f}")

# Step 3: Visualize the data
plt.scatter(x, y, color='blue', alpha=0.5)
plt.xlabel('Length (m)')
plt.ylabel('Period^2 (s^2)')
plt.title('Scatter Plot of Length vs. Period^2')
plt.grid(True)
plt.xlim([0, max(x)])
plt.ylim([0, max(y)])
plt.show()

# Step 4: Split the data into training and testing sets
split_index = int(0.5 * len(x))
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training set size: {len(x_train)}, Testing set size: {len(x_test)}")

# Step 5: Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict the period^2 on the test set
y_pred = model.predict(x_test)

# Step 6: Evaluation
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")

# Calculate and print the R^2 score
r2_score = model.score(x_test, y_test)
print(f"R^2 Score: {r2_score:.4f}")

# Print the model's coefficient (slope) and intercept
print(f"Model Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Model Intercept: {model.intercept_:.4f}")

# Estimate the gravitational constant g using the model's slope
g_estimated = (4 * np.pi**2) / model.coef_[0]
print(f"Estimated g: {g_estimated:.4f} m/s^2")

# Step 7: Visualize the results
plt.scatter(x_train, y_train, label='Train Data', color='blue')
plt.scatter(x_test, y_test, label='Test Data', color='green')
plt.scatter(x_test, y_pred, label='Predicted Data', color='red')
plt.xlabel('Length (m)')
plt.ylabel('Period^2 (s^2)')
plt.title('Predicted vs. Actual Period^2')
plt.legend()
plt.grid(True)
plt.show()
