import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([[10], [20], [30], [40], [50]])
y = np.array([5, 10, 15, 20, 25])  # Just a simple linear relationship for illustration

# ---- Min-Max Scaling ----
min_max_scaler = MinMaxScaler()
x_min_max_scaled = min_max_scaler.fit_transform(x)

# ---- Standardization (Z-score scaling) ----
standard_scaler = StandardScaler()
x_standard_scaled = standard_scaler.fit_transform(x)

# ---- Linear Regression with Min-Max Scaling ----
model_min_max = LinearRegression()
model_min_max.fit(x_min_max_scaled, y)
y_pred_min_max = model_min_max.predict(x_min_max_scaled)

# ---- Linear Regression with Standardization ----
model_standard = LinearRegression()
model_standard.fit(x_standard_scaled, y)
y_pred_standard = model_standard.predict(x_standard_scaled)

# Output the results
print("Original x values:\n", x)
print("Min-Max Scaled x values:\n", x_min_max_scaled)
print("Standardized x values:\n", x_standard_scaled)
print("Predictions with Min-Max scaling:", y_pred_min_max)
print("Predictions with Standardization:", y_pred_standard)



import numpy as np

# Original data
x = np.array([10, 20, 30, 40, 50])

# Mean normalization formula
x_mean = np.mean(x)
x_min = np.min(x)
x_max = np.max(x)

# Applying mean normalization
x_normalized = (x - x_mean) / (x_max - x_min)

# Output the result
print("Original x values:", x)
print("Mean Normalized x values:", x_normalized)
