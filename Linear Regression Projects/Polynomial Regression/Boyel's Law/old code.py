import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Original values for L (in cm) and H (in cm Hg)
L_original = np.array([6, 5.7, 4.5, 4.4, 4])
H_original = np.array([10, 20, 30, 40, 50])

# Define how large the dataset should be (e.g., 100 points)
n_samples = 50

# Generate larger H values by interpolating in the range of H_original and adding some randomness
H_large = np.linspace(min(H_original), max(H_original), n_samples)  # Linearly spaced H values
H_large += np.random.normal(0, 2, size=n_samples)  # Add small random noise to H values

# Use interpolation to calculate corresponding L values for the new H_large values
L_large = np.interp(H_large, H_original, L_original)

# Optionally add small random variations to L_large
L_large += np.random.normal(0, 0.1, size=n_samples)

# Combine original and generated data
H_combined = np.concatenate((H_original, H_large))
L_combined = np.concatenate((L_original, L_large))

# Sort the combined data by H
sorted_indices = np.argsort(H_combined)
H_sorted = H_combined[sorted_indices]
L_sorted = L_combined[sorted_indices]

# Print the sorted data in two columns
print(f"{'H_sorted':<10}{'L_sorted':<10}")
for H, L in zip(H_sorted, L_sorted):
    print(f"{H:<10.2f}{L:<10.2f}")

# Plot the sorted data for visualization
plt.scatter(H_original, L_original, color='red', label='Original Data')
plt.scatter(H_sorted, L_sorted, color='blue', label='Sorted Combined Data', alpha=0.6)
plt.xlabel('H (cm Hg)')
plt.ylabel('L (cm)')
plt.legend()
plt.title('Sorted Original and Generated Data')
plt.show()

x=H_sorted
y=L_sorted
# Step 4: Split the Data into Training and Testing Sets (50% for each)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)



print(f"Training set size: {len(x_train)}, Testing set size: {len(x_test)}")


# Define constants
r = 0.5  # radius in cm
h = 10   # height in cm
pi = np.pi


# Calculate A (Area)
A = pi * r * (r + h)  # A = pi * r * (r + h)

# Calculate V and P
P_sorted = H_sorted + 76  # P = H + 76
V_sorted = L_sorted * A   # V = L * A

# Print the sorted data for P and V
print(f"{'P_sorted':<10}{'V_sorted':<10}")
for P, V in zip(P_sorted, V_sorted):
    print(f"{P:<10.2f}{V:<10.2f}")

# Plot P vs V
plt.scatter(V_sorted, P_sorted, color='green', label='P vs V')
plt.xlabel('Volume (V) [cm^3]')
plt.ylabel('Pressure (P) [cm Hg]')
plt.title('Pressure vs Volume (Boyle\'s Law)')
plt.legend()
plt.show()




x=V_sorted
y=P_sorted
split_index = int(0.5 * len(x))
V_train, V_test = x[:split_index], x[split_index:]
P_train, P_test = y[:split_index], y[split_index:]

print(f"Training set size: {len(x_train)}, Testing set size: {len(x_test)}")




# Apply exponential regression (log-transform P to linearize the model)
P_train_log = np.log(P_train)

# Train the linear regression model
model = LinearRegression()
model.fit(V_train.reshape(-1, 1), P_train_log)

# Predict log(P) for the test set and exponentiate to get the actual P values
P_test_log_pred = model.predict(V_test.reshape(-1, 1))
P_test_pred = np.exp(P_test_log_pred)  # Get actual P from predicted log(P)

# Evaluate the model: calculate R^2 score
r2 = r2_score(P_test, P_test_pred)
print(f"R-squared (R²): {r2:.4f}")

# Plot the results
plt.scatter(V_train, P_train, color='green', label='Train Data (True Values)')
plt.scatter(V_test, P_test, color='blue', label='Test Data (True Values)')
plt.scatter(V_test, P_test_pred, color='red', label='Predicted Values', alpha=0.6)
plt.xlabel('Volume (V) [cm^3]')
plt.ylabel('Pressure (P) [cm Hg]')
plt.title('Exponential Regression: Pressure vs Volume (Test Data)')
plt.legend()
plt.show()
