import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Time in seconds for discharging case
t = np.arange(0, 10, 0.5)  # Time in seconds

# Voltage during discharging (hypothetical data for a capacitor discharging)
Vc_discharging = [18, 17.03, 15.34, 13.82, 12.46, 11.27, 10.22, 9.32, 8.54, 7.88, 7.32, 
                  6.84, 6.43, 6.08, 5.77, 5.51, 5.27, 5.06, 4.87, 4.7]

# Verify the length of t and Vc_discharging
print(len(t), len(Vc_discharging))
if len(t) == len(Vc_discharging):
    print("The length of data is similar")
else:
    print("The length of data is not correct")

# Plot the curve for discharging
plt.scatter(t, Vc_discharging, label='Capacitor Discharging Curve')

# Create a higher-resolution dataset for smoother plotting
x_smooth = np.linspace(min(t), max(t), 300)  # Increase data points for a smoother curve
pchip = PchipInterpolator(t, Vc_discharging)  # Pchip interpolation
y_smooth = pchip(x_smooth)

# Plot the smoother curve
plt.plot(x_smooth, y_smooth, label='Smooth Discharge Curve')

# Plot the horizontal line intersecting at the initial voltage
plt.axhline(y=max(Vc_discharging), color='r', linestyle='--', label=f'Initial Voltage V₀={max(Vc_discharging)}V')

# Step to find time constant: First calculate the voltage at 1/e of V₀
V0 = max(Vc_discharging)
V_at_tau = V0 * (1 / np.e)
print("Voltage at time constant (1/e of V₀):", V_at_tau)

# Plot the horizontal line at V₀/e
plt.axhline(y=V_at_tau, color='black', linestyle='--', label=f'Horizontal Line at V₀/e={round(V_at_tau, 2)}V')

# Find the index where Vc is closest to V₀/e to estimate τ (time constant)
intersection_index = np.argmin(np.abs(np.array(Vc_discharging) - V_at_tau))
print("Intersection index:", intersection_index)

# Time corresponding to the intersection point
intersection_t = t[intersection_index]
print("Intersection time:", intersection_t)

# Plot the vertical line at the intersection point
plt.axvline(x=intersection_t, color='green', linestyle='--', label=f'Time constant τ ≈ {round(intersection_t, 2)} seconds')

# Calculate the capacitance from τ and given R
R = 10**4  # Resistance in Ohms
C = intersection_t / R  # Capacitance in Farads
print(f"Capacitance = {C:.6f} F")

# Display capacitance on the plot
plt.text(5, 5, f'Capacitance = {C:.6f} F', fontsize=16, color='blue', fontweight='bold')

# Add labels and legend
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.xlim(0, max(t) + 1)
plt.ylim(0, max(Vc_discharging) + 1)

# Show the plot
plt.show()

# ----------------------------------------------
# Part 2: Fitting the Exponential Discharge Model
# Define the exponential discharging function: V(t) = V0 * exp(-t / tau)
def exp_model_discharge(t, V0, tau):
    return V0 * np.exp(-t / tau)

# Split the data into training and testing sets (80% train, 20% test)
t_train, t_test, Vc_train, Vc_test = train_test_split(
    t, Vc_discharging, test_size=0.2, random_state=42
)

# Initial guess for V0 and tau
initial_guess = [V0, 1.0]  # Initial guess for tau is 1 second

# Fit the model to training data using curve_fit
params, covariance = curve_fit(exp_model_discharge, t_train, Vc_train, p0=initial_guess)

# Extract the fitted parameters
V0_fit, tau_fit = params
print(f"Fitted V0 (Initial Voltage) = {V0_fit:.4f} V")
print(f"Fitted Time Constant (τ) = {tau_fit:.4f} s")

# Calculate the capacitance using tau = R * C, hence C = tau / R
C_fit = tau_fit / R
C_microfarads_fit = C_fit * 1e6  # Convert Farads to microfarads
print(f"Fitted Capacitance = {C_fit:.6f} F ({C_microfarads_fit:.2f} µF)")

# Evaluate the model on test data
Vc_pred_test = exp_model_discharge(t_test, V0_fit, tau_fit)

# Calculate evaluation metrics
mse = mean_squared_error(Vc_test, Vc_pred_test)
r2 = r2_score(Vc_test, Vc_pred_test)
print(f"Mean Squared Error (MSE) = {mse:.4f} V²")
print(f"R² Score = {r2:.4f}")

# Generate a range of time values for plotting the fitted curve
t_extended = np.linspace(0, 10, 100)  # Time from 0 to 10 seconds

# Predict voltages using the fitted model for the extended time range
Vc_pred = exp_model_discharge(t_extended, V0_fit, tau_fit)

# Plot the fitted model against the data
plt.figure(figsize=(12, 8))

# Scatter plot for training data
plt.scatter(t_train, Vc_train, label='Training Data', color='blue', alpha=0.7, s=100, marker='o')

# Scatter plot for testing data
plt.scatter(t_test, Vc_test, label='Testing Data', color='green', alpha=0.7, s=100, marker='^')

# Plot the fitted exponential curve
plt.plot(t_extended, Vc_pred, label=f'Fitted Exponential Discharge Curve (τ = {tau_fit:.2f} s)', 
         color='red', linewidth=2, linestyle='--')

# Annotate with the fitted values
annotation_text = (
    f"Fitted V0: {V0_fit:.2f} V\n"
    f"Time Constant (τ): {tau_fit:.2f} s\n"
    f"Capacitance (C): {C_microfarads_fit:.2f} µF"
)
plt.annotate(
    annotation_text,
    xy=(0.05, 0.95), xycoords='axes fraction',
    fontsize=12, color='black',
    bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'),
    verticalalignment='top'
)

# Label the axes
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=14)

# Set plot title
plt.title('Exponential Fit of Capacitor Discharging', fontsize=16)

# Show the legend and grid
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()
