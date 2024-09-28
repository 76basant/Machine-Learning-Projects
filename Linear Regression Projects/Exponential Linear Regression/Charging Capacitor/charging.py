import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

t = list(np.arange(0, 16, 0.5))
Vc = [0, 5.89, 9.24, 11.7, 13.5, 14.78, 15.78, 16.36, 17.2, 17.46, 17.64, 17.77, 17.86, 17.91, 17.95, 17.98, 18, 18.01, 18.02, 18.03]

t = list(np.arange(0, 10, 0.5))

print(len(t),len(Vc))
if len(t)==len(Vc):
  print("The length of data is similar")
else:
  print("The length of data is not correct")

# Plot the curve
plt.scatter(t, Vc, label='Capacitor Cahrgung Curve')
#plt.plot(t, Vc,":")

# Create a higher-resolution dataset
x_smooth = np.linspace(min(t), max(t), 300)  # Increase data points for a smoother curve
pchip = PchipInterpolator(t, Vc)  # Pchip interpolation
y_smooth = pchip(x_smooth)

# Plot with smoother curve
#plt.scatter(x_data, y_data)
plt.plot(x_smooth, y_smooth)




c=max(Vc)
print(c)
# Plot the horizontal line intersecting the y-axis at 'b'
plt.axhline(y=max(Vc), color='r', linestyle='--', label=f'Horizontal Line at $\epsilon={max(Vc)}$')

#steps to find time constant
#first step
b=max(Vc)*(1-np.exp(-1))
print(b)


# Plot the horizontal line intersecting the y-axis at 'b'
plt.axhline(y=b, color='black', linestyle='--', label=f'Horizontal Line at b={round(b,2)}')


#second step:
# Find the intersection point between the horizontal line and the curve
intersection_index = np.argmin(np.abs(np.array(Vc) - b))

print(intersection_index)

intersection_t = t[intersection_index]

intersection_point_curve = (intersection_t, b)

print(intersection_point_curve )

# Plot the vertical line at the intersection point on the curve
plt.axvline(x=intersection_t, color='green', linestyle='--', label=f'Vertical Line at τ={round(intersection_t, 2)} minute')



#steps to calcualte capacitance value 

R=10**4
c=intersection_point_curve[0]*60/R

print("Capacitance= ",c,"Farad")
print(intersection_point_curve[0])


# Print the value of c as text on the figure
plt.text(5,5, f'Capacitance = {c:.4f} Farad', fontsize=16, color='blue',fontweight='bold')



# Add labels and legend
plt.xlabel('Time(minute)')
plt.ylabel('Vc(Volt)')
plt.legend()
plt.xlim(0,max(t)+1)
plt.ylim(0,max(Vc)+1)
# Show the plot

plt.show ()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the exponential charging function using the time constant tau
def exp_model(t, V_inf, tau):
    return V_inf * (1 - np.exp(-t / tau))

# Real data of capacitor charging
t = np.arange(0, 10, 0.5)  # Time in seconds: [0.0, 0.5, 1.0, ..., 9.5]
Vc = [
    0, 5.89, 9.24, 11.7, 13.5, 14.78, 15.78, 16.36,
    17.2, 17.46, 17.64, 17.77, 17.86, 17.91, 17.95,
    17.98, 18, 18.01, 18.02, 18.03
]  # Voltage in volts

# Set V_inf as the maximum observed voltage
V_inf_initial = max(Vc)

# Split data into training and testing sets
# 80% training data and 20% testing data
t_train, t_test, Vc_train, Vc_test = train_test_split(
    t, Vc, test_size=0.2, random_state=42
)

# Display the split data
print("Training Data:")
for ti, vi in zip(t_train, Vc_train):
    print(f"t = {ti:.1f} s, Vc = {vi} V")

print("\nTesting Data:")
for ti, vi in zip(t_test, Vc_test):
    print(f"t = {ti:.1f} s, Vc = {vi} V")

# Initial guess for [V_inf, tau]
initial_guess = [V_inf_initial, 1.0]  # Guessing tau=1 second

# Fit the model to training data using curve_fit
params, covariance = curve_fit(exp_model, t_train, Vc_train, p0=initial_guess)

print(params)

# Extract the fitted parameters
V_inf_fit, tau_fit = params

# Display the fitted parameters
print(f"\nFitted Parameters:")
print(f"V_infinity (V_inf_fit) = {V_inf_fit:.4f} V")
print(f"Time Constant (tau_fit) = {tau_fit:.4f} s")

# Given resistance R in Ohms
R = 10**4  # 10,000 Ohms

# Calculate capacitance C using tau = R * C => C = tau / R
C = tau_fit / R  # Capacitance in Farads

# Optionally, convert C to microfarads for readability
C_microfarads = C * 1e6  # 1 F = 1,000,000 µF

# Display the capacitance
print(f"\nCalculated Capacitance:")
print(f"C = {C:.6f} F ({C_microfarads:.2f} µF)")

# Predict on test data
Vc_pred_test = exp_model(t_test, V_inf_fit, tau_fit)

# Calculate Mean Squared Error (MSE) and R² Score
mse = mean_squared_error(Vc_test, Vc_pred_test)
r2 = r2_score(Vc_test, Vc_pred_test)

# Display the evaluation metrics
print(f"\nModel Evaluation on Test Data:")
print(f"Mean Squared Error (MSE) = {mse:.4f} V²")
print(f"R² Score = {r2:.4f}")

# Generate a range of time values for plotting the fitted curve
t_extended = np.linspace(0, 10, 100)  # Time from 0 to 10 seconds

# Predict voltages using the fitted model for the extended time range
Vc_pred = exp_model(t_extended, V_inf_fit, tau_fit)

# Create the plot
plt.figure(figsize=(12, 8))

# Scatter plot for training data
plt.scatter(t_train, Vc_train, label='Training Data', color='blue', alpha=0.7, s=100, marker='o')

# Scatter plot for testing data
plt.scatter(t_test, Vc_test, label='Testing Data', color='green', alpha=0.7, s=100, marker='^')

# Plot the fitted exponential curve
plt.plot(t_extended, Vc_pred, label=f'Fitted Exponential Curve (tau = {tau_fit:.2f} s)', 
         color='red', linewidth=2, linestyle='--')

# Annotate the plot with tau and capacitance
annotation_text = (
    f"Fitted V_infinity: {V_inf_fit:.2f} V\n"
    f"Time Constant (tau): {tau_fit:.2f} s\n"
    f"Capacitance (C): {C_microfarads:.2f} µF"
)
plt.annotate(
    annotation_text,
    xy=(0.05, 0.95), xycoords='axes fraction',
    fontsize=12, color='black',
    bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'),
    verticalalignment='top'
)

# Labeling the axes
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=14)

# Title of the plot
plt.title('Exponential Fit of Capacitor Charging', fontsize=16)

# Display the legend
plt.legend(fontsize=12)

# Show grid for better readability
plt.grid(True)

# Display the plot
plt.show()
