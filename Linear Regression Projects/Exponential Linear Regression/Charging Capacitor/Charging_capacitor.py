
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

# Real data of capacitor charging
t = np.arange(0, 10, 0.5)  # Truncated Time (in seconds) to match Vc length
Vc = [0, 5.89, 9.24, 11.7, 13.5, 14.78, 15.78, 16.36, 17.2, 17.46, 17.64, 17.77, 17.86, 17.91, 17.95, 17.98, 18, 18.01, 18.02, 18.03]  # Voltage (in volts)

# Step 1: Define the exponential charging function (without postulating k)
def exp_model(t, V_inf, k):
    return V_inf * (1 - np.exp(-k * t))

# Step 2: Set V_inf (V_infinity) as max(Vc)
V_inf = max(Vc)

# Step 3: Split data into train and test sets
t_train, t_test, Vc_train, Vc_test = train_test_split(t, Vc, test_size=0.2, random_state=42)

# Step 4: Fit the model to training data using curve fitting (with V_inf initialized as max(Vc))
initial_guess = [V_inf, 1]  # We guess a small value for k
params, covariance = curve_fit(exp_model, t_train, Vc_train, p0=initial_guess)

# Extract the fitted parameters
V_inf_fit, k_fit = params

# Step 5: Predict on test data
Vc_pred_test = exp_model(np.array(t_test), V_inf_fit, k_fit)
print(1/k_fit)
R=10**4
print((1/k_fit)*60/R)
# Step 6: Calculate the mean squared error and R^2 score
mse = mean_squared_error(Vc_test, Vc_pred_test)
r2 = r2_score(Vc_test, Vc_pred_test)

# Step 7: Display the results
print(f"Fitted V_infinity: {V_inf_fit:.4f}")
print(f"Fitted k (time constant): {k_fit:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Step 8: Visualize the data and the fit
t_extended = np.linspace(0, 10, 100)  # Extended time range for smooth plotting
Vc_pred = exp_model(t_extended, V_inf_fit, k_fit)



plt.scatter(t_train, Vc_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(t_test, Vc_test, label='Testing Data', color='green', alpha=0.7)
plt.plot(t_extended, Vc_pred, label=f'Fitted Exponential Curve (k = {k_fit:.4f})', color='red', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Exponential Fit of Capacitor Charging')
plt.legend()
plt.grid(True)
plt.show()
