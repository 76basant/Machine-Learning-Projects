import math

import numpy as np
from numpy import arange
from scipy.optimize import curve_fit

# define the true objective function
def objective(x,z):
 return (z * x) 
#Red Colour 
# load the dataset
t= [5,9,12,15,16,18] 
x = [11,21,33,44,56,66] 
y=[(i/10)**2 for i in t]


#data = dataframe.values
# choose the input and output variables
#x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
z = popt

scgs=(4*3.14**2)/z[0]
#ssi=(3.14*10)/(z*0.2*180*1000)

print("slope= ",z)
#print('h=',ssi,'si')
print('h=',scgs,'cgs')
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs

step_x = (x[-1] - x[-2]) / 3
step_y = (y[-1] - y[-2]) / 3

x_line = arange(0, max(x) + step_x, 0.01)

# Adding annotation with frame box
text = "$g$= " + str((round(scgs,2)))+ "$cm/sec^2$"
plt.annotate(text, xy=(x[1], y[1]), fontsize=20, color='blue', bbox=dict(boxstyle='round,pad=0.9', edgecolor='black',facecolor='white'))


# calculate the output for the range
y_line = objective(x_line, z)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
plt.xlim(0,max(x)+step_x)
plt.ylim(0,max(y)+step_y)

# naming the y axis 
plt.ylabel('$T^2 ( sec^2)$') 
plt.xlabel('$L ( cm)$') 
plt.title('Pemdulum') 
pyplot.show()



import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----- Data Preparation -----

# Data: Length of pendulum (L) in cm and Time period squared (T^2) in sec^2
t = [5, 9, 12, 15, 16, 18]       # Period time T in seconds
x = [11, 21, 33, 44, 56, 66]     # Length L in cm
y = [(i / 10) ** 2 for i in t]    # T^2 (sec^2)

# Convert lists to NumPy arrays
X = np.array(x).reshape(-1, 1)    # Feature matrix (Length)
Y = np.array(y)                    # Target vector (T^2)

# ----- Train-Test Split -----

# Split the data: 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ----- Model Training -----

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, Y_train)

# ----- Model Evaluation -----

# Predict on the test data
Y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)

# Calculate R² Score
r2 = r2_score(Y_test, Y_pred)

# ----- Gravitational Acceleration Calculation -----

# Extract the slope (z) from the model
z = model.coef_[0]

# Calculate g using the formula g = 4π² / z
g = (4 * math.pi**2) / z

# ----- Display Results -----

print("----- Linear Regression Results -----")
print(f"Slope (z) = {z:.4f} sec²/cm")
print(f"Calculated g = {g:.2f} cm/s²")
print(f"Mean Squared Error (MSE) on Test Data = {mse:.4f} sec²")
print(f"R² Score on Test Data = {r2:.4f}")

# ----- Visualization -----

# Generate a range of x values for plotting the regression line
x_line = np.linspace(min(X), max(X), 100).reshape(-1, 1)

# Predict y values using the trained model for the regression line
y_line = model.predict(x_line)

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot for training data
plt.scatter(X_train, Y_train, color='blue', label='Training Data', s=100)

# Scatter plot for testing data
plt.scatter(X_test, Y_test, color='green', label='Testing Data', s=100)

# Plot the regression line
plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')

# Annotate the plot with the calculated g value
plt.annotate(
    f"$g$ = {g:.2f} cm/s²",
    xy=(0.7, 0.9), xycoords='axes fraction',
    fontsize=14, color='purple',
    bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white')
)

# Labeling the axes
plt.xlabel('$L$ (cm)', fontsize=14)
plt.ylabel('$T^2$ (sec²)', fontsize=14)
plt.title('Pendulum: Linear Regression with Train-Test Split', fontsize=16)

# Display legend
plt.legend(fontsize=12)

# Show grid
plt.grid(True)

# Display the plot
plt.show()
