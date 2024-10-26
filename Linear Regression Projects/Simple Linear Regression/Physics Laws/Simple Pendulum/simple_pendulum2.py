import math
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.linear_model import LinearRegression

#Linear Regression by scipy
# Define the objective function for curve fitting
def objective(x, z):
    return z * x

# Load the dataset
t = [5, 9, 12, 15, 16, 18]  # Time (in seconds)
x = [11, 21, 33, 44, 56, 66]  # Values of x
y = [(i / 10)**2 for i in t]  # y = (t/10)^2

# Curve fitting to find the optimal slope (z)
popt, _ = curve_fit(objective, x, y)
z = popt[0]

# Calculate the value of g in CGS units
scgs = (4 * math.pi**2) / z

# Output the results
print("Slope (z): ", z)
print("g in cgs: ", scgs)

# Plot the data points (x vs y)
pyplot.scatter(x, y, label="Data Points")

# Define the range for plotting the fitted line
step_x = (x[-1] - x[-2]) / 3
step_y = (y[-1] - y[-2]) / 3
x_line = arange(0, max(x) + step_x, 0.01)

# Calculate the fitted y values
y_line = objective(x_line, z)

# Add annotation with a box
text = "$g$= " + str(round(scgs, 2)) + "$cm/sec^2$"
plt.annotate(text, xy=(20, 3), fontsize=20, color='blue',
             bbox=dict(boxstyle='round,pad=0.9', edgecolor='black', facecolor='white'))

# Plot the fitted line
pyplot.plot(x_line, y_line, '--', color='red', label="Fitted Line")

# Set the axis limits
plt.xlim(0, max(x) + step_x)
plt.ylim(0, max(y) + step_y)

# Labeling the axes and title
plt.ylabel('$T^2$ (sec$^2$)')
plt.xlabel('$L (cm)$')
plt.title('Pendulum')

# Show legend and plot
plt.legend()
pyplot.show()

###########################################
#Linear Regression by definition linear function

# Create a DataFrame from x and y
data = pd.DataFrame({'x': x, 'y': y})
print("DataFrame:")
print(data)
print(data.cov())

cov=data.cov()
b = cov['x']['y'] / cov['x']['x']
print('Slope of the line =',b)

# intercept of the line a
# a = mean(y) - b * mean(x)
a = data['y'].mean()- b * data['x'].mean()
print('Intercept of the line = ',a)

def model_equation(x):
  y_hat = a + b * x
  return y_hat

y_pred =model_equation(data['x'])

data2 = pd.DataFrame({'y_true': data['y'], 'y_predicted': y_pred})
print("DataFrame:",data2)

plt.figure(figsize=(10,6))
plt.plot(data['x'],data['y'],'bo',markersize=10,markerfacecolor='w')
# linear regression line
plt.scatter(data['x'],y_pred,color='red')
plt.plot(data['x'],y_pred,'ro')
#
#plt.plot(data['x'],y_pred,'ro',markersize=15,markerfacecolor='w',alpha=0.8)
#plt.plot(data['x'],y_pred,'r')

#plt.plot(data['x'],y_pred,'ro')

plt.xlabel('independent')
plt.ylabel('dependent')
plt.grid()
plt.show()

#############################################
####Linear Regression model by  Scikitit learn 
df=data
X = df.iloc[:,:-1].values # converting to array
y = df.iloc[:,-1].values 

print(X.shape,y.shape)

model = LinearRegression()
model.fit(X,y)
print('Model trained or fitted sucessfully')

# intercept , slope
print(model.intercept_ , model.coef_ )

# prediction
y_pred_sk = model.predict(X)

plt.figure(figsize=(10,6))
plt.plot(df['x'],df['y'],'bo',markersize=10,markerfacecolor='w')
# linear regression line
plt.plot(df['x'],y_pred_sk,'go',markersize=15,markerfacecolor='w',alpha=0.8)
plt.plot(df['x'],y_pred_sk,'g')
plt.plot(df['x'],y_pred_sk,'go')


plt.xlabel('independent')
plt.ylabel('dependent')
plt.grid()
plt.show()
