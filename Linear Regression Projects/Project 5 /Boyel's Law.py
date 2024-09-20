import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Example data for Boyle's Law (P = c/V)
V_original = np.array([6, 5.7, 4.5, 4.4, 4])  # Volumes in cm^3
P_original = np.array([10, 20, 30, 40, 50])   # Pressures in cm Hg

# Logarithmic Transformation: Apply log to both P and V
log_V = np.log(V_original)
log_P = np.log(P_original)

# Apply linear regression to the log-transformed data
model_log = LinearRegression()
model_log.fit(log_V.reshape(-1, 1), log_P)

# Predictions for log-transformed data
log_P_pred = model_log.predict(log_V.reshape(-1, 1))

# Convert predictions back to original scale
P_pred_log = np.exp(log_P_pred)

# Evaluate the model with R-squared
r2_log = r2_score(P_original, P_pred_log)
print(f"R-squared for logarithmic transformation: {r2_log:.4f}")

# Plotting Logarithmic Transformation Results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(V_original, P_original, color='blue', label='Original Data')
plt.plot(V_original, P_pred_log, color='red', label='Log Transformation Fit')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('Logarithmic Transformation Fit')
plt.legend()

# Reciprocal Transformation: Use 1/V as a feature for linear regression
reciprocal_V = 1 / V_original

# Apply linear regression to the reciprocal-transformed data
model_recip = LinearRegression()
model_recip.fit(reciprocal_V.reshape(-1, 1), P_original)

# Predictions for the reciprocal-transformed data
P_pred_recip = model_recip.predict(reciprocal_V.reshape(-1, 1))

# Evaluate the model with R-squared
r2_recip = r2_score(P_original, P_pred_recip)
print(f"R-squared for reciprocal transformation: {r2_recip:.4f}")

# Plotting Reciprocal Transformation Results
plt.subplot(1, 2, 2)
plt.scatter(V_original, P_original, color='blue', label='Original Data')
plt.plot(V_original, P_pred_recip, color='green', label='Reciprocal Transformation Fit')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('Reciprocal Transformation Fit')
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Original data for Boyle's Law (P = c/V)
V_original = np.array([6, 5.7, 4.5, 4.4, 4])  # Volumes in cm^3
P_original = np.array([10, 20, 30, 40, 50])   # Pressures in cm Hg

# Polynomial Regression: Create polynomial features (degree 2 for quadratic)
poly = PolynomialFeatures(degree=2)
V_poly = poly.fit_transform(V_original.reshape(-1, 1))

# Apply linear regression on the polynomial-transformed data
model_poly = LinearRegression()
model_poly.fit(V_poly, P_original)

# Predictions using polynomial regression
P_pred_poly = model_poly.predict(V_poly)

# Evaluate the model with R-squared
r2_poly = r2_score(P_original, P_pred_poly)
print(f"R-squared for polynomial regression: {r2_poly:.4f}")

# Plotting Polynomial Regression Results
plt.figure(figsize=(8, 6))
plt.scatter(V_original, P_original, color='blue', label='Original Data')
plt.plot(V_original, P_pred_poly, color='red', label='Polynomial Regression Fit')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('Polynomial Regression (Degree 2) Fit')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Example data for Boyle's Law (P = c/V)
V_original = np.array([6, 5.7, 4.5, 4.4, 4])  # Volumes in cm^3
P_original = np.array([10, 20, 30, 40, 50])   # Pressures in cm Hg

# Apply log transformation to pressure (P)
log_P = np.log(P_original)

# Apply linear regression on V and log(P)
model_exp = LinearRegression()
model_exp.fit(V_original.reshape(-1, 1), log_P)

# Predict log(P) and convert back to the original P scale
log_P_pred = model_exp.predict(V_original.reshape(-1, 1))
P_pred_exp = np.exp(log_P_pred)  # Exponentiate to get back to original P values

# Calculate R-squared to evaluate the exponential regression
r2_exp = r2_score(P_original, P_pred_exp)
print(f"R-squared for exponential regression: {r2_exp:.4f}")

# Plotting Exponential Regression Results
plt.figure(figsize=(8, 6))
plt.scatter(V_original, P_original, color='blue', label='Original Data')
plt.plot(V_original, P_pred_exp, color='red', label='Exponential Regression Fit')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('Exponential Regression Fit')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Example data for Boyle's Law (P = c/V)
V_original = np.array([6, 5.7, 4.5, 4.4, 4])  # Volumes in cm^3
P_original = np.array([10, 20, 30, 40, 50])   # Pressures in cm Hg

# Define the nonlinear model (Boyle's Law: P = c/V)
def boyles_law(V, c):
    return c / V

# Fit the model to the data using curve_fit
params, covariance = curve_fit(boyles_law, V_original, P_original)

# Extract the fitted parameter 'c'
c_fitted = params[0]
print(f"Fitted constant 'c': {c_fitted:.4f}")

# Predict pressure using the fitted nonlinear model
P_pred_nonlinear = boyles_law(V_original, c_fitted)

# Calculate R-squared for the nonlinear regression
r2_nonlinear = r2_score(P_original, P_pred_nonlinear)
print(f"R-squared for nonlinear regression: {r2_nonlinear:.4f}")

# Plotting Nonlinear Regression Results
plt.figure(figsize=(8, 6))
plt.scatter(V_original, P_original, color='blue', label='Original Data')
plt.plot(V_original, P_pred_nonlinear, color='red', label='Nonlinear Regression Fit (P = c/V)')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('Nonlinear Regression Fit: Boyle\'s Law (P = c/V)')
plt.legend()
plt.show()

