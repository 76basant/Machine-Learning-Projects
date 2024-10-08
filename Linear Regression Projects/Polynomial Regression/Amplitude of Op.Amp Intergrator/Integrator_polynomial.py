
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Your data
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([20, 19, 18, 15, 11, 10, 8])

# Polynomial Regression
poly_degree = 4
poly_features = PolynomialFeatures(degree=poly_degree)
x_poly = poly_features.fit_transform(x)

poly_model = LinearRegression()
poly_model.fit(x_poly, y)

# Predicting on Training Data
y_pred_train = poly_model.predict(x_poly)

# Calculating R² Score on Training Data
r2_train = poly_model.score(x_poly, y)
print(f"R² Score (Training Data): {r2_train:.4f}")

# Calculating Additional Metrics
mse_train = mean_squared_error(y, y_pred_train)
print(f"Mean Squared Error (Training Data): {mse_train:.4f}")

mae_train = mean_absolute_error(y, y_pred_train)
print(f"Mean Absolute Error (Training Data): {mae_train:.4f}")


# Predicting for Plotting
x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_fit_poly = poly_features.transform(x_fit)
y_poly_fit = poly_model.predict(x_fit_poly)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_poly_fit, color='green', label=f'Polynomial Regression (Degree {poly_degree})')
plt.xlabel('f*100(khz)')
plt.ylabel('Amplitude (mVolt)')
plt.title('Integrator Experiment No. 11')
plt.legend()
plt.grid(True)
plt.show()
