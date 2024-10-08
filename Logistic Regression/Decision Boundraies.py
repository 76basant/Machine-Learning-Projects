#Decision boundary 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create synthetic tumor size data (X) and their malignancy labels (y)
np.random.seed(42)
# Tumor sizes (X) between 0 and 10
X = np.random.rand(100, 1) * 10
# Labels: 1 = malignant, 0 = benign (based on size, tumors > 5 are more likely malignant)
y = (X > 5).astype(int).ravel()

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Visualize the decision boundary

# Scatter plot of the training data
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.xlabel('Tumor Size')
plt.ylabel('Malignancy (1 = Malignant, 0 = Benign)')
plt.title('Tumor Size vs Malignancy')

# Generate a range of tumor sizes for plotting the logistic regression curve
X_range = np.linspace(0, 10, 100).reshape(-1, 1)

# Predict the probability of malignancy using the logistic model
y_prob = model.predict_proba(X_range)[:, 1]  # Get the probability for class 1 (malignant)

# Plot the logistic regression curve (decision boundary curve)
plt.plot(X_range, y_prob, color='blue', label='Logistic Regression Curve')

# Step 5: Find and plot the decision boundary (P = 0.5)
decision_boundary = X_range[np.argmin(np.abs(y_prob - 0.5))]  # The point where P(malignant) ≈ 0.5
plt.axvline(x=decision_boundary, color='green', linestyle='--', label=f'Decision Boundary at {decision_boundary[0]:.2f}')

# Add a legend
plt.legend()

# Step 6: Display the plot
plt.show()

# Step 7: Print the decision boundary
print(f'Tumor size decision boundary: {decision_boundary[0]:.2f}')
