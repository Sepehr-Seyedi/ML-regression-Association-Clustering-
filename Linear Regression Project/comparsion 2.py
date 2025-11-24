import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

csv_file_path = 'D:\Download\dataset.csv'

data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# Extract the independent (x) and dependent (y) variables
x = data[:, :2]
y = data[:, 2]

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

#weights using the direct method
X_transpose = np.transpose(x)
X_transpose_X_inv = np.linalg.inv(np.dot(X_transpose, x))
weights_direct_method = np.dot(np.dot(X_transpose_X_inv, X_transpose), y)
print('Model weights (Direct Method):', weights_direct_method)

plt.scatter(x[:, 1], y, color='blue', label='Actual Data')

# Calculate (Direct Method)
x_range_direct = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
y_direct = weights_direct_method[0] + weights_direct_method[1] * x_range_direct + weights_direct_method[2] * x_range_direct**2

# (Direct Method)
plt.plot(x_range_direct, y_direct, color='green', label='Linear Regression (Direct Method)')

# scikit-learn for linear regression
model = LinearRegression()
model.fit(x[:, 1].reshape(-1, 1), y)

# Calculate  (Scikit-learn)
y_sklearn = model.intercept_ + model.coef_[0] * x_range_direct

# Plot (Scikit-learn)
plt.plot(x_range_direct, y_sklearn, color='red', label='Linear Regression (Scikit-learn)')

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

plt.legend()

# Display the plot
plt.show()
