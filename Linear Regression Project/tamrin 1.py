import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv_file_path = 'D:\Download\dataset.csv'

data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

x = data[:, :2] 
y = data[:, 2]   

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

#weights using the direct method
X_transpose = np.transpose(x)
X_transpose_X = np.dot(X_transpose, x)
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_y = np.dot(X_transpose, y)
weights = np.dot(X_transpose_X_inv, X_transpose_y)
print('Model weights:', weights)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:, 1], x[:, 2], y, color='blue', label='Actual Data')

x1_range = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 10)
x2_range = np.linspace(np.min(x[:, 2]), np.max(x[:, 2]), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y = weights[0] + weights[1] * X1 + weights[2] * X2

ax.plot_surface(X1, X2, Y, alpha=0.5, color='red', label='Linear Regression Line')

# Label the axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

ax.legend()

# Display the plot
plt.show()
