import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)[:, None]
y = np.sin(x).ravel() + np.random.normal(0, 0.2, 100)

# Locally Weighted Regression Function
def lwr(x, y, tau):
    y_pred = []
    for x0 in x:
        w = np.exp(-((x - x0)**2) / (2 * tau**2))
        W = np.diag(w.ravel())
        X = np.hstack((np.ones_like(x), x))
        theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        y_pred.append([1, x0[0]] @ theta)
    return y_pred

# Predict
y_hat = lwr(x, y, tau=0.5)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=10, color='green', label='Original Data')
plt.plot(x, y_hat, color='orange', label='LWR Prediction')
plt.title('Locally Weighted Regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
