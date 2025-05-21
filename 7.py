import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression on Boston dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data.to_numpy(), boston.target.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Boston Linear Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, R2: {r2_score(y_test, y_pred):.2f}")
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', linewidth=2)
plt.xlabel("True Values"); plt.ylabel("Predictions")
plt.title("Linear Regression (Boston)")
plt.grid(True)
plt.show()

# Polynomial Regression on Auto MPG dataset
auto = fetch_openml(name="autoMpg", version=1, as_frame=True)
data, target = auto.data.dropna(subset=['horsepower']), auto.target.astype(float)
target = target.loc[data.index]
X_hp = data[['horsepower']].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X_hp, target, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=3)
X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.transform(X_test)
lr_poly = LinearRegression().fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)
print("\nAuto MPG Polynomial Regression (Degree=3):")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}, R2: {r2_score(y_test, y_pred_poly):.2f}")

# Plot sorted predictions for smooth curve
sorted_idx = X_test['horsepower'].argsort()
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='True values')
plt.plot(X_test.iloc[sorted_idx], y_pred_poly[sorted_idx], 'r-', linewidth=2, label='Polynomial fit')
plt.xlabel("Horsepower"); plt.ylabel("MPG")
plt.title("Polynomial Regression (Auto MPG)")
plt.legend(); plt.grid(True)
plt.show()
