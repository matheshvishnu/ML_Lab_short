import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
x = np.random.rand(100)
x_train, x_test = x[:50].reshape(-1, 1), x[50:].reshape(-1, 1)
y_train = np.where(x[:50] <= 0.5, 1, 2)

print("KNN Classification results:")
for k in [1, 2, 3, 4, 5, 20, 30]:
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    print(f"\nk={k}:\nPredicted Classes: {model.predict(x_test)}")