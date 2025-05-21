import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names

x_scaled = StandardScaler().fit_transform(x)
x_pca = PCA(n_components=2).fit_transform(x_scaled)

df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])
df['Target'] = y

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']

for i, color in enumerate(colors):
    plt.scatter(df[df['Target'] == i]['PC1'], df[df['Target'] == i]['PC2'],
                color=color, alpha=0.6, label=target_names[i])

plt.title('PCA of Iris Dataset (2 Components)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Target')
plt.grid(alpha=0.3)
plt.show()
