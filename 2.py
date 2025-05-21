import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)

corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.title("Correlation Matrix of California housing features")
plt.show()
sns.pairplot(df)
plt.show()