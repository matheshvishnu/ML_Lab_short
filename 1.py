import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()
df=pd.DataFrame(housing.data,columns=housing.feature_names)
print(df)

plt.figure(figsize=(12,8))
df.hist(bins=30,figsize=(12,8),color='blue',alpha=0.7)
plt.suptitle("Histograms of Numerical Features",fontsize=14)
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(df)
plt.title("Boxplot for Numerical Features",fontsize=14)
plt.show()

outlier_info = {}
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    outlier_info[col] = outliers.count()

print("\nOutliers Count per Features:")
for col, count in outlier_info.items():
    print(f"{col}: {count} outliers")
