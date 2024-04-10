import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

### Load the data
cust_df = pd.read_csv('Cust_Segmentation.csv')


### Pre-process the data
df = cust_df.drop('Address', axis=1) # drop the address column (axis=1 indicates column drop, not row)

X = df.values[:,1:] # get all rows and columns but exclude first column
X = np.nan_to_num(X) # replace all non-numbers with zero
Clus_dataSet = StandardScaler().fit_transform(X) # standardize the data 

### Model
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_

### Visualizations

# 2d

df["Clus_km"] = labels # add column for the labels to the data frane

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# 3d
# 3d plot with updated method for Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Updated line
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

# Ensure you're plotting the standardized data here if it makes sense for your visualization context
# Or adjust the plot to use the non-standardized data but be aware of potential scaling issues
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(float))
plt.show()