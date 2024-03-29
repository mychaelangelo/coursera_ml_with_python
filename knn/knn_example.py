import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

### Load the data
df = pd.read_csv('teleCust1000t.csv')

### Select feature set
X = df[
    ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ',
     'retire', 'gender', 'reside']
    ].values

y = df['custcat'].values


### Normalize the data
'''
We do this so that all features are on the same scale.
We aren't changing the relationship between the data points.
We're just transforming the scale of the features.
The transformation adjusts the values so that they have mean 0 and std dev 1.
This makes it easier to for distance metrics (e.g. Euclidiean distance) to work.
See charts/plotting at end of code for visualisations on this.
'''
X_normalized = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

### Split test/training data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=4)


### Set up KNN and train model
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

### Make predictions
yhat = neigh.predict(X_test)
#print(yhat[:5])

### Run eval with jaccard_score function
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

### Let's search for the best K
print("Now to search for the best K value")
Ks = 10
mean_acc = np.zeros((Ks-1)) # set up array of len 9 with [0,0,0..]
std_acc = np.zeros((Ks-1)) # like above, we want array to store accuracy vals

for n in range(1, Ks): # we will start from 1, since k=0 nonsensical
    # train model iteratively
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(f"The best accuracy score was {mean_acc.max()} where k={mean_acc.argmax()+1}")

## Let's plot model accuracy for different Ks
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


'''
### Visualizations on impact of normalization

# Set up the figure for plots
plt.figure(figsize=(12, 6))

# Scatter Plot before Normalization: Age vs. Income
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st subplot
plt.scatter(X[:, 2], X[:, 5], alpha=0.5) # Age on x-axis, Income on y-axis
plt.title('Original Data: Age vs. Income')
plt.xlabel('Age')
plt.ylabel('Income')

# Scatter Plot after Normalization: Age vs. Income
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd subplot
plt.scatter(X_normalized[:, 2], X_normalized[:, 5], alpha=0.5)
plt.title('Normalized Data: Age vs. Income')
plt.xlabel('Normalized Age')
plt.ylabel('Normalized Income')

# Histogram for Region vs. Income before Normalization
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd subplot
# For histogram, we group income by region to see distribution
for region in np.unique(X[:, 0]): # Loop through each unique region value
    plt.hist(X[X[:, 0] == region, 5], bins=20, alpha=0.5, label=f'Region {int(region)}')
plt.title('Original Data: Income by Region')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.legend()

# Histogram for Region vs. Income after Normalization
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th subplot
# Again, group normalized income by region
for region in np.unique(X_normalized[:, 0]):
    plt.hist(X_normalized[X_normalized[:, 0] == region, 5], bins=20, alpha=0.5, label=f'Region {int(region)}')
plt.title('Normalized Data: Income by Region')
plt.xlabel('Normalized Income')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout() # Adjusts subplot params so that subplots fit into the figure area.
plt.show()
'''