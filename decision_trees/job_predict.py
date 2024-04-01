import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.metrics import confusion_matrix
import seaborn as sns

# read in the data
my_data = pd.read_csv('job_predict_made_up.csv',delimiter=",")


### PREPROCESSING ###

# Select the feature matrix i.e. the relevant features and their data
X = my_data[['Age', 'Sex', 'Region', 'Salary']].values # this is a NumPy array with dimensions

# Populate target variable
y = my_data["Job"] # this is a pandas series object (a 1-d array of data)

### TRAIN/TEST SPLITS ###

# train_test_split returns a 4-element tuple with
# training data subset, testing data, training labels 'y', testing labels 'y'
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

### MODELLING ###

# Create instance of the DecisionTreeClassifier
jobTree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Fit the data with training feature matrix X_trainset with response vector y_trainset
jobTree.fit(X_trainset, y_trainset) # our model is in var drugTree now.

### PREDICTIONS ###
predTree = jobTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

### EVALUATIONS ###
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))



### VIZ ###



# Create tree with plot_tree from sklearn.tree
plt.figure(figsize=(20,10))
plot_tree(
    jobTree, 
    filled=True,
    feature_names=['Age', 'Sex', 'Region', 'Salary']
    )
plt.show()


# Calculate confusion matrix
cm = confusion_matrix(y_testset, predTree)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=jobTree.classes_, yticklabels=jobTree.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()