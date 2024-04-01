import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

# read in the data
my_data = pd.read_csv('drug200.csv',delimiter=",")


### PREPROCESSING ###

# Select the feature matrix i.e. the relevant features and their data
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values # this is a NumPy array with dimensions

# Use dummy variables for the features that are categorical (sex)
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M']) # define the mapping of categories so can assing numbers to them
X[:,1] = le_sex.transform(X[:,1]) # the ':' selects all rows of X, and 1 is 2nd column (colums index starts at 0)

# Use dummy variables for the next feature that's categorical (blood pressure)
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# Use dummy variables for the next feature that's categorical (cholerstral)
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['NORMAL', 'HIGH'])
X[:,3] = le_BP.transform(X[:,3])

# Populate target variable
y = my_data["Drug"] # this is a pandas series object (a 1-d array of data)

### TRAIN/TEST SPLITS ###

# train_test_split returns a 4-element tuple with
# training data subset, testing data, training labels 'y', testing labels 'y'
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

### MODELLING ###

# Create instance of the DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Fit the data with training feature matrix X_trainset with response vector y_trainset
drugTree.fit(X_trainset, y_trainset) # our model is in var drugTree now.

### PREDICTIONS ###
predTree = drugTree.predict(X_testset)
# predTree = drugTree.predict([[23, 0, 0, 1, 25.355]]) # first row data test
# print(predTree)

### EVALUATIONS ###
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


### VIZ ###
'''
# Create .dot file for the tree
export_graphviz(
    drugTree, 
    out_file='tree.dot', 
    filled=True,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    )
'''

# Create tree with plot_tree from sklearn.tree
plt.figure(figsize=(20,10))
plot_tree(
    drugTree, 
    filled=True,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    )
plt.show()