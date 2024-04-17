import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, jaccard_score
import itertools

### LOAD THE DATA

cell_df = pd.read_csv("cell_samples.csv")

""" # visualize the data
malignant_data = cell_df[cell_df['Class']==4][0:50] # get 50 rows of malignant data (i.e. where class = 4)
benign_data = cell_df[cell_df['Class']==2][0:50] # get 50 rows of benign data (i.e. where class = 2)

ax = malignant_data.plot( # plot the malignant cells
    kind='scatter',
    x='Clump',
    y='UnifSize',
    color='Red',
    label='malignant'
)

benign_data.plot( # plot the benign cells
    kind='scatter',
    x='Clump',
    y='UnifSize',
    color='DarkBlue',
    label='benign',
    ax=ax
)
plt.show()  """

### PRE-PROCESS THE DATA

#print(cell_df.dtypes) # check the data types of each column to ensure valid

# we drop the non-numerical values in the BareNuc column
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')


### SELECT FEATURES

feature_df = cell_df[
    [
        'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 
         'BareNuc', 'BlandChrom', 'NormNucl', 'Mit'
     ]
]
X = np.asarray(feature_df)

y = np.asanyarray(cell_df['Class'])

### TRAIN/TEST DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

### MODEL

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

### PREDICTIONS

yhat = clf.predict(X_test)
#print(yhat[:5])

### EVALS

# Compute confusion matrix
cm = confusion_matrix(y_test, yhat, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant','Benign'])

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print(f"Avg F1-score: {f1_score(y_test, yhat, average='weighted'):.4f}")
print(f"Jaccard score: {jaccard_score(y_test, yhat, pos_label=2):.4f}")



# This is a simplified example for illustration purposes
# Let's say you've projected your data into two dimensions: dim1 and dim2

