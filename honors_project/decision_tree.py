import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# MISC
pd.set_option('future.no_silent_downcasting', True)

### LOAD THE DATA
df = pd.read_csv('Weather_Data.csv')

### PRE-PROCESSING
df_processed = pd.get_dummies(df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

df_processed.replace({'No': 0, 'Yes': 1}, inplace=True)
df_processed.drop('Date', axis=1, inplace=True)
df_processed = df_processed.astype(float)
df_processed['RainTomorrow'] = df_processed['RainTomorrow'].astype(int) # ensure categorisation is int 1 or 0


### SELECT FEATURES AND TARGET VARIABLE
features = df_processed.drop(columns='RainTomorrow').values # convert to NumPy arrays with '.values'
Y = df_processed['RainTomorrow'].values # the '.values' ensures NumPy
features_normalized = StandardScaler().fit_transform(features)

"""
# Print out all features being used
print("Features being used:")
print(df_processed.drop(columns='RainTomorrow').columns.tolist())
"""

### SPLIT TRAINING & TEST DATA
X_train, X_test, y_train, y_test = train_test_split(features_normalized, Y, test_size=0.2, random_state=4)

### TRAIN MODEL
Tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
Tree.fit(X_train, y_train)

### PREDICTIONS
y_predictions = Tree.predict(X_test)

### EVALUATIONS
Tree_Accuracy_Score = accuracy_score(y_test, y_predictions)
Tree_JaccardIndex = jaccard_score(y_test, y_predictions)
Tree_F1_Score = f1_score(y_test, y_predictions)
print(f'Tree Accuracy Score: {Tree_Accuracy_Score}')
print(f'Jaccard Index: {Tree_JaccardIndex}')
print(f'F1 Score: {Tree_F1_Score}')


# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predictions))


# Create tree with plot_tree from sklearn.tree
features_list = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday_No', 'RainToday_Yes', 'WindGustDir_E', 'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW', 'WindDir9am_E', 'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NE', 'WindDir9am_NNE', 'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_S', 'WindDir9am_SE', 'WindDir9am_SSE', 'WindDir9am_SSW', 'WindDir9am_SW', 'WindDir9am_W', 'WindDir9am_WNW', 'WindDir9am_WSW', 'WindDir3pm_E', 'WindDir3pm_ENE', 'WindDir3pm_ESE', 'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE', 'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE', 'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW', 'WindDir3pm_WSW']
plt.figure(figsize=(20,10))
plot_tree(
    Tree, 
    filled=True,
    feature_names=features_list
    )
plt.show()
