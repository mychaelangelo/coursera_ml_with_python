import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

### MISC
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


### SPLIT TRAINING & TEST DATA
X_train, X_test, y_train, y_test = train_test_split(features_normalized, Y, test_size=0.2, random_state=1)

### TRAIN THE MODEL
SVM = svm.SVC(kernel='rbf')
SVM.fit(X_train, y_train)

### PREDICTIONS
predictions = SVM.predict(X_test)

# Count occurrences of each class in predictions
unique, counts = np.unique(predictions, return_counts=True)
predictions_count = dict(zip(unique, counts))

# Print the results
print("Number of 'No Rain' predictions:", predictions_count.get(0, 0))  
print("Number of 'Rain' predictions:", predictions_count.get(1, 0))  

### EVALUATIONS
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)
print(f'Accuracy Score: {SVM_Accuracy_Score:.4f}')
print(f'Jaccard Index: {SVM_JaccardIndex:.4f}')
print(f'F1 Score: {SVM_F1_Score:.4f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

key_metrics = {
    'matric': ["Accuracy Score", "Jaccard Index", "F1_Score"],
    'values': [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]
}

metrics_df = pd.DataFrame(key_metrics)
print(metrics_df)

### VISUALS

# Compute confusion matrix
cm = confusion_matrix(y_test, predictions)

# reorder the axis, since default is done in asc order of 0, 1
cm = cm[[1, 0], :][:, [1, 0]]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rain', 'No Rain'])

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()