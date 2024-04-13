import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score
from sklearn.metrics import classification_report

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

### SPLIT TRAINING & TEST DATA
X_train, X_test, y_train, y_test = train_test_split(features_normalized, Y, test_size=0.2, random_state=4)

### TRAIN MODEL
k = 4 
KNN = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

### PREDICTIONS
y_predictions = KNN.predict(X_test) 

### EVALUATIONS
KNN_Accuracy_Score = accuracy_score(y_test, y_predictions)
KNN_JaccardIndex = jaccard_score(y_test, y_predictions)
KNN_F1_Score = f1_score(y_test, y_predictions)
print(f'KNN Accuracy Score: {KNN_Accuracy_Score}')
print(f'Jaccard Index: {KNN_JaccardIndex}')
print(f'F1 Score: {KNN_F1_Score}')



# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predictions))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_predictions)
print("Confusion Matrix:\n", conf_matrix)


""" # Cross-validate to find the best 'k'
from sklearn.model_selection import cross_val_score
best_score = 0
best_k = 0
for k in range(1, 20):
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_train, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

print(f"Best K: {best_k} with cross-validated accuracy of {best_score}") """