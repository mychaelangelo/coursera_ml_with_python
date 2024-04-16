import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
X_train, X_test, y_train, y_test = train_test_split(features_normalized, Y, test_size=0.2, random_state=1)

### TRAIN MODEL
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

### PREDICTIONS
predictions = LR.predict(X_test)
predict_proba = LR.predict_proba(X_test)

### EVALUATIONS
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)
print(f'Accuracy Score: {LR_Accuracy_Score}')
print(f'Jaccard Index: {LR_JaccardIndex}')
print(f'F1 Score: {LR_F1_Score}')
print(f'Log Loss: {LR_Log_Loss}')



### Plotting the coefficients

coefficients = LR.coef_[0] # Get the coefficients
feature_names = df_processed.drop(columns='RainTomorrow').columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)
top_features = coef_df.head(10) # select top 15 features

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=top_features)
plt.title('Top 10 Logistic Regression Coefficients')
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)

# Visualize confusion matrix
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, 
    display_labels=["No Rain", "Rain"]
)
fig, ax = plt.subplots(figsize=(8, 6))
cm_display.plot(ax=ax)

ax.set_title('Confusion Matrix', pad=20)
ax.xaxis.set_label_position('top') 
ax.set_xlabel('Predicted Values', labelpad=10)
ax.set_ylabel('Actual Values', labelpad=10)
plt.tight_layout()
plt.show()

