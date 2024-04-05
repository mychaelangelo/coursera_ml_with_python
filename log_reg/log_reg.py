import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

### LOAD THE DATA
churn_df = pd.read_csv("ChurnData.csv")

### SELECT FEATURES & PRE-PROCESS
churn_df = churn_df[
    [
        'tenure', 'age', 'address', 'income', 'ed', 'employ', 
        'equip',   'callcard', 'wireless','churn'
    ]
]

churn_df['churn'] = churn_df['churn'].astype('int') # ensure data is int

X = np.asarray(
    churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]
)

y = np.asarray(churn_df['churn'])


X = preprocessing.StandardScaler().fit(X).transform(X) # Normalize the dataset

### TRAIN/TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

### MODELLING
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

### USE THE MODEL
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
#print(yhat)
#print(yhat_prob)

### EVALUATE THE MODEL

#print(jaccard_score(y_test, yhat, pos_label=0)) # print jaccard index

# confusion matrix
cm = confusion_matrix(y_test, yhat, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['churn=1','churn=0'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# classification report
print(classification_report(y_test, yhat))

# log loss
print(f"Log loss is {log_loss(y_test, yhat_prob)}")


# Step 1: Extract the coefficients and the intercept
coefficients = LR.coef_[0]  # Get the coefficients for the first (and only) class
intercept = LR.intercept_

# Step 2: Create a DataFrame for easier plotting
# Include the feature names from your dataset
feature_names = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
coeff_df = pd.DataFrame(coefficients, index=feature_names, columns=['Coefficient'])

# Optionally, you can add the intercept as a separate row if you wish
# coeff_df.loc['intercept', 'Coefficient'] = intercept[0]

# Step 3: Plot the coefficients
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
sns.barplot(x=coeff_df.index, y=coeff_df['Coefficient'], palette="vlag")

plt.title('Coefficients of Logistic Regression Model')
plt.xticks(rotation=45)  # Rotate feature names for better readability
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()

