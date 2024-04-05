import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting tool
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import ConfusionMatrixDisplay

### LOAD THE DATA
churn_df = pd.read_csv("ChurnData.csv")

### SELECT FEATURES & PRE-PROCESS
churn_df = churn_df[
    [
        'tenure', 'age', 'address', 'income', 'ed', 'employ', 
        'equip', 'callcard', 'wireless', 'churn'
    ]
]

churn_df['churn'] = churn_df['churn'].astype('int')

### DATA FOR VISUALIZATION (Before Standardization)
tenure = churn_df['tenure'].values
equip = churn_df['equip'].values
churn = churn_df['churn'].values

### NORMALIZE DATA
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

X = preprocessing.StandardScaler().fit(X).transform(X)  # Normalize the dataset



### TRAIN/TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

### MODELLING
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

### USE THE MODEL
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
#print(yhat)
#print(yhat_prob)

### USE THE MODEL
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
#print(yhat)
#print(yhat_prob)

### 3D SCATTER PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Color by churn value
colors = ['blue' if value == 0 else 'red' for value in churn]

# Plot
ax.scatter(tenure, equip, tenure, c=colors, marker='o')
ax.set_xlabel('Tenure')
ax.set_ylabel('Equip')
ax.set_zlabel('Tenure')  # Adjust accordingly if x7 is not 'equip'
ax.set_title('3D Scatter Plot of Tenure, Equip vs. Churn')

plt.show()

### CONTINUE WITH YOUR EXISTING CODE FOR MODEL TRAINING AND EVALUATION
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming `X` and `y` are your features and target variable from the standardized training dataset
# And `LR` is your trained Logistic Regression model

# Generate a mesh grid for tenure and equip values (you might need to adjust ranges)
x_range = np.linspace(churn_df['tenure'].min(), churn_df['tenure'].max(), num=20)
y_range = np.linspace(churn_df['equip'].min(), churn_df['equip'].max(), num=20)
xx, yy = np.meshgrid(x_range, y_range)

# Flatten xx and yy to apply scaler
grid = np.c_[xx.ravel(), yy.ravel()]

# Add dummy columns for the other features to match the input shape of the model
dummy_features = np.zeros((len(grid), X.shape[1]-2))  # X.shape[1] - 2 because we're using 2 features
grid = np.c_[grid, dummy_features]

# Standardize the grid (using the same scaler you used for your model training)
scaler = StandardScaler().fit(X[:, :2])  # Assuming the first two columns are tenure and equip
grid_scaled = scaler.transform(grid[:, :2])

# We replace the first two columns of grid with the standardized values
grid[:, :2] = grid_scaled

# Predict using the logistic regression model
Z = LR.predict(grid)

# Reshape the prediction to match xx's shape
Z = Z.reshape(xx.shape)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original points
colors = ['blue' if value == 0 else 'red' for value in churn]
ax.scatter(tenure, equip, churn, c=colors, marker='o')

# Plot decision boundary
# Note: We're using Z*max(churn) to put the decision boundary at the max churn level for visibility
ax.plot_surface(xx, yy, Z*np.max(churn), color='b', alpha=0.3)

ax.set_xlabel('Tenure')
ax.set_ylabel('Equip')
ax.set_zlabel('Churn')
ax.set_title('3D Scatter Plot with Decision Boundary')

plt.show()
