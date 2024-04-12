import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

### LOAD THE DATA

df = pd.read_csv('Weather_Data.csv')


### PRE-PROCESSING

# this turns non-numerical categorical variables to columns with True/False
df_sydney_processed = pd.get_dummies(
    data=df,
    columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
)

# replace instances of yes/no with 0 and 1
pd.set_option('future.no_silent_downcasting', True)
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

# remove date column
df_sydney_processed.drop('Date', axis=1, inplace=True) #inplace - modify existing df

# convert all data to floats
df_sydney_processed = df_sydney_processed.astype(float) 


### SELECT FEATURES (x values) and Y (target variable)

# create new df to separate features from the target var, raintomorrw
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1) 

# create series object of the target variable (the labels data) and assign to Y 
Y = df_sydney_processed['RainTomorrow']


### SPLIT TRAINING & TEST DATA

X_train, X_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.2, random_state=4
)

### TRAIN MODEL

LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)


### PREDICTIONS
y_predictions = LinearReg.predict(X_test)

### EVALUATIONS
LinearRegression_MAE = np.mean(np.absolute(y_predictions - y_test))
LinearRegression_MSE = np.mean(np.square(y_predictions - y_test))
LinearRegression_R2 = metrics.r2_score(y_test, y_predictions)

metrics_data = {
    'Metric': ['LinearRegression_MAE', 'LinearRegression_MSE', 'LinearRegression_R2'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}

metrics_dframe = pd.DataFrame(metrics_data)
print(metrics_dframe)
