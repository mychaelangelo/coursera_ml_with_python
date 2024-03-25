import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# load the data
df = pd.read_csv("FuelConsumption.csv")

# select the key features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

'''
Splt the data approximately 80/20 for training and test set.
Here we use np.random.rand(n) to gen n-number of floats between 0 and 1.
Sets the array to have True/False items where random num < 0.8
e.g. [True, True, False, True, False] where n-length is 5
'''
msk = np.random.rand(len(df)) < 0.8

# Selects rows from the DataFrame cdf based on the boolean mask msk. 
# Only rows corresponding to True values in msk will be selected
# And vice-versa for rows re: False
train = cdf[msk]
test = cdf[~msk]

# Use sklearn package to model the data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print("Using engine size for predictions:")
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


'''
# Plot the fit line of the model
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()
'''

# Evaluate with these metrics
# MAE - avg abs diff between prediction (test_y_) and actual values (test_y).
# MSE - avg of the sqrs of the diffs between the predicted & actual values
# R2-score - proportion of variance explained by predicted value (test_y_)
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)


print("Using engine size for predictions:")
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
print("")