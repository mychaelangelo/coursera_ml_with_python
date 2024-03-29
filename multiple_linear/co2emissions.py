import matplotlib.pyplot as plt # for plotting and visualization
import pandas as pd # for data manipulation/analysis
import pylab as pl # for MATLAB-like plotting with matplotlib
import numpy as np # for numerical computation & array operations
from sklearn import linear_model # import ML library

# Read in the data
df = pd.read_csv("FuelConsumption.csv")

# Select the features
cdf = df[
    [
        'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
        'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',
        'CO2EMISSIONS'
        ]
    ]

'''
# Plot the data as a preview
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()
'''

# Prepare training & test data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Set up the regression model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)
print('Y-intercept', regr.intercept_)

# Make some predidciton
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))