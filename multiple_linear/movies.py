import matplotlib.pyplot as plt # for plotting and visualization
import pandas as pd # for data manipulation/analysis
import pylab as pl # for MATLAB-like plotting with matplotlib
import numpy as np # for numerical computation & array operations
from sklearn import linear_model # import ML library
import warnings
warnings.filterwarnings('ignore', message="X has feature names, but LinearRegression was fitted without feature names")


# Read in the data
df = pd.read_csv("movies.csv")

# Select the features
cdf = df[['IMDBRating', 'Metascore', 'Year']]

'''
# Plot the data as a preview
plt.scatter(cdf.IMDBRating, cdf.Metascore, color='blue')
plt.xlabel("IMDB Rating")
plt.ylabel("Metascore")
plt.show()
'''


# Prepare training & test data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Set up the regression model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['IMDBRating', 'Year']])
y = np.asanyarray(train[['Metascore']])
regr.fit(x, y)
# print('Coefficients: ', regr.coef_)
# print('Y-intercept', regr.intercept_)


# Make some predidciton
y_hat= regr.predict(test[['IMDBRating', 'Year']])
x = np.asanyarray(test[['IMDBRating', 'Year']])
y = np.asanyarray(test[['Metascore']])
print("Mean Squared Error (MSE) : %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


def predict_metascore(imdb_rating, year):
    predicted_metascore = regr.predict([[imdb_rating, year]])
    rounded_metascore = round(predicted_metascore[0][0])
    return rounded_metascore  # Return the rounded predicted value

# Demo
imdb_rating = 8.0 
year = 2021  #
predicted_metascore = predict_metascore(imdb_rating, year)
print(f"Predicted Metascore for a movie with IMDb rating {imdb_rating} and year {year}: {predicted_metascore}")

