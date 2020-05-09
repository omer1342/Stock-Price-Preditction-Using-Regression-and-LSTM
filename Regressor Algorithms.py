import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
plt.style.use('fivethirtyeight')


#Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end = '2019-12-19')

#Create a new df with only the close column
data = df.filter(['Close'])
#Convert the df to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0: training_data_len, :]
#Split the data int x_train and y_train data sets
x_train = []
y_train = []


for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])


#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Build the regressors models

classifiers = [
    [svm.SVR(), "SVR"], 															#Support vector regression
    [linear_model.SGDRegressor(), "SGD Regression"],								#SGD regression
    [linear_model.BayesianRidge(), "Bayesian Regression"],							#Bayesian regression
    [linear_model.LassoLars(), "Least-angle regression"],							#Least-angle regression
    [linear_model.ARDRegression(), "ARD regression"],								#Automatic Relevance Determination regression (ARD)
    [linear_model.PassiveAggressiveRegressor(), "Passive Aggressive regressor"], 	#Passive aggressive regression
    [linear_model.TheilSenRegressor(), "Theil–Sen estimator"], 						#Theil–Sen estimator
    [linear_model.LinearRegression(), "Linear Regression"]] 						#Linear regression

models = [clf[0].fit(x_train, y_train) for clf in classifiers]

#Creating the testing data set
#Create a new array containing scaled values from index 1544 to 2006
test_data = scaled_data[training_data_len - 60:, :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#Get the models predict price values

for clf, item in zip(models, classifiers):

	#Get the models predict price values
	predictions = clf.predict(x_test)
	predictions = scaler.inverse_transform(predictions.reshape(predictions.shape[0], 1))


	#Get the root mean squared error (RMSE)
	rmse = np.sqrt( np.mean( predictions - y_test )**2 )
	print("clf: %s rmse: %f" % (item[1], rmse))

	#Plot the data
	train = data[:training_data_len]
	valid = data[training_data_len:]
	valid['Predictions'] = predictions
	#Visualize the data
	plt.figure(figsize=(16,8))
	plt.title(item[1])
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	plt.plot(train['Close'])
	plt.plot(valid[['Close', 'Predictions']])
	plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	plt.show()


