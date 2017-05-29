import pandas as pd
import numpy as np
import math

def getData(link):
	data = pd.io.parsers.read_csv(
		filepath_or_buffer=link,
		skiprows = 1,
		header=None,
		sep='	')
	data.dropna(how='all', inplace=True)
	data.tail()
	return data

def get_x_values(df):
	x = df.iloc[:,1:96]
	x.loc[:, 96] = np.array([1] * len(x))
	return x

def getTrainRMSE(Lambda, df_train):
	y = df_train[[0]]
	x = get_x_values(df_train)
	W = ordinaryLeastSquares(x, y, Lambda)
	yHat = yPredictions(x, W)
	rmse = RMSE(y, yHat)
	return rmse, W

def getTestRMSE(W, df_test):
	y_test = df_test[[0]]
	x_test = get_x_values(df_test)
	yNew = yPredictions(x_test, W)
	rmse = RMSE(y_test, yNew)
	return rmse

def ridgeRegressionClosed(df_train, df_test):
	#Perform linear regression directly using the closed form solution
	Lambda = getLambda(df_train, 400)
	rmse, W = getTrainRMSE(Lambda, df_train)
	print "Traning RMSE: "
	print rmse
	y_test = df_test[[0]]
	x_test = get_x_values(df_test)
	yNew = yPredictions(x_test, W)
	rmse = getTestRMSE(W, df_test)
	print "Test RMSE: "
	print rmse

def getLambda(df_train, Lambda):
	#performing cross validation with k=5
	#Begin by running the solver with λ = 400. 
	#Then, cut λ down by a factor of 2 and run again. 
	#Continue the process of cutting λ by a factor of 2 until there's models for 10 values of λ in total.
	size = len(df_train) / 5
	tempLambda = Lambda
	bestRMSE = float("inf")
	test = []
	train = []
	for _ in range(10):
		tempRMSE = 0
		for i in range(5):
			start = i*size
			end = (i+1)*size
			test = df_train[start:end]
			train = df_train.drop(df_train.index[start:end])
			ignore, tempW = getTrainRMSE(tempLambda, train)
			tempRMSE += getTestRMSE(tempW, test)
		avgRMSE = tempRMSE / 5
		if avgRMSE < bestRMSE:
			bestRMSE = avgRMSE
			Lambda = tempLambda
		tempLambda = tempLambda / 2

	return Lambda

def ordinaryLeastSquares(x, y, Lambda):
	I = np.identity(96)
	lambda_matrix = Lambda * I
	xTranspose = np.transpose(x)
	xproduct = np.add(np.dot(xTranspose, x), lambda_matrix)
	xproductInv = np.linalg.inv(xproduct)
	w = np.dot(np.dot(xproductInv, xTranspose), y)
	return w

def yPredictions(x, w):
	y_array = [0] * len(x)
	i = 0
	wTranspose = np.transpose(w)
	for index, row in x.iterrows():
		y = np.dot(wTranspose, row)
		y_array[i] = y[0]
		i += 1
	return y_array

def RMSE(y, yHat):
	y = y.as_matrix()
	M = len(y)
	sum = 0
	for i in range(M):
		dif = (yHat[i] - y[i]) ** 2
		sum += dif
	sum = sum / M
	return math.sqrt(sum)


if __name__ == "__main__":

	trainLink = 'http://www.cse.scu.edu/~yfang/coen129/crime-train.txt'
	df_train = getData(trainLink)
	testLink = 'http://www.cse.scu.edu/~yfang/coen129/crime-test.txt'
	df_test = getData(testLink)
	ridgeRegressionClosed(df_train, df_test)

