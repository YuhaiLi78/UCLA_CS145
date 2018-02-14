# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y

# Applies z-score normalization to the dataframe and returns a normalized dataframe
def applyZScore(dataframe):
    normalized_dataframe = ( dataframe - dataframe.mean(0) ) / dataframe.std(0)
    ########## Please Fill Missing Lines Here ##########
    
    return normalized_dataframe

# train_x and train_y are numpy arrays
# function returns value of beta calculated using (0) the formula beta = (X^T*X)^ -1)*(X^T*Y)
def getBeta(train_x, train_y):
    beta = np.zeros(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    x = train_x
    xT = x.transpose()
    y = train_y

    A = np.linalg.inv(np.dot(xT,x))

    B = np.dot(xT,y)
    
    beta = np.dot(A,B)
    
    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (1) batch gradient descent
def getBetaBatchGradient(train_x, train_y, alpha):
    beta = np.random.rand(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    n, d = train_x.shape
    err_list = list()

    for t in range(10000):
        gradient = 0
        for i in range(n):
            gradient += (np.dot(train_x[i], beta) - train_y[i]) * train_x[i].transpose()
        beta -= alpha * gradient
        y_pred = np.dot(train_x, beta)
        err_list.append(np.sum(np.power(train_y - y_pred, 2)) / float(n))

        # stop?
        if abs(err_list[t] - err_list[t-1]) <= eps :
            break
    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (2) stochastic gradient descent
def getBetaStochasticGradient(train_x, train_y, alpha):
    beta = np.random.rand(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    for i in range(train_x.shape[0]):
        beta += alpha * (train_y[i] - train_x[i].dot(beta)) * train_x[i]
    return beta

# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
def compute_mse(predicted_y, test_y):
    mse = 100.0
    ########## Please Fill Missing Lines Here ##########
    mse = ((predicted_y - test_y) ** 2).mean()
    print
    print 'MSE: ', mse
    
# Linear Regression implementation
class LinearRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 - closed form, 1 - batch gradient, 2 - stochastic gradient)
    # Performs z-score normalization if z_score is 1
    def __init__(self, beta_type, z_score = 0):
        self.alpha = 0.001
        self.beta_type = beta_type
        self.z_score = z_score

        self.train_x, self.train_y = getDataframe('linear-regression-train.csv')
        self.test_x, self.test_y = getDataframe('linear-regression-test.csv')
        
        if(z_score == 1):
            self.train_x = applyZScore(self.train_x)
            self.test_x = applyZScore(self.test_x)
        
        # Prepend columns of 1 for beta 0
        self.train_x.insert(0, 'offset', 1)
        self.test_x.insert(0, 'offset', 1)
        
        self.linearModel()
    
    # Gets the beta according to input
    def linearModel(self):
        if(self.beta_type == 0):
            self.beta = getBeta(self.train_x.values, self.train_y.values)
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 1):
            self.beta = getBetaBatchGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 2):
            self.beta = getBetaStochasticGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        else:
            print 'Incorrect beta_type! Usage: 0 - closed form solution, 1 - batch gradient descent, 2 - stochastic gradient descent'
            
    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "linear-regression-output_betaType_zScore" inside "output" folder
    # Computes MSE
    def predict(self):
        self.predicted_y = self.test_x.values.dot(self.beta)
        np.savetxt('output/linear-regression-output' + '_' + str(self.beta_type) + '_' + str(self.z_score) + '.txt', self.predicted_y)
        compute_mse(self.predicted_y, self.test_y.values)
        
if __name__ == '__main__':
    # Change 1st paramter to 0 for closed form, 1 for batch gradient, 2 for stochastic gradient
    # Add a second paramter with value 1 for z score normalization
    print '------------------------------------------------'
    print 'Closed Form Without Normalization'
    lm = LinearRegression(0)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient Without Normalization'
    lm = LinearRegression(1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient Without Normalization'
    lm = LinearRegression(2)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Closed Form With Normalization'
    lm = LinearRegression(0, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient With Normalization'
    lm = LinearRegression(1, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient With Normalization'
    lm = LinearRegression(2, 1)
    lm.predict()
    print '------------------------------------------------'