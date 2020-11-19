'''

Softmax classifier using Gradient Descent on OCR digits dataset

'''

### Recognizing OCR Digits using Softmax regression

#### Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#### Loading the datasets

#### Loading the training dataset
train_data = pd.read_csv('training1.csv')
train_data

#### Loading the test dataset
test_data = pd.read_csv('test1.csv')
test_data

#### Splitting the data into features and class labels

#### Required Utility functions
def mean(X):
    mean = np.mean(X,axis=0)
    return mean
# Utility function to find variance
def variance(X):
    variance = np.var(X,axis=0)
    return variance

#### Splitting training data
X = np.array(train_data.iloc[:,:-1].astype(float))
m,n = X.shape
#### Normalizing and augmenting features
X=(X-mean(X))/np.sqrt(variance(X))
X=np.column_stack((np.ones((m,1),dtype=float),X))
# print(X)
Y = np.array(train_data.iloc[:,-1])
Y

#### Number of class labels
K = 10
# print(K)

#### Splitting test data
X_test = test_data.iloc[:,:-1]
X_test = np.array(X_test.astype(float))
mt = X_test.shape[0]
# print(mt)
#### Normalizing and augmenting features
X_test = (X_test-mean(X_test))/np.sqrt(variance(X_test))
X_test =np.column_stack((np.ones((mt,1),dtype=float),X_test))
# print(X_test.shape)
y_test = test_data.iloc[:,-1]
y_test = np.array(y_test)
y_test

#### Required Functions

'''

This is the indicator function
It takes y values and return 1 if it belongs to that particular class else 0
It takes a kxm matrix as input and returns a kxm matrix with either zeroes or ones accordingly.

'''

def indicatorFunction(y):
    s = y.shape[0]
    result = np.zeros((K,s))
#     print(result[0][0])
    for i in range(s):
#         print(y[i]-1)
#         print(i)
        result[y[i]-1][i] = 1
    return result

# indicatorFunction(Y)

'''

This function calculates the exponential part of the posterior probability.
It takes a theta matrix of size kxn and x matrix of size mxn as input
It returns the negative exponential value of their dot product.
Output is a matrix with kxm size

'''

def myExponentialFunction(theta,x):
    # print(-thetaTranspose*x)
    value = np.dot(theta,x)
#     print(np.exp(-value))
#     print(np.exp(-value).shape)
    return np.exp(-value)

'''

This function takes in the following arguments:
theta - theta matrix of kxn size
x - data matrix of mxn size
All vectors and arrays are numpy
Output is a kxm matrix

'''

def posteriorProbability(theta,x):
    numerator = myExponentialFunction(theta,x.T)
    denominator = np.sum(numerator,axis=0)
    result = numerator/denominator
#     print(numerator)
#     print(denominator)
#     print(result)
    return result
   
# posteriorProbability(np.zeros((K,n+1)),X)

'''

This function takes in input the following arguments:
theta - theta matrix of kxn size
x - data of mxn size
y vector of mx1 size
Output is loss
It uses cross entropy loss function.

'''

def crossEntropyLossFunction(theta,x,y):
    term4 = posteriorProbability(theta,x)
    term3 = np.log(term4,out=term4,where=term4>0)
    term2 = indicatorFunction(y)
    term1 = term2 * term3
    loss = np.sum(term1)
    return -loss

'''

This function computes the gradients of the loss function.
Input:
theta - Weight matrix of size kxn
x - data matrix of size mxn
y - label vector of size mx1
Output:
delta - Matrix of same size as theta, kxn

'''

def gradient(x,theta,y):
    term4 = indicatorFunction(y)
    term5 = posteriorProbability(theta,x)
    term1 = term4 - term5
    delta = np.dot(term1,x)
    return delta

'''

This function is used to find the accuracy of the model
Input:
Target y values and predicted y values of size kxm each
Output:
Float value as percentage and prints number of correctly identifed samples

'''

def accuracy(y,Y):
    count = 0
    total = y.shape[0]
#     print(y.shape[0])
    for i in range(total):
        if y[i] == Y[i]:
            count+=1
    print("These samples were predicted correctly: ")
    print(count)
    return (count/total)*100

'''

This function is used to find the probabilites and figure out the final predictions on the dataset.
Input:
theta matrix of size kxn
x - data matrix of size mxn
Output:
Predictions of size kxm

'''

def findH_Y(theta,x):
    res = posteriorProbability(theta,x)
    result = res.argmax(axis=0)
    result = result + 1
    print(result)
    print(result.shape)
    return result

'''

This is the main function which involved gradient descent and calls the other utility function to find out gradients
and update weights.
Input:
x - training data features, size - mxn
y - training data labels, size - mx1
X - testing data features
y - testing data labels
epochs - number of iterations
alpha - learning rate
Output:
Prints the time taken by the model, plots the costs per iterations and returns the obtained accuracy

'''

def predict(x,y,X,Y,epochs,alpha,fileName):
    begin = time.time()
    theta = np.zeros((K,x.shape[1]))
    costList = []
    for i in range(epochs):
        cost = crossEntropyLossFunction(theta,x,y)
        costList.append(cost)
        deltaTheta = gradient(x,theta,y)
#         print(deltaTheta)
        theta = theta - alpha * deltaTheta
#         print(theta)
    y_pred = findH_Y(theta,X)
    accu = accuracy(y_pred,Y)
    end = time.time()
    print("Time taken: ")
    print(end-begin,end=" s\n")
    print("Accuracy: ")
    print(accu,end=" %\n")
    plt.plot(costList)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Visualizing loss vs number of iterations")
    plt.savefig(fileName)
    plt.show()
    return accu

### Predictions on the data 
#### Accuracy is shown as the final result

#### Predictions on training data
res1 = predict(X,Y,X,Y,10,0.01,"softmax_train_loss1.png")
res1 = predict(X,Y,X,Y,100,0.01,"softmax_train_loss2.png")
res1 = predict(X,Y,X,Y,1000,0.01,"softmax_train_loss3.png")

#### Predictions on test data
res2 = predict(X,Y,X_test,y_test,10,0.01,"softmax_test_loss1.png")
res2 = predict(X,Y,X_test,y_test,100,0.01,"softmax_test_loss2.png")
res2 = predict(X,Y,X_test,y_test,1000,0.01,"softmax_test_loss3.png")

print("Accuracy obtained on the training dataset is: ")
print(res1,end=" %\n")
print("Accuracy obtained on the test dataset is: ")
print(res2,end=" %\n")