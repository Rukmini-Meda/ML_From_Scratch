'''

Feed Forward Neural Network implementation for OCR digit recognition
Assumptions:
Number of hidden layer = 1
Number of neurons in the hidden layer = 250
Loss function is Squared error loss function
Activation function is Sigmoid function


'''
### Recognizing OCR Digits using Softmax regression

#### Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#### Loading the datasets

#### Loading the train dataset
train_data = pd.read_csv('training1.csv')
train_data

#### Loading the test dataset
test_data = pd.read_csv('test1.csv')
test_data

#### Splitting the data into features and class labels

#### Required functions

def mean(X):
    mean = np.mean(X,axis=0)
    return mean

def variance(X):
    variance = np.var(X,axis=0)
    return variance

#### Splitting training data

X = np.array(train_data.iloc[:,:-1].astype(float))
m,n = X.shape

#### Normalizing and augmenting the data
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

#### Normalizing and augmenting the data
X_test = (X_test-mean(X_test))/np.sqrt(variance(X_test))
X_test =np.column_stack((np.ones((mt,1),dtype=float),X_test))
# print(X_test.shape)
y_test = test_data.iloc[:,-1]
y_test = np.array(y_test)
y_test

'''

This function computes the value of the sigmoid function
Input:
x - numpy array
Output:
Numpy array of sigmoid values (0-1)

'''

def sigmoid(x):
  return 1/(1+np.exp(-x))

'''

This function computes the net value in the feed forward network.
In simple terms it finds the dot product of the 2 matrices.
Input:
W - weight matrix, sizes in this case are 250xn and kx250
Output:
Matrix, size of either 250xm or kxm

'''

def net(W,x):
  # print("W")
  # print(W.shape)
  # print("x")
  # print(x.shape)
  return np.dot(W,x.T)

'''

Computes random weights for initialization
Weights are taken from standard normal distribution (z-distribution)
Input:
size - a tuple
Output:
weights - a matrix of random numbers from z distribution

'''

def randomWeights(size):
  weights = np.random.normal(loc=0,scale=1,size=size)
  return weights

'''

Utility function to find the y values

'''

def findH_y(x,W1,W2):
  net1 = net(W1,x)
  y1 = sigmoid(net1)
  net2 = net(W2,y1.T)
  y_pred = sigmoid(net2)
  return y_pred

'''

This computes the squared error loss function
Input:
target and predicted y values of size kxm each
Output:
Loss matrix for all data points of size kxm

'''

def squaredLossFunction(target,y_pred):
  term = (target-y_pred)*(target-y_pred)
  return -(term/2)

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
    print("The number of samples that were predicted correctly: ")
    print(count)
    return (count/total)*100

'''

Utility function to find the derivative of the sigmoid

'''

def sigmoidDerivative(value):
  return sigmoid(value) * (1 - sigmoid(value))

'''

Utility function to find the final class labels from obtained probability matrix

'''

def makePredictions(y):
  result = np.argmax(y,axis=0)
  result = result+1
  return result

'''

This is the main function which runs gradient descent and updates weights.
Input:
x - training data features
X - test data features
y - training data labels
Y - test data labels
epochs - number of iterations
alpha - learning rate

'''

def predict(x,y,X,Y,epochs,alpha,fileName):
  begin = time.time()
  W1 = randomWeights((250,x.shape[1]))
  W2 = randomWeights((K,250))
  costList = []
  for i in range(epochs):
    ypred = findH_y(x,W1,W2)
    cost = squaredLossFunction(y,ypred)
    costList.append(np.sum(cost))
    net1 = net(W1,x)
    y1 = sigmoid(net1)
    net2 = net(W2,y1.T)
    term1 = -(y-ypred)
    term2 = sigmoidDerivative(net2)
    term3 = sigmoidDerivative(net1)
    term4 = term1 * term2 #kxm
    term5 = np.dot(W2.T,term4)
    term6 = term5 * term3
    dW1 = np.dot(term6,x)
    deltaK = (y - ypred) * sigmoidDerivative(net2)
    dW2 = -np.dot(deltaK,y1.T)
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2

  y_pred = findH_y(X,W1,W2)
  y_final = makePredictions(y_pred)
  accu = accuracy(y_final,Y)
  end = time.time()
  print("Time taken is: ")
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

#### Making prections on the data

#### On training data
res1 = predict(X,Y,X,Y,10,0.01,"ffnn_loss1.png")
res1 = predict(X,Y,X,Y,100,0.01,"ffnn_loss2.png")

print("Accuracy obtained on training dataset: ")
print(res1,end=" %\n")

#### On test data
res2 = predict(X,Y,X_test,y_test,10,0.01,"ffnn_loss4.png")
res2 = predict(X,Y,X_test,y_test,100,0.01,"ffnn_loss5.png")

print("Accuracy obtained on test dataset: ")
print(res2,end=" %\n")

