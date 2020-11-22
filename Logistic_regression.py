import pandas as pd
import numpy as np
import math as m
import plotly
import plotly.graph_objs as go

# Utility function to find mean
def mean(X):
    mean = np.mean(X,axis=0)
    return mean

# Utility function to find variance
def variance(X):
    variance = np.var(X,axis=0)
    return variance

def plotPredicted(y_pred,x,y):
    markercolor = x[4]
    fig1 = go.Scatter3d(x=x[1],y=x[2],z=x[3],marker=dict(color=markercolor,opacity=1,reversescale=True,colorscale='Blues',size=5),line=dict(width=0.02),mode='markers')
    myLayout = go.Layout(scene=dict(xaxis=dict(title="x1"),yaxis=dict(title="x2"),zaxis=dict(title="x3")))
    plotly.offline.plot({"data":[fig1],"layout":myLayout},auto_open=True,filename=("4DPlot.html"))



# Case #1
# Function to optimize the loss function and figure out the parameters based on the stopping constraint, epsilon
def LogisticRegressionBasedOnEpsilon(lr,X,y,X1):
    n_samples, n_features = X.shape
    # init parameters
    weights = np.zeros(n_features)
    epsilon = 0.01
    # gradient descent
    error_norm = epsilon
    while error_norm >= epsilon:
        # approximate y with linear combination of weights and x, plus bias
        linear_model = np.dot(X, weights)
        # apply sigmoid function
        y_predicted = sigmoid(linear_model)
        # compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        error_norm = np.linalg.norm(db+dw)
        # update parameters
        weights -= lr * dw
    #Predictions
    linear_model = np.dot(X1,weights)
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

# Case #2
# Function to optimize the loss function and figure out parameters based on number of iterations
def LogisticRegressionBasedOnNumIterations(lr,n_iters,X,y,X1):
    n_samples, n_features = X.shape
    # init parameters
    weights = np.zeros(n_features)

    # gradient descent
    for _ in range(n_iters):
        # approximate y with linear combination of weights and x, plus bias
        linear_model = np.dot(X, weights)
        # apply sigmoid function
        y_predicted = sigmoid(linear_model)

        # compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        # update parameters
        weights -= lr * dw
    #Predictions
    linear_model = np.dot(X1,weights)
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

# Utility function to find the value of the sigmoid function of the input
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# Utility function to determine the accuracy of the model
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Loading the training dataset
test_data = pd.read_csv("test_2.csv")
train_data = pd.read_excel("train_2.xlsx")

# Data splitting
# Made into numpy arrays
X_train = np.array(train_data.iloc[:, :-1].astype(float))
y_train = np.array(train_data.iloc[:, -1])
X_test = np.array(test_data.iloc[:, :-1].astype(float))
y_test = np.array(test_data.iloc[:, -1])

# Encoding Category labels
test_classes = []
for value in y_test:
    if value == 'Iris-versicolor':
        test_classes.append(0)
    elif value == 'Iris-virginica':
        test_classes.append(1)

train_classes = []
for value in y_train:
    if value == 'Iris-versicolor':
        train_classes.append(0)
    elif value == 'Iris-virginica':
        train_classes.append(1)

# Pre-processing Input
X1=(X_train-mean(X_train))/np.sqrt(variance(X_train))
X2=(X_test-mean(X_train))/np.sqrt(variance(X_train))
X1=np.column_stack((np.ones((71,1),dtype=float),X1))
X2=np.column_stack((np.ones((29,1),dtype=float),X2))


# Output
predictions = LogisticRegressionBasedOnEpsilon(0.02,X1,train_classes,X2)
# print(test_classes)
# print(predictions)
print("LR classification accuracy for test set:", accuracy(test_classes, predictions)*100)

plotPredicted(predictions,X2,test_classes)

predictions = LogisticRegressionBasedOnEpsilon(0.02,X1,train_classes,X1)
# print(train_classes)
# print(predictions)
print("LR classification accuracy for train set:", accuracy(train_classes, predictions)*100)

predictions = LogisticRegressionBasedOnNumIterations(0.02,80,X1,train_classes,X2)
# print(test_classes)
# print(predictions)
print("LR classification accuracy for test set:", accuracy(test_classes, predictions)*100)

predictions = LogisticRegressionBasedOnNumIterations(0.02,80,X1,train_classes,X1)
# print(train_classes)
# print(predictions)
print("LR classification accuracy for train set:", accuracy(train_classes, predictions)*100)