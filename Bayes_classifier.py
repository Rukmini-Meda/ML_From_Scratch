# Importing required libraries
import pandas as pd
import numpy as np
import math as m

# Utility function to find mean
def mean(X):
    mean = np.mean(X,axis=0)
    return mean

# Utility function to find variance
def variance(X):
    variance = np.var(X,axis=0)
    return variance

# Utility function to separate training data into classes for computation
def separate_class(X,y):
    
#     Dictionary containing the dataset segregated classwise
    separated = dict()
    n = X.shape[0]
    for i in range(n):
        vector = X[i]
        label = y[i]
        
#         Adding to the dictionary according to label
        if label not in separated:
            separated[label] = list()
        separated[label].append(vector)
        
#     Converting into numpy arrays for easy computations
    for label in separated:
        separated[label] = np.array(separated[label])
    return separated

# Utility function to find the covariance matrix of individual classes
def covariance_matrix(sep):
    
#     Dictionary to store individual covariance matrices
    covmatrices = dict()
    for label in sep:
        x = np.array(sep[label]).T
        covmatrices[label] = np.cov(x)
    return covmatrices

# Function that uses the discriminant function and predicts the label
def discriminant_predict(X_train,y_train,X_test):
    
#     Separated the data class wise
    sep1 = separate_class(X_train,y_train)
    
#     Finds the individual covariance matrices of all classes
    sep = covariance_matrix(sep1)
    n = X_test.shape[0]
    
#     Our predictions or classified labels in a list
    y_pred = list()
    for i in range(n):
        x = X_test[i].reshape(4,1)
        res = dict()
        for label in sep:
            
#             Inverse of the covariance matrix
            sigmainv = np.linalg.inv(sep[label])
    
#     Mean of the data of that particular label
            meani = mean(sep1[label]).reshape(4,1)
#     Required terms for computation
            W = (-1/2)*(sigmainv)
            w = sigmainv.dot(meani)
            dets = np.linalg.det(sep[label])
            wi0 = ((-1/2)*(((meani.T).dot(sigmainv)).dot(meani)))+((-1/2)*(m.log(dets)))+(m.log(1/3))
            term1 = ((x.T).dot(W)).dot(x)
            term2 = (w.T).dot(x)
            term3 = wi0
            
#             Value of the discriminant function
            g = term1+term2+term3
            res[label] = g
#         Finds the maximum g value in the dict
        finl = max(res, key=res.get)
        y_pred.append(finl)
    y_pred = np.array(y_pred)
    return y_pred

# Utility function find the accuracy of predictions made
def accuracy(y_pred,y_test):
    n = y_pred.shape[0]
    count = 0
    for i in range(n):
        if y_pred[i] == y_test[i]:
            count+=1
    return count/n


# Loading the training dataset
df = pd.read_csv("train.csv")

# Data splitting

# Dataset made into a list of lists
dataset = list()
n = len(df)
for i in range(n):
    vector = list(df.iloc[i,:])
    dataset.append(vector)

# Independent features extracted as  list of lists
X = list()

for i in range(n):
    vector = dataset[i][0:4]
    X.append(vector)
    
# Dependent feature or label extracted into a list    
y = list()
for i in range(n):
    y.append(dataset[i][4])

# Made into numpy arrays for easy computation
X_train = np.array(X)
y_train = np.array(y)

# Loading the test dataset
df = pd.read_csv("test.csv")

# Data splitting

# Dataset made into a list of lists
dataset = list()
n = len(df)
for i in range(n):
    vector = list(df.iloc[i,:])
    dataset.append(vector)

# Independent features extracted as list of lists
X = list()

for i in range(n):
    vector = dataset[i][0:4]
    X.append(vector)
    
# Dependent feature or label extracted into a list
y = list()
for i in range(n):
    y.append(dataset[i][4])
    
# Made into numpy arrays
X_test = np.array(X)
y_test = np.array(y)

# Calling the prediction function to obtain the classified labels of test dataset
y = discriminant_predict(X_train,y_train,X_test)
# Calling the prediction function to obtain the classified labels of train dataset
y1 = discriminant_predict(X_train,y_train,X_train)

val = accuracy(y,y_test)
print("The test set accuracy is ", val*100, "%.")

val = accuracy(y1,y_train)
print("The training set accuracy is ",val*100,"%.")