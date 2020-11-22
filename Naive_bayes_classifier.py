# Importing required libraries
import pandas as pd
import numpy as np

# Loading training and test datasets
train = pd.read_csv('train.csv')
train.rename(columns = {'0':'sepal_length','1':'sepal_width','2':'petal_length', '3':'petal_width','4':'Species'}, inplace = True)
test = pd.read_csv("test.csv")

print(train.keys())

# Extracting the label list
print ()
print ("Class Labels are:")
Species_list=['Iris-setosa','Iris-versicolor','Iris-virginica']
print(Species_list)

# Sample statistics calculated
mean=train.groupby('Species').mean()
stddev=train.groupby('Species').std()

# Computation of pdf for Naive bayes classification
def naive_bayes(x,i,j):
    return ((1 / np.sqrt(2 * np.pi * stddev.iloc[i,j]**2)) * np.exp((-1*((x - mean.iloc[i,j])**2)) / (2 * stddev.iloc[i,j]**2)))

# Function to make predictions on the dataset
def predict(x):
    best=0
    best1=0
    for j in range(len(Species_list)):
        p=1
        for i in range(4):
            p=p*naive_bayes(x[i],j,i)
        if(best==0 or best<p):
            best=p
            best1=j
            #print("*",j)
    #print(best1)
    return best1

# Classifying the training dataset

d=0
for i in range(len(train)):
    X = train.iloc[i]
    y=predict(X)
    if(Species_list[y]==X['Species']):
        d=d+1
print()     
print("Accuracy of training set = ",d/len(train)*100,"%")

# Classifying the test dataset

c=0
for i in range(len(test)):
    X = test.iloc[i]
    y=predict(X)
    if(Species_list[y]==X['4']):
        c=c+1
print()     
print("Accuracy of test set = ",c/len(test)*100,"%")