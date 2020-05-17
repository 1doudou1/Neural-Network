#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:49 2020

@author: Francis Chen
"""

# Package Area
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

# Here comes the Function
X,Y = load_planar_dataset()  # X is point and Y is label

plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)

shape_X = X.shape    # The raw data
shape_Y = Y.shape    # The label of data
m= Y.shape[1]        # Total Data number, in here is 400

print("The dimension for X is "+str(shape_X))
print("THe dimension for Y is "+str(shape_Y))
print("The dataset has total data "+ str(m))

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

plot_decision_boundary(lambda x: clf.predict(x),X,Y)
plt.title("Logistic Regression") 
LR_predictions = clf.predict(X.T)
print("The accuracy for logistic regression is %d" %float((np.dot(Y,LR_predictions)+\
       np.dot(1-Y,1-LR_predictions))/float(Y.size)*100)+"% right point" ) 
# print: All str is in the "". And the variables are in form of % float sth like that 

def layer_sizes(X,Y):
    n_x = X.shape[0]   # Input  layer
    n_h = 4            # Hidden layer
    n_y = Y.shape[0]   # Output layer
    return(n_x,n_h,n_y)


def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    w2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))
    
    assert(w1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(w2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"w1" : w1,
                  "b1" : b1,
                  "w2" : w2,
                  "b2" : b2
        }
    return parameters

def forward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(w1,X)  + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1" : Z1,
             "A1" : A1,
             "Z2" : Z2,
             "A2" : A2
        }
    return(A2,cache)
   

def compute_cost(A2,Y,parameters):
    m  = Y.shape[1]
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost,float))
    
    return cost

def backward_propagation(parameters,cache,X,Y):
    m  = X.shape[1]
    
    w1 = parameters["w1"] 
    w2 = parameters["w2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y 
    dw2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(w2.T,dZ2),1-np.power(A1,2)) 
    dw1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dw1":dw1,
             "db1":db1,
             "dw2":dw2,
             "db2":db2
        }
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    w1,w2 = parameters["w1"],parameters["w2"]
    b1,b2 = parameters["b1"],parameters["b2"]
    
    dw1,dw2 = grads["dw1"],grads["dw2"]
    db1,db2 = grads["db1"],grads["db2"]
    
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    
    parameters = { "w1":w1,
                   "b1":b1,
                   "w2":w2,
                   "b2":b2
        }
    return parameters

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate = 0.5)
        
        if print_cost:
            if i%1000 ==0:
                print("The",i,"iteration cost is "+str(cost))
    return parameters

def predict(parameters,X):
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#--------------------MAIN  FUNCTION--------------------------
parameters = nn_model(X,Y,n_h = 4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda  x: predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for hidden layer size" +str(4))

predictions = predict(parameters,X)
print('The accuracy is %d' %float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)+'%')

#-------------------END OF MAIN FUNCTION------------------------


plt.figure(figsize=(16,32))
hidden_layer_sizes = [1,2,3,4,5,20,50] # The number of hidden layer
for i,n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5,2,i+1)
    plt.title('Hidden Layer of size %d' %n_h)
    parameters = nn_model(X,Y,n_h,num_iterations=5000)
    plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
    predictions = predict(parameters,X)
    accuracy = float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))\
                     /float(Y.size)*100)        
    print("THE NUMBER OF HIDDEN LAYER:{} Accuracy:{} %".format(n_h,accuracy))
        
    
    
    
