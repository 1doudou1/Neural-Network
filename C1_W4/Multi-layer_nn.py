#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:34:24 2020
@author: Francis Chen
"""

#---------------------DEFINE PACKAGE------------------------------------------
import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils
np.random.seed(1)   

#-------------------Programming Area------------------------------------------
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x  = test_x_flatten / 255
test_y  = test_set_y

layers_dims = [12288,30,7,5,1]    # five layer model 

# Through the dimension we see the weight and bias
def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}       # dictionary
    L=len(layers_dims)  
    for l in range(1,L):  # For Each layer. 1-5 in this case
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1])  # All the column stroge the input feature and all the rows representing the number of nodes in next layer
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))  
        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l],1))
    return parameters

# Through the input training data and weight and bias we finish forward prob.
def L_model_forward(X,parameters):
    caches = []
    A=X                      # X is acuallly the training data
    L=len(parameters) //2    # fix  
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)],\
                                             parameters['b'+str(l)],"relu")
        caches.append(cache)
    AL , cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],\
        "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

# Calculating the cost
def compute_cost(AL,Y):
    m=Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m
    cost = np.squeeze(cost)    # list,tuple --> array
    assert(cost.shape == ())
    return cost

# Through the cost we derive the grads
def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    return grads

# Through the parameters, grads and the hyper-parameter learning_rate, we finish the update
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters) //2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]
        
    return parameters

#--------------------------------Finish One circut-----------------------------

def initialize_parameters(n_x,n_h,n_y):
    w1 = np.random.randn(n_h,n_x)*0.01  # Each Hidden nodes is coressponding to multiple input features
    b1 = np.zeros((n_h,1))              # The number of hidden layer is equal to the number of b needed
    w2 = np.random.randn(n_y,n_h)*0.01  # Each output args has n_h hidden layer connect to it
    b2 = np.zeros((n_y,1))              # Intialize b to be 0.    
    assert(w1.shape == (n_h,n_x))       # If the input Args are wrong, the use assert we could identify the wronging problems 
    assert(b1.shape == (n_h,1))
    assert(w2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))    
    parameters={"w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2
        }
    return parameters

def linear_activation_forward(A_prev,W,b,activation):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z,cache

def linear_backward(dZ,cache):
    A_prev, W, b=cache
    m  = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) /m
    db = np.sum(dZ,axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T,dZ)
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA,cache, activation = "relu"):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev, dW, db

    
def Main(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000, print_cost=False, isPlot = True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0,num_iterations):
        AL, caches = L_model_forward(X,parameters)
        cost  = compute_cost(AL, Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("The",i,"iteration cost is ", np.squeeze(cost))
    if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations(per tens)')
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()
    return parameters

parameters  = Main(train_x,train_y,layers_dims,learning_rate=0.0075,num_iterations=2500,print_cost=True,isPlot=True)
# 被程序直接调用的需要摆在最后面（？），其他程序编译关系随意（？）
    
    
    