#!/usr/bin/env python
# coding: utf-8

# In[900]:


import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# In[903]:


# Initialize weights
def initialize_weights1(num_in, n_hidden, num_out):
    weights = []
    np.random.seed(0) 
    hidden_layer = [{'weights':[np.random.normal(0, 1) for i in range(num_in + 1)]} for i in range(n_hidden)]
    weights.append(hidden_layer)
    hidden_layer = [{'weights':[np.random.normal(0, 1) for i in range(n_hidden + 1)]} for i in range(n_hidden)]
    weights.append(hidden_layer)
    output_layer = [{'weights':[np.random.normal(0, 1) for i in range(n_hidden + 1)]} for i in range(num_out)]
    weights.append(output_layer)
    return weights

def initialize_weights2(num_in, n_hidden, num_out):
    weights = []
    np.random.seed(0) 
    hidden_layer1 = [{'weights':[0 for i in range(num_in + 1)]} for i in range(n_hidden)]
    weights.append(hidden_layer1)
    hidden_layer2 = [{'weights':[0 for i in range(n_hidden + 1)]} for i in range(n_hidden)]
    weights.append(hidden_layer2)
    output_layer = [{'weights':[0 for i in range(n_hidden + 1)]} for i in range(num_out)]
    weights.append(output_layer)
    return weights

# Forward propagate input to a network output
def forward(network, inputs):
    for layer in network:
        new_inputs = []
        for node in layer:
            activation = node['weights'][-1]
            for i in range(len(node['weights'])-1):
                activation += node['weights'][i] * inputs[i]            
            node['output'] = 1/ (1 + np.exp(-activation))
            new_inputs.append(node['output'])
        inputs = new_inputs
    return inputs

# Backpropagate error and store in neurons
def backward(network, actual):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0
				for node in network[i + 1]:
					error += (node['weights'][j] * node['error'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				node = layer[j]
				errors.append(node['output'] - actual[j])
		for j in range(len(layer)):
			node = layer[j]
			node['error'] = errors[j] * (node['output']) * (1.0 - (node['output']))

# Update network weights with error
def update_weights(network, inputs, gamma):
	for i in range(len(network)):
		if i != 0:
			inputs = [node['output'] for node in network[i - 1]]
		for node in network[i]:
			for j in range(len(inputs)):
				node['weights'][j] -= gamma * node['error'] * inputs[j]
			node['weights'][-1] -= gamma * node['error']

def predict(network, inputs):
    outputs = forward(network, inputs)
    if max(outputs)>0.5:
        predcition = 1
    else:
        predcition = -1
    return predcition     

# Train a network for a fixed number of epochs
def SGD(network, x,y, gamma, d,n_epoch, num_out, tolerance=1e-5):
    gamma_0 = gamma
    for epoch in range(n_epoch):
        error = 0
        x, y = shuffle(x, y,random_state = 1)
        for i in range(len(x)):
            predcition = forward(network, x[i])
            if y[i]== -1:
                actual=[-1,0]
            else:
                actual=[0,1]
            backward(network, actual)
            update_weights(network, x[i], gamma)
            gamma = gamma_0/(1+gamma_0/d*epoch)  

'''
# SGD
def fit(weights, x, y, gamma, d,n_outputs, T=100, tolerance=1e-5):
    gamma_0 = gamma
    for j in range(T):
        sum_error = 0
        x, y = shuffle(x, y,random_state = 1)
        for i in range(len(x)):
            row = x[i]
            outputs = forward(network, row)
            expected = [0]*n_outputs
            expected[y[i]] = 1            
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward(network, expected)
            update_weights(network, row, gamma)
        gamma = gamma_0/(1+gamma_0/d*j)
        #convergence
        if (sum_error < tolerance):
            break 
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (j, gamma, sum_error))
'''


# In[904]:


train = pd.read_csv('/home/u1413911/Downloads/train.csv', header=None)
test = pd.read_csv('/home/u1413911/Downloads/test.csv', header=None)

# In[905]:


# Train with initialize_weights as Gussian
train = pd.DataFrame(train,dtype='float64')
train = train.values.tolist()
x = [row[0:(len(train)-1)] for row in train]
y = [round(row[-1]) for row in train]
num_in = len(train[0]) - 1
num_out = len(set([row[-1] for row in train]))

width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights1(num_in, i, num_out)
    SGD(network, x,y, 0.02, 0.01, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))


# In[868]:


# Test with initialize_weights as Gussian
test = pd.DataFrame(test,dtype='float64')
test = test.values.tolist()
x = [row[0:(len(test)-1)] for row in test]
y = [round(row[-1]) for row in test]
num_in = len(test[0]) - 1
num_out = len(set([row[-1] for row in test]))
width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights1(num_in, i, num_out)
    SGD(network, x,y, 0.05, 0.03, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))


# In[870]:


width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights2(num_in, i, num_out)
    SGD(network, x,y, 0.05, 0.03, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))


# In[869]:

width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights2(num_in, i, num_out)
    SGD(network, x,y, 0.05, 0.03, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))


# In[806]:


'''
# Initialize weights
def initialize_weights(n_inputs,n_hidden_states, n_hidden, n_outputs):
    weights = list()
    for i in range(n_hidden):
        np.random.seed(0) 
        hidden_layer = [{'weights':[np.random.normal(0, 1) for i in range(n_inputs+1)]} for i in range(n_hidden_states)]
        weights.append(hidden_layer)
    output_layer = [{'weights':[np.random.normal(0, 1) for i in range(n_hidden_states + 1)]} for i in range(n_outputs)]
    weights.append(output_layer)
    return weights

def activate(weights, inputs):
    activation = 0
    if len(weights) != len(inputs):
        inputs.insert(0, 1)
    for i in range(len(weights)):
        activation = activation + weights[i] * inputs[i]
    return activation

# Forward propagate input to a network output
def forward(weights, inputs):
    for layer in weights:
        if (layer != weights[-1]):
            new_inputs = []
            for node in layer:
                print(node)
                activation = activate(node['weights'], inputs)
                node['output'] = 1 / (1 + np.exp(-activation))
                new_inputs.append(node['output'])
            inputs = new_inputs# the inputs of the next layer
        else:
            new_inputs = []
            for node in layer:
                node['output'] = activate(node['weights'], inputs)
                new_inputs.append(node['output'] )
            inputs = new_inputs
    return inputs

# Make a prediction with a network
def predict(weights, inputs):
    outputs = forward(network, row)
    return outputs.index(max(outputs))

def backward(weights, output):#the output of current layer
    gradient = []
    for i in reversed(range(len(weights))):
        layer = weights[i]
        errors = []
        if (layer != weights[-1]):
            for j in range(len(layer)):
                error = 0
                for node in weights[i + 1]:
                    error += (node['weights'][j] * node['error'])
                errors.append(error)
            for j in range(len(layer)):
                node = layer[j]
                node['error'] = errors[j] * node['output'] * (1 - node['output'])  
                gradient.append(node['error'])    
        else:
            for j in range(len(layer)):
                node = layer[j]
                errors.append(node['output'] - output[j])
            for j in range(len(layer)):
                node = layer[j]
                node['error'] = errors[j] 
                gradient.append(node['error'])

# Update network weights with error
#weight = weight - learning_rate * error * input
def update_weights(weights, inputs, gamma):
    for i in range(len(weights)):
        if i != 0:
            inputs = [node['output'] for node in weights[i - 1]]
        for node in weights[i]:
            for j in range(len(inputs)):
                node['weights'][j] -= gamma * node['error'] * inputs[j]
                
# SGD
def fit(weights, x, y, gamma, d,n_outputs, T=100, tolerance=1e-5):
    gamma_0 = gamma
    for j in range(T):
        sum_error = 0
        x, y = shuffle(x, y,random_state = 1)
        for i in range(len(x)):
            row = x[i]
            outputs = forward(network, row)
            expected = [0]*n_outputs
            expected[y[i]] = 1            
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward(network, expected)
            update_weights(network, row, gamma)
        gamma = gamma_0/(1+gamma_0/d*j)
        #convergence
        if (sum_error < tolerance):
            break 
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (j, gamma, sum_error))
'''


# In[ ]:




