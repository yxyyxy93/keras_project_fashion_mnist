# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:43:17 2020

@author: alizadeh
"""
#This file is a somewhat more advanced application of Tesnorflow in running
#linear regression. We use three strategies: 1) Direct use of formula, 
#2) Using gradient descent in Tensorflow and 3) Using stichastic batch 
#gradient descent

import numpy as np

# Setup the data. Note dat here is not a Pandas data frame (a la R), but a
#two diemsional "tensor":
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()

m,n=housing.data.shape

#Need the pipeline package to run several stages of data clean up
from sklearn.pipeline import Pipeline

#The Imputer package is a tool for hanbdling missing items in the data
from sklearn.preprocessing import Imputer
#StandardScaler is a package that scales the data using standardization
from sklearn.preprocessing import StandardScaler

# To see the effect of not normalizing comment out the next two statements ...
num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),
                ('std_scalr', StandardScaler()),
                 ])
housing_data_plus_bias= num_pipeline.fit_transform(housing.data)

# ... and uncomment the following.
#housing_data_plus_bias=np.c_[np.ones((m,1)), housing.data]
housing_data_plus_bias=np.c_[np.ones((m,1)), housing_data_plus_bias]

# Set up Tensorflow variables:

import tensorflow as tf
# X is the matrix of independent variables
X= tf.constant(housing_data_plus_bias, dtype=tf.float32,name ="X")
#y is the vector of response variables
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")

Xt=tf.transpose(X)

#Here is the first method: Using the explicit formula for the regression 
#parameters. Define theta as Tensorflow variable and then call the eval() 
#method  on it:
theta=tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt,X)),Xt),y)

with tf.Session() as sess:
    theta_value=theta.eval()
    
print("Regression weights from straightforward matrix calculations:\n",theta_value)

# Using Tensorflow and GD optimizer:


input("Now running gradient descent and:")
      
n_epochs = 1000
learning_rate= 0.01


#tf.nn.batch_normalization()
X= tf.constant(housing_data_plus_bias, dtype=tf.float32,name ="X")
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
#gradients = tf.gradients(mse,[theta])[0]  #invokes automatic gradient

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
trainig_op = optimizer.minimize(mse)
init= tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())  
            print(theta.eval())            
        sess.run(trainig_op)
    best_theta = theta.eval()
print("Regression weights from the GD method:\n", best_theta)

