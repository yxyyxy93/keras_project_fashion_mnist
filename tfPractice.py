# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:08:39 2020

@author: alizadeh
"""

import tensorflow as tf

# No compuation is actually performed, these are just a from of declaration:
x=tf.Variable(3, name="x") 
y=tf.Variable(4, name="y")
f = x*y*x +y + 2

init=tf.global_variables_initializer()  #prepare an init node
with tf.Session() as sess:
    #x.initializer.run() #This is the same as tf.get_default_session().run(x.initializer())
    #y.initializer.run() #The with statement 'factors out tf.get_default...
    init.run()
    result=f.eval()
    print("f=",result)
    
# The dependency is computed automatically in Tensorflow:

input("running graph runs, first the inefficient way:")
    
w=tf.constant(3) #These are just *definitions* not yet evaluated
x=w+2
y=x+5
z=x*3
    
with tf.Session() as sess:
    print("y=",y.eval()) # To evaluate y it knows it has to evalute x, and for x to evaluate w
    print("z=",z.eval()) # It knows it has to evaluate x and thus w. It does it *again*
    
    
# To make that x and w are evaluated once we shoulld do i within the same run:
input("running graph runs, Now the efficient way:")

with tf.Session() as sess:
    y_val, z_val= sess.run([y,z])
    print("y=",y_val, "z=", z_val)