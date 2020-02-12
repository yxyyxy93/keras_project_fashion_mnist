# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:46:57 2020

@author: alizadeh
"""
#This script runs over the test data and one by one draws those hand-
#writing images that were mis classified. (You can change != to == in the 
#if statement to look at those which are correctly classified.)

import matplotlib.pyplot as plt


print("label of this images:", test_labels[4])

pred=network.predict_classes(test_images)
for i in range(len(test_images)):
    if test_labels[i]!=pred[i]:
        plt.imshow(test_images[i].reshape(28,28),cmap=plt.cm.binary)
        plt.show()
        print("predicted value of ",i,':', pred[i], " actual value:", test_labels[i])
        if input("continue? y/n")=='n':
            break