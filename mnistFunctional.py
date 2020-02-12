# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:52:28 2020

@author: alizadeh
"""

#THis code is adopted from "Deep Learning with Python", by F. Chollet, CH. 2
#This is the "functional" form of the model
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

from keras.layers import Input, Flatten, Dense
from keras.models import Model

import numpy as np

np.random.seed(13) #To obtain consistent and reproducible data
inputLayer=Input(shape=(28,28)) 
tmp=Flatten()(inputLayer) #Will turn int a 1D array
#tmp=keras.layers.Concatenate(input,tmp)
tmp=Dense(units=512, activation='relu')(tmp)
outputLayer=Dense(units=10,activation='softmax')(tmp)

network=Model(inputLayer,outputLayer)

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255

from keras.utils import to_categorical

train_labels01 = to_categorical(train_labels)
test_labels01 = to_categorical(test_labels)

network.fit(train_images, train_labels01, epochs=5, batch_size=128)

test_loss, test_acc=network.evaluate(test_images, test_labels01)
print('test_acc:',test_acc)
