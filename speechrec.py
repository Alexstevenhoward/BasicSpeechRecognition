#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:37:50 2018

@author: alexanderhoward
"""

import tflearn
import speech_data

#hyperparams = learning rate
learn_rate = 0.0001
train_iters = 250000

batch = word_batch = speech_data.mfcc_batch_generator(64)
#each wave file is a recording of spoken digit
X, Y = next(batch)

#our test and training sets split up
trainX, trainY = X, Y
testX, testY = X, Y

inputMatrix = tflearn.input_data([None, 20 , 80])
lstm = tflearn.lstm(inputMatrix, 128, dropout=0.75)
#Long Short Term Memory Neural Network function
activation = tflearn.fully_connected(lstm, 10, activation='softmax')
#Insert our hyper parameters for the neural network
net = tflearn.regression(activation, optimizer='adam', learning_rate=learn_rate, loss='categorical_crossentropy')

#generate the module
model = tflearn.DNN(net, tensorboard_verbose=0)

while 1:
    model.fit(trainX, trainY, n_epochs=10, validation_set=(testX, testY), show_metric=True,batch_size=64)
    _y=model.predict(X)
model.save('tflearn.lstm.model')
print(_y)
