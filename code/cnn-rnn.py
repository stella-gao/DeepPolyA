from __future__ import division

import os
os.environ['THEANO_FLAGS'] = "device=gpu"
import sys
sys.setrecursionlimit(15000)

import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.local import LocallyConnected1D
from keras.layers.pooling import AveragePooling1D

from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional


from keras.layers.normalization import BatchNormalization
from residual_blocks import building_residual_block
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sn_sp import *

import matplotlib.pyplot as plt

print 'loading data'
model_file_name = "./model/best_model_cnn_rnn.hdf5";
result_file_name = "./result/best_model_cnn_rnn.res";
print model_file_name;
print result_file_name;

trainmat = h5py.File('./train.hdf5', 'r')
validmat = h5py.File('./valid.hdf5', 'r')
testmat = h5py.File('./test.hdf5', 'r')

X_train = np.transpose(np.array(trainmat['x_train']),axes=(0,2,1))
y_train = np.array(trainmat['y_train'])

X_test = np.transpose(np.array(testmat['x_test']),axes=(0,2,1))
y_test = np.array(testmat['y_test'])

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=7)

X_valid = np.transpose(np.array(validmat['x_valid']),axes=(0,2,1))
y_valid = np.array(validmat['y_valid'])


NUM_FILTER1 = 16
INPUT_LENGTH = 256

print 'building model'

model = Sequential()
##first conv layer
model.add(Convolution1D(input_dim=256,
                        input_length=INPUT_LENGTH,
                        nb_filter=NUM_FILTER1,
                        filter_length=4,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1, init='glorot_normal'))

#model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.7))


##residual blocks
input_length2, input_dim2 = model.output_shape[1:]
nb_filter2 = 32
filter_length = 2
subsample = 1


model.add(Convolution1D(input_dim = input_dim2,
                        input_length = input_length2,
                        nb_filter = nb_filter2,
                        filter_length = 4,
                        border_mode = "valid",
                        activation='relu',
                        subsample_length = 1, init = 'glorot_normal'))

model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.4))
#model.add(Flatten())

input_length0, input_dim0 = model.output_shape[1:]
model.add(LSTM(32, input_dim=input_dim0, input_length=input_length0))
model.add(Dropout(0.2))

model.add(Dense(output_dim=64, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(output_dim=1))
model.add(Activation('sigmoid'))

print 'compiling model'
sgd = SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


print 'running at most 60 epochs'

checkpointer = ModelCheckpoint(filepath=model_file_name, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=500, verbose=1)


print model.summary()
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10000, shuffle=True, validation_data=(X_valid, y_valid), callbacks=[checkpointer,earlystopper])


model.layers[1].get_weights()

tresults = model.evaluate(X_test, y_test)

print 'predicting on test sequences'
model.load_weights(model_file_name)
predrslts = model.predict(X_test, verbose=1)


auc = roc_auc_score(y_test, predrslts)
predrslts_class = model.predict_classes(X_test, verbose=1)
mcc = matthews_corrcoef(y_test, predrslts_class)
acc = accuracy_score(y_test, predrslts_class)
sn, sp = SensitivityAndSpecificity(predrslts, y_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predrslts)
#print auc(false_positive_rate, true_positive_rate)
print tresults
print 'auc:', auc
print 'mcc:', mcc
print 'acc:', acc
print 'sn:', sn
print 'sp:', sp


fw = open(result_file_name, 'w')
fw.write('\t'.join(['tresults', 'auc', 'mcc', 'acc', 'sn', 'sp']) +'\n')
fw.write('\t'.join([str(tresults), str(auc), str(mcc), str(acc), str(sn), str(sp)]) +'\n')
fw.close();

fwr = open("crnn-fpr.txt", 'w')
fwr.write(str(false_positive_rate))
fwr.close();


fwr = open("crnn-tpr.txt", 'w')
fwr.write(str(true_positive_rate))
fwr.close();

