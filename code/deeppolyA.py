from __future__ import division

import os
os.environ['THEANO_FLAGS'] = "device=gpu"
import sys
sys.setrecursionlimit(15000)

import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
np.set_printoptions(threshold=np.inf)  
from keras import backend as K
from sklearn.metrics import f1_score
from keras.models import load_model, Model
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
#from seya.layers.recurrent import Bidirectional
#from keras.utils.layer_utils import print_layer_shapes

from keras.layers.normalization import BatchNormalization
#from residual_blocks import building_residual_block
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
from vis.visualization import visualize_saliency
from vis.utils import utils
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from keras.utils import plot_model
from vis.visualization import visualize_saliency
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.utils import utils
from pylab import plot, show
print 'loading data'
#saliency_img_file = "saliency_map.png"
#original_img_file = "original.png"
model_file_name = "./deeppolya.hdf5";
result_file_name = "./deeppolya.res";


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
INPUT_LENGTH = 162

print 'building model'

model = Sequential()

##first conv layer
model.add(Convolution1D(input_dim=4,
                        input_length=INPUT_LENGTH,
                        nb_filter=NUM_FILTER1,
                        filter_length=8,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1, init='glorot_normal'))

#model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.4))


##residual blocks

nb_filter2 = 64
filter_length = 2
subsample = 1


model.add(Convolution1D(input_dim = input_dim2,
                        input_length = input_length2,
                        nb_filter = nb_filter2,
                        filter_length = 6,
                        border_mode = "valid",
                        activation='relu',
                        subsample_length = 1, init = 'glorot_normal'))

model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.4))

#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(output_dim=64, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(output_dim=1, name='preds'))
model.add(Activation('sigmoid'))

#print 'compiling model'
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")


#model = load_model(model_file_name)

print 'compiling model'
sgd = SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


print 'running at most 60 epochs'

checkpointer = ModelCheckpoint(filepath=model_file_name,monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopper = EarlyStopping(monitor='val_loss', patience=200, verbose=1)


print model.summary()
result = model.fit(X_train, y_train, batch_size=128, nb_epoch=10000, initial_epoch=5, shuffle=True, validation_data=(X_valid, y_valid), callbacks=[checkpointer,earlystopper])

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

save_obj(result.history, "run01.pkl")

def load_result(files):
    labels = ["acc", "loss", "val_loss", "val_acc"]
    ret = {"acc":[], "loss":[], "val_loss":[], "val_acc":[],}
    for path in files:
        result = load_obj(path)
        for l in labels:
            ret[l] = ret[l] + result[l]
    return ret

result = load_result(["deeppolya.pkl"])

fig = plt.figure(figsize=(20,4))
plt.subplot("121")
plt.plot(range(len(result["loss"])), result["loss"], label="train_loss")
plt.plot(range(len(result["val_loss"])), result["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("LogLoss")

plt.subplot("122")
plt.plot(range(len(result["acc"])), result["acc"], label="train_acc")
plt.plot(range(len(result["val_acc"])), result["val_acc"], label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('epochs.png')

model.layers[1].get_weights()
tresults = model.evaluate(X_test, y_test)

print 'predicting on test sequences'
model.load_weights(model_file_name)
predrslts = model.predict(X_test, verbose=1)



from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

#from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (4, 162)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'preds')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
outf = np.expand_dims(img, axis=0)
plt.imsave(saliency_img_file, img)


auc = roc_auc_score(y_test, predrslts)
predrslts_class = model.predict_classes(X_test, verbose=1)
mcc = matthews_corrcoef(y_test, predrslts_class)
acc = accuracy_score(y_test, predrslts_class)
sn, sp = SensitivityAndSpecificity(predrslts, y_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predrslts)
#print auc(false_positive_rate, true_positive_rate)
f1 = f1_score(y_test, predrslts_class, average='binary')


print tresults
print 'auc:', auc
print 'mcc:', mcc
print 'acc:', acc
print 'sn:', sn
print 'sp:', sp
print 'f1:', f1


fw = open(result_file_name, 'w')
fw.write('\t'.join(['tresults', 'auc', 'mcc', 'acc', 'sn', 'sp']) +'\n')
fw.write('\t'.join([str(tresults), str(auc), str(mcc), str(acc), str(sn), str(sp)]) +'\n')
fw.close();



13.6421895825563    [[9616   50]  [2157    0]]    [[6790  601]  [3952  480]]    0.526588804892551    0.494607005691132    -0.030785279953475    0.045328249532409    0.813329950097268    0.614903154867631
