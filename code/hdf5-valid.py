import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from dna_io_1mer import *

file_name_true = 'pos-valid'
file_name_false = 'neg-valid'

seq_vecs_T = hash_sequences_1hot(file_name_true + '.fa')
seq_vecs_F = hash_sequences_1hot(file_name_false + '.fa')

seq_headers_T = sorted(seq_vecs_T.keys())
seq_headers_F = sorted(seq_vecs_F.keys())


train_seqs = []
train_scores = []

for header in seq_headers_T:
    train_seqs.append(seq_vecs_T[header])
    train_scores.append([1])

for header in seq_headers_F:
    train_seqs.append(seq_vecs_F[header])
    train_scores.append([0])

train_seqs = np.array(train_seqs)
train_scores = np.array(train_scores)


h5f = h5py.File('valid.hdf5', 'w')
h5f.create_dataset('x_valid', data=train_seqs)
h5f.create_dataset('y_valid', data=train_scores)
h5f.close()

