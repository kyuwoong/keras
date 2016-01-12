# -*- coding: utf-8 -*-
'''Sequence to sequence learning for auto-encoder
Input: "how are you"
Output: "how are you"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

'''

from __future__ import print_function
from keras.datasets import atis
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.layers import recurrent
from keras.layers import noise
import numpy as np
from six.moves import range

import sitecustomize


import argparse

parser = argparse.ArgumentParser(description='Paraphrase generator')
parser.add_argument('--mode', action='store', dest='mode', default='train')

args = parser.parse_args()



class WordTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, words, maxlen):
        self.words = words
        self.char_indices = dict((c, i) for i, c in enumerate(self.words))
        self.indices_char = dict((i, c) for i, c in enumerate(self.words))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.words)))
        for i, c in enumerate(C):
            try:
                index = self.char_indices[c]
            except KeyError:
                index = self.char_indices['<UNK>']
            X[i, index] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ' '.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = 20
MODEL_FILE_NAME = 'paraphrase_rnn_weights000950.h5'


sentences, words = atis.load_data(maxlen=MAXLEN)

words = ['.'] + words

wtable = WordTable(words, MAXLEN)

print('Vectorization...')
X = np.zeros((len(sentences), MAXLEN, len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    X[i] = wtable.encode(sentence, maxlen=MAXLEN)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(sentences))
np.random.shuffle(indices)
X = X[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))

print(X_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(words)), return_sequences=True))
model.add(RNN(HIDDEN_SIZE))
model.add(noise.GaussianNoise(0.5, input_shape=(MAXLEN, len(words))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributedDense(len(words)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

import os

if os.path.exists(MODEL_FILE_NAME):
    model.load_weights(MODEL_FILE_NAME)


if args.mode == 'train':

    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, 1000):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, X_train, batch_size=BATCH_SIZE, nb_epoch=1,
                  validation_data=(X_val, X_val), show_accuracy=True)
        ###
        # Select 10 samples from the train set at random so we can visualize errors
        print('---- TRAIN ----')
        for i in range(5):
            ind = np.random.randint(0, len(X_train))
            rowX, rowy = X_train[np.array([ind])], X_train[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = wtable.decode(rowX[0])
            correct = wtable.decode(rowy[0])
            guess = wtable.decode(preds[0], calc_argmax=False)
            # print('Q', q[::-1] if INVERT else q)
            print('Q', q)
            # print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
            print('---')

        # Select 10 samples from the validation set at random so we can visualize errors
        print('---- TEST ----')
        for i in range(5):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], X_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = wtable.decode(rowX[0])
            correct = wtable.decode(rowy[0])
            guess = wtable.decode(preds[0], calc_argmax=False)
            # print('Q', q[::-1] if INVERT else q)
            print('Q', q)
            # print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
            print('---')


        if (iteration+1)%50 == 0:
            model.save_weights('paraphrase_rnn_weights%06d.h5' % (iteration+1))

elif args.mode == 'test':
    while True:
        sentence = raw_input('Type a sentence:')
        if sentence == 'quit':
            break
        rowX = np.zeros((1, MAXLEN, len(words)), dtype=np.bool)
        rowX[0,:,:] = wtable.encode(sentence.split(), maxlen=MAXLEN)
        preds = model.predict_classes(rowX, verbose=1)
        print('Q', wtable.decode(rowX[0]))
        print('A', wtable.decode(preds[0], calc_argmax=False))
