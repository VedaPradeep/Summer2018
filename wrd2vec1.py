# Word2Vec implementation using n-skip model and using cnkt
from __future__  import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import pickle
import sys

import numpy as np
from cntk.initializer import uniform
from cntk.learner import learning_rate_schedule , sgd , UnitType
from cntk.ops import *
from cntk.trainer import Trainer
from cntk.utils import ProgressPrinter

from sklearn.mainfold import TSNE
import matplotlib.pyplot as plt

#%matplotlib inline

np.random.seed(0)


def  lrmodel(inp, out_dim):
    inp_dim = inp.shape[0]
    wt = parameter(shape=(inp_dim, out_dim), init=uniform(scale=1.0))
    b = parameter(shape=(out_dim), init=uniform(scale=1.0))
    out = times(inp, wt) + b
    return out


def train(emb_size, vocab_size):
    global embeddings, words_to_train
    inp = input_variable(shape=(vocab_size,))
    label = input_variable(shape=(vocab_size,))
    init_width = 0.5 / emb_size
    emb = parameter(shape=(vocab_size, emb_size), init=uniform(scale=init_width))
    embeddings = emb
    embinp = times(inp, emb)

    z = softmax(lrmodel(embinp, vocab_size))  # logistic regression model

    loss = - label * log(z) - ((1 - label) / (vocab_size - 1)) * log(1 - z)
    eval_error = classification_error(z, label)

    lr_per_sample = [0.003] * 4 + [0.0015] * 24 + [0.0003]
    lr_per_minibatch = [x * minibatch_size for x in lr_per_sample]
    lr_schedule = learning_rate_schedule(lr_per_minibatch, UnitType.minibatch)

    learner = sgd(z.parameters, lr=lr_schedule)
    trainer = Trainer(z, loss, eval_error, learner)

    return inp, label, trainer

#Building the dataset
def  build_datasetbuild_d ():
    global data, num_epochs, words_per_epoch, words_to_train
    with open(datapickle, 'rb') as handle:
        data = pickle.load(handle)
    words_per_epoch = len(data)
    words_to_train = num_epochs * words_per_epoch

#Generating mini batches of data
def generate_batch(batch_size, skip_window):
    """ Function to generate a training batch for the skip-gram model. """

    global data, curr_epoch, words_per_epoch, words_seen

    data_index = words_seen - curr_epoch * words_per_epoch
    num_skips = 2 * skip_window
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        words_seen += 1
        data_index += 1
        if data_index >= len(data):
            curr_epoch += 1
            data_index -= len(data)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j, 0] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        words_seen += 1
        data_index += 1
        if data_index >= len(data):
            curr_epoch += 1
            data_index -= len(data)

    return batch, labels

#Creating One-Hot set
def get_one_hot(origlabels):
    global minibatch_size, vocab_size
    labels = np.zeros(shape=(minibatch_size, vocab_size), dtype=np.float32)
    for t in range(len(origlabels)):
        if origlabels[t, 0] < vocab_size and origlabels[t, 0] >= 0:
            labels[t, origlabels[t, 0]] = 1.0
    return labels
#Testing & training
build_dataset()
    inp, label, trainer = train(emb_size, vocab_size)
    print('Model Creation Done.')
    pp = ProgressPrinter(50)
    for _epoch in range(num_epochs):
        i = 0
        while curr_epoch == _epoch:
            features, labels = generate_batch(minibatch_size, skip_window)
            features = get_one_hot(features)
            labels = get_one_hot(labels)

            trainer.train_minibatch({inp: features, label: labels})
            pp.update_with_trainer(trainer)
            i += 1
            if i % 200 == 0:
                print('Saving Embeddings..')
                with open(embpickle, 'wb') as handle:
                    pickle.dump(embeddings.value, handle)

        pp.epoch_summary()

    test_features, test_labels = generate_batch(minibatch_size, skip_window)
    test_features = get_one_hot(test_features)
    test_labels = get_one_hot(test_labels)

    avg_error = trainer.test_minibatch({inp: test_features, label: test_labels})
    print('Avg. Error on Test Set: ', avg_error)