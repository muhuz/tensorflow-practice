import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('~/Personal/tensorflow-practice')
import utils

# enable eager execution
tfe.enable_eager_execution()

# Get the data

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016

import word2vec_utils

def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES,
                                        VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW,
                                        VISUAL_FLD)

class Word2Vec(object):
    def __init__(self, vocab_size, embed_size, num_sampled=NUM_SAMPLED):
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.embed_matrix = np.random.rand(vocab_size, embed_size) 
        self.nce_weight = 
        self.nce_bias = 

    def compute_loss(self, center_words, target_words):
        #look up embeddings

        # compute loss
        loss = tf.reduce_mean()
        return loss

