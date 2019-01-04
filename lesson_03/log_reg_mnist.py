import numpy as np
import tensorflow as tf
import os
import urllib
import gzip
import shutil
import sys

sys.path.append('../')
from utils import read_mnist  

# You can use this to get the data but we'll write our own paser
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('data/mnist', one_hot=True)
path = 'data/mnist'
url = 'http://yann.lecun.com/exdb/mnist'
filenames = ['train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz']

# for filename in filenames:
    # download_url = os.path.join(url, filename)
    # local_dest = os.path.join(path, filename)
    # local_file, _ = urllib.request.urlretrieve(download_url, local_dest)

    # with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3], 'wb') as f_out:
        # shutil.copyfileobj(f_in, f_out)
    # os.remove(local_dest)

batch_size = 128 

train, val, test = read_mnist(path)
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # shuffles the data
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                          train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

w = tf.get_variable('weights', initializer=tf.random.uniform([784]))
b = tf.get_variable('bias', initializer=tf.random.uniform([784]))

pred = tf.sigmoid(tf.matmul(img * w) + b)
loss = -1 * tf.reduce_mean(label*tf.log(pred) + (1-pred) * log(1-tf.log(pred)))

accuracy = 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    for i in range(100):
        sess.run(train_init)
        try:
            while True:
                _, l = sess.run([optimizer, loss])
        except tf.errors.OutOfRangeError:
            pass

    sess.run(test_init)
    try:
        while True:
            ress.run(accuracy)
    except tf.errors.OutOfRangeError:
        pass







