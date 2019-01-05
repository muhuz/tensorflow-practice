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
label = tf.cast(label, tf.float32)

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

w = tf.get_variable('weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable('bias', shape=(1, 10), initializer=tf.zeros_initializer())

pred = tf.matmul(img, w) + b
# loss = -1 * tf.reduce_mean(label*tf.log(pred) + (1-label) * tf.log(1-tf.log(pred)))
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

pred_vals = tf.nn.softmax(pred)
correct_preds = tf.equal(tf.argmax(pred_vals, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

# accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), dtype=tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average Loss: Epoch {0}: {1}'.format(i, total_loss/n_batches))

    sess.run(test_init)
    total_correct = 0
    try:
        while True:
            acc_batch = sess.run(accuracy)
            total_correct += acc_batch
    except tf.errors.OutOfRangeError:
        pass
    print("Accuracy: {}".format(total_correct/10000))

   #  w_out, b_out = sess.run([w, b])
    # acc_out = sess.run([accuracy])
writer.close()







