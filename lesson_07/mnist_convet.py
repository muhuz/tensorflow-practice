import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time 

import tensorflow as tf
import tf.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

# Define a function that combines the convolution layer with the non-linearity
def conv_relu(inputs, filters, k_size, stried, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)

# Define the maxpool function
def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=TF.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                            ksize=[1, ksize, ksize, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return pool

# Define full connected layer
def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=TF.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.truncated_normal_initializer())
        out = tf.matmul(inputs, w) + b
    return out

class Convnet:
    def __init__(self, dataset, batch_size, lr, dropout, n_epoch, skip_step):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.n_epoch = n_epoch
        self.skip_step = skip_step
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def inference(self):
        conv1 = conv_relu(inputs=self.img,
                        filters=32,
                        k_size=5,
                        stride=1,
                        padding='SAME'
                        scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=64,
                          k_size=5,
                          stride=1,
                          padding='SAME'
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.nn.relu(fully_connected(pool2, 1024, 'fc'))
        dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training, name='dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')

    def eval(self):
        ''' Count the number of right predictions in a batch '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    
    def create_loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.inference()
        self.eval()
        self.create_loss()
        self.create_optimizer()
        self.create_summaries()

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.global_step.eval()

            num_batches = int(mnist.train.num_examples / self.batch_size)
            total_loss = 0
            for index in range(0, num_batches * self.n_epochs):
                X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
                _, loss_batch, summary = sess.run([self.optimizer, self.loss, self.summary_op],
                        feed_dict={X: X_batch, Y:Y_batch, dropout: self.dropout})
                writer.add_summary(summary, global_step=index)
                total_loss += loss_batch
                if (index+1) % self.skip_step == 0:
                    print('Average Loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                    total_loss = 0
                    saver.save(sess, 'checkpoints/convnet_mnist/mnist-convment', index)
            print('Optimization Finished')

    def test(self):















