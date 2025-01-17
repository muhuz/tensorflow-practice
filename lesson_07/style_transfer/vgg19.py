import numpy as np
import os
import scipy.io as sio
import tensorflow as tf

file_path = 'vgg_files'
file_name = 'imagenet-vgg-verydeep-19.mat'

# vgg_weights =  sio.loadmat(os.path.join(file_path, file_name))
# vgg_layers = vgg_weights['layers']

class VGG():
    def __init__(self, input_img):
        self.input = input_img
        self.vgg_layers = sio.loadmat(os.path.join('vgg_files', 'imagenet-vgg-verydeep-19.mat'))['layers']
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    def load_weights(self, layer_idx, expected_layer_name):
        """
        Returns the weights and bias of the layer indexed by layer_idx in VGG.
        """
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b.reshape(b.size)

    def conv2d_relu(self, inputs, layer_idx, scope_name, stride=1, padding='SAME'):
        """
        Returns the conv2d + relu calculation to the graph.
        """
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            W, b = self.load_weights(layer_idx, scope_name)
            conv = tf.nn.conv2d(inputs, W, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + b, name=scope.name)

    def avgpool(self, inputs, ksize=2, strides=2, padding='SAME', scope_name='avgpool'):
        """
        Implementation of avgpool using numpy.
        """
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputs, ksize=[1, 2, 2,1],
                                    strides=[1, 2, 2, 1],
                                    padding=padding)

    # def avgpool(inputs, ksize, strides, padding='VALID', scope_name='avgpool'):
        # """
        # Implementation of avgpool using tensorflow pool.
        # """
        # with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            # pool = tf.nn.avg_pool(inputs, ksize, strides, padding)
        # return pool

    def build_graph(self):
        """
        Builds the full VGG19 computation graph
        """
        self.conv1_1 = self.conv2d_relu(self.input, 0, 'conv1_1')
        self.conv1_2 = self.conv2d_relu(self.conv1_1, 2, 'conv1_2')
        self.avgpool1 = self.avgpool(self.conv1_2, scope_name='avgpool1')
        self.conv2_1 = self.conv2d_relu(self.avgpool1, 5, 'conv2_1')
        self.conv2_2 = self.conv2d_relu(self.conv2_1, 7, 'conv2_2')
        self.avgpool2 = self.avgpool(self.conv2_2, scope_name='avgpool2')
        self.conv3_1 = self.conv2d_relu(self.avgpool2, 10, 'conv3_1')
        self.conv3_2 = self.conv2d_relu(self.conv3_1, 12, 'conv3_2')
        self.conv3_3 = self.conv2d_relu(self.conv3_2, 14, 'conv3_3')
        self.conv3_4 = self.conv2d_relu(self.conv3_3, 16, 'conv3_4')
        self.avgpool3 = self.avgpool(self.conv3_4, scope_name='avgpool3')
        self.conv4_1 = self.conv2d_relu(self.avgpool3, 19, 'conv4_1')
        self.conv4_2 = self.conv2d_relu(self.conv4_1, 21, 'conv4_2')
        self.conv4_3 = self.conv2d_relu(self.conv4_2, 23, 'conv4_3')
        self.conv4_4 = self.conv2d_relu(self.conv4_3, 25, 'conv4_4')
        self.avgpool4 = self.avgpool(self.conv4_4, scope_name='avgpool4')
        self.conv5_1 = self.conv2d_relu(self.avgpool4, 28, 'conv5_1')
        self.conv5_2 = self.conv2d_relu(self.conv5_1, 30, 'conv5_2')
        self.conv5_3 = self.conv2d_relu(self.conv5_2, 32, 'conv5_3')
        self.conv5_4 = self.conv2d_relu(self.conv5_3, 34, 'conv5_4')
        self.avgpool5 = self.avgpool(self.conv5_4, scope_name='avgpool5')

