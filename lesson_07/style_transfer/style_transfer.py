import numpy as np
import os
import tensorflow as tf

from vgg19 import VGG

class StyleTransfer():

    def __init__(self, content_img, style_img, img_height, img_width):
        self.content_img = content_img
        self.style_img = style_img
        self.img_height = img_height
        self.img_width = img_width

        #hyperparamters
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0),
                                           trainable=False)

    def input_img(self):
        """
        Initialize the blank image that will be come the final combined image
        """
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('input_img', 
                                            shape = [1, self.img_height, self.img_width, 3],
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

    def load_vgg(self):
        self.vgg = VGG(self.input_img)
        self.vgg.build_graph()
        self.content_img - self.vgg.mean_pixels
        self.style_img - self.vgg.mean_pixels

    def content_loss(self, F, P):
        """
        F is the matrix representing the activations of given layer of 
        VGG. The rows of F are the activations of a filter, and the columns
        represent the position. P represents the original content image.
        """
        coef = 0.5 
        self.content_loss = coef * tf.reduce_sum(tf.square(F - P))

    def style_loss(self, G, A):
        """
        G is the Gram matrix of the original image and A is 
        the Gram matrix of the input image.
        """
        self.style_loss = coef * tf.reduce_sum(tf.square(G - A))

