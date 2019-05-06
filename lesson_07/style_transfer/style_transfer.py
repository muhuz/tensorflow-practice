import numpy as np
import os
import tensorflow as tf
import time

from utils import get_resized_image, init_random_image
from vgg19 import VGG

class StyleTransfer():

    def __init__(self, content_img, style_img, img_height, img_width):
        self.content_img = get_resized_image(content_img, img_width, img_height)
        self.style_img = get_resized_image(style_img, img_width, img_height)
        self.initial_img = init_random_image(self.content_img, img_width, img_height) 
        self.img_height = img_height
        self.img_width = img_width

        #hyperparamters
        self.lr = 2.0 
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0),
                                           trainable=False)
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]
        # The alpha and beta hyperparameters that weight the total loss function
        self.content_w = 0.01 
        self.style_w = 1.0

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
        # coef = 1 / 
        self.content_loss = tf.reduce_sum(tf.square(F - P)) / (4.0 * tf.size(P, out_type=tf.float32)) 

    def gram_mat(self, F, N, M):
        """
        Given the activations of a layer, F, reshape the layer into
        a matrix and return the Gram matrix.
        """
        F_mat = tf.reshape(F, (M,N))
        return tf.matmul(F_mat, F_mat, transpose_a=True)

    def single_style_loss(self, a, g):
        """
        g is the feature representation of the original image and a is 
        the feature representation of the input image at a certain layer.
        """
        N = a.shape[3] # number of filters
        M = a.shape[1] * a.shape[2] # height * width of filter
        A =  self.gram_mat(a, N, M)
        G = self.gram_mat(g, N, M)
        return tf.reduce_sum(tf.square(G - A)/ ((2 * N * M)**2))

    def style_loss(self, A):
        """
        A is a layer of the net.
        """
        n_layers = len(A)
        losses = [self.single_style_loss(A[i],
                  getattr(self.vgg, self.style_layers[i])) for i in range(n_layers)]
        self.style_loss = tf.reduce_sum([losses[i] * self.style_layer_w[i] for i in range(n_layers)])

    def loss(self):
        """
        Total Loss is a combination of style loss and content loss.
        """
        with tf.variable_scope('loss') as scope:
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self.content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])
                self.style_loss(style_layers)
            self.loss = self.content_w * self.content_loss + self.style_loss * self.style_loss

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def create_summary(self):
        tf.summary.scalar('content loss', self.content_loss)
        tf.summary.scalar('style loss', self.style_loss)
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self.input_img()
        self.load_vgg()
        self.loss()
        self.optimize()
        self.create_summary()

    def train(self, n_iters):
        skip_step = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize image
            sess.run(self.input_img.assign(self.initial_img))

            # create writer
            writer = tf.summary.FileWriter("graphs")
            writer.add_graph(sess.graph)

            # create saver
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Restored")

            initial_step = self.global_step.eval()
            
            start_time = time.time()
            for index in range(initial_step, n_iters):
                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20
                    
                sess.run(self.opt)
                if (index+1) % skip_step == 0:
                    gen_image, loss, summary = sess.run([self.input_img, self.loss, self.summary_op])
                    print(gen_image)
                    gen_image = gen_image + self.vgg.mean_pixels

                    writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    print('   Loss: {:5.1f}'.format(loss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()

                    if (index + 1) % 20 == 0:
                        saver.save(sess, "checkpoints/style_transfer", index) 

if __name__ == '__main__':
    machine = StyleTransfer('images/nicol_bolas.jpg', 'images/starry_night.jpg', 200, 100)
    machine.build_graph()
    machine.train(300)
                
