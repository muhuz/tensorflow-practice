import tensorflow as tf
import numpy as np

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

# Writer class that logs all the activity of the graph
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(c)
writer.close()
