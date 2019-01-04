import tensorflow as tf
import numpy as np

data_file = 'birth_life_2010.txt'

def read_birth_life_file(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

data, n_samples = read_birth_life_file(data_file)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

prediction = w * X + b
loss = tf.square(Y - prediction, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #run 100 epochs
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X:x, Y:y})

    w_out, b_out = sess.run([w, b])
writer.close()

#def huber loss using tensorflow
def huber_loss(label, pred, delta=14.0):
    residual = tf.abs(label - pred)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5*tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

"""
There is something called tf.data that is a better way to load the data
that is going to be used for training (better than tf.placeholder).

This works by storing the data in a tensorflow object rather than 
an array like we are doing above. tf.data.Dataset.from_tensor_slices
will turn our array into the appropriate tf.data object.


"""



