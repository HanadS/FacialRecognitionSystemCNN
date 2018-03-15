from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


batch_size = 100
test_size = 256

def init_weights(shape,name):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

def loadData():
    faces = fetch_lfw_people(color = False,min_faces_per_person=10)
    X = faces.data
    y = faces.target
    names = faces.target_names
    targets = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)         
    
    return X_train, y_train,X_test, y_test,names,targets



def _encode_labels(y, k):
    onehot = np.zeros((y.shape[0], k))
    for idx, val in enumerate(y):
        onehot[idx, val] = 1.0
    return onehot


def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       
                        strides=[1, 1, 1, 1], padding='SAME'))
    

    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],             
                        strides=[1, 1, 1, 1], padding='SAME')


    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                      
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],             
                        strides=[1, 1, 1, 1], padding='SAME')

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                       
                        strides=[1, 2, 2, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              
                        strides=[1, 1, 1, 1], padding='SAME')


    l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])    
    l5 = tf.nn.relu(tf.matmul(l4, w_fc))
    pyx = tf.matmul(l5, w_o)
    return pyx


trX_raw, trY_raw, teX_raw, teY_raw, names,targets = loadData()

k = np.amax(trY_raw)+1;  # this is the number of classes
k2 = np.amax(teY_raw)+1;



trX = trX_raw.reshape(-1, 62, 47, 1)  # 28x28x1 input img
teX = teX_raw.reshape(-1, 62, 47, 1)  # 28x28x1 input img
trY = _encode_labels(trY_raw,k)
teY = _encode_labels(teY_raw,k2)

X = tf.placeholder("float", [None, 62, 47, 1])
Y = tf.placeholder("float", [None, k])

w = init_weights([5, 5, 1, 16],"w") 
w2 = init_weights([4, 4, 16, 32],"w2") 
w3 = init_weights([3, 3, 32, 48], "w3")       
w_fc = init_weights([48 * 31 * 24, 160], "w_fc")
w_o = init_weights([160, k],"w_o")         

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:

            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))


