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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)         
    
    return X_train, y_train,X_test, y_test


def _encode_labels(y, k):
    onehot = np.zeros((y.shape[0], k))
    for idx, val in enumerate(y):
        onehot[idx, val] = 1.0
    return onehot


def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                     
                        strides=[1, 1, 1, 1], padding='VALID'))

    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w2,                      
                        strides=[1, 1, 1, 1], padding='VALID'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],             
                        strides=[1, 1, 1, 1], padding='SAME')


    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                      
                        strides=[1, 1, 1, 1], padding='VALID'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 3, 3, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')


    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4,                       
                        strides=[1, 1, 1, 1], padding='VALID'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],              
                        strides=[1, 1, 1, 1], padding='SAME')

    l5 = tf.reshape(l4, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)
    l6 = tf.nn.relu(tf.matmul(l5, w_fc))
    pyx = tf.matmul(l6, w_o)
    return pyx


trX, trY, teX, teY = loadData()

k = np.amax(trY)+1;  # NUmber of classes
k2 = np.amax(teY)+1; # NUmber of classes




trX = trX.reshape(-1, 62, 47, 1)  
teX = teX.reshape(-1, 62, 47, 1)  
trY = _encode_labels(trY,k)
teY = _encode_labels(teY,k2)


X = tf.placeholder("float", [None, 62, 47, 1])
Y = tf.placeholder("float", [None, k])

w = init_weights([3, 3, 1, 16],"w") 
w2 = init_weights([3, 3, 16, 16],"w2") 
w3 = init_weights([3, 3, 16, 32], "w3")       # 3x3x1 conv, 32 outputs
w4 = init_weights([3, 3, 32, 48],"w4") 
w_fc = init_weights([48 * 26 * 19, 160],"w_fc") # FC 32 * 14 * 14 inputs, 625 outputs
w_o = init_weights([160, k],"w_o")         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables

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



