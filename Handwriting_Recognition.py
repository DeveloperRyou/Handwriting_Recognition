"""
Handwriting_Recognition
ryou73
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x=tf.placeholder(tf.float32,[None,784])
result_default=tf.placeholder(tf.float32,[None,10])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(x,W)+b
learning_rate=0.5

result=tf.nn.softmax(y)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=result_default))

train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    print(0)
    train_data,result_data=mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:train_data,result_default:result_data})
