"""
Handwriting_Recognition
ryou73
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#변수 선언
x=tf.placeholder(tf.float32,[None,784])
result_default=tf.placeholder(tf.float32,[None,10])
W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.random_normal([10]))
learning_rate=0.01

#결과값
y=tf.matmul(x,W)+b
result=tf.nn.softmax(y)

#비용함수
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=result_default))

#경사하강법을 이용한 학습
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cost)

#세션 설정
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#정확도함수
comp1=tf.argmax(result,1)
comp2=tf.argmax(result_default,1)
eccu_list=tf.equal(comp1,comp2)
eccu=tf.reduce_mean(tf.cast(eccu_list,tf.float32))

#텐서보드를 위한 전처리
eccuracy_hist=tf.summary.scalar('eccuracy_hist',eccu)
cost_hist=tf.summary.scalar('cost_hist',cost)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./board/Handwriting_Recognition/1layer',sess.graph)


#학습
for i in range(10000):
    train_data,train_result=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:train_data,result_default:train_result})
    
    #텐서보드 로그기록
    if i%10==0:
        test_data,test_result=mnist.test.next_batch(5000)
        summary=sess.run(merged,feed_dict={x:test_data,result_default:test_result})
        writer.add_summary(summary,i)
        writer.flush()
