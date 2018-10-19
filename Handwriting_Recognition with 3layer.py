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

W1=tf.Variable(tf.random_normal([784,256]))
W2=tf.Variable(tf.random_normal([256,256]))
W3=tf.Variable(tf.random_normal([256,10]))

b1=tf.Variable(tf.zeros([256]))
b2=tf.Variable(tf.zeros([256]))
b3=tf.Variable(tf.zeros([10]))

learning_rate=0.01

#결과값
L1=tf.nn.relu(tf.matmul(x,W1)+b1)
L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
y=tf.matmul(L2,W3)+b3
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
writer=tf.summary.FileWriter('./board/Handwriting_Recognition/3layer',sess.graph)


#학습
for i in range(50000):
    train_data,train_result=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:train_data,result_default:train_result})
    
    #텐서보드 로그기록
    if i%100==0:
        test_data,test_result=mnist.test.next_batch(5000)
        summary=sess.run(merged,feed_dict={x:test_data,result_default:test_result})
        writer.add_summary(summary,i)
        writer.flush()
