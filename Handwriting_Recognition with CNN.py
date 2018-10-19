"""
Handwriting_Recognition
ryou73
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#변수 선언
x=tf.placeholder(tf.float32,[None,28,28,1])
result_default=tf.placeholder(tf.float32,[None,10])

W1=tf.Variable(tf.random_normal([3,3,1,32])) #3.3.1필터 32개
W2=tf.Variable(tf.random_normal([3,3,32,64])) #3.3.32필터 64개
W3=tf.Variable(tf.random_normal([64*7*7,512])) #Full Connected Layer
W4=tf.Variable(tf.random_normal([512,10])) # Softmax

learning_rate=0.01

#결과값
L1=tf.nn.relu(tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')) #?,28,28,32
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #?,14,14,32

L2=tf.nn.relu(tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')) #?,14,14,64
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #?,7,7,64
L2=tf.reshape(L2,[-1,7*7*64]) #?,7*7*64

L3=tf.nn.relu(tf.matmul(L2,W3)) #?,512

y=tf.matmul(L3,W4) #?,10
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
writer=tf.summary.FileWriter('./board/Handwriting_Recognition/CNN',sess.graph)


#학습
for i in range(1000):
    train_data,train_result=mnist.train.next_batch(100)
    train_data=train_data.reshape(-1,28,28,1)
    
    sess.run(train_step,feed_dict={x:train_data,result_default:train_result})
    
    #텐서보드 로그기록
    if i%100==0:
        test_data,test_result=mnist.test.next_batch(5000)
        test_data=test_data.reshape(-1,28,28,1)
        
        summary=sess.run(merged,feed_dict={x:test_data,result_default:test_result})
        writer.add_summary(summary,i)
        writer.flush()
        
