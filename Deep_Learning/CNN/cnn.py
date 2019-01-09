
'''
A fully connected layer is added after hidden layer (convoluotoional +pooling),it is same as your hidden layer!!
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets('' ,one_hot=True)
#mnist data set is have 60,000 rows where each row is a image of 28x28 pixels and each pixel is either 0 or 1 which is hand written script and output is (see below)
'''
This is output of our data set
now it is a multi classification problem with 10 classes from 0-9
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
etc..
'''
n_classes=10
batch_size=128 

x=tf.placeholder('float',[None,784]) #seems like a long straing tensor with one row or 784pixels wide
y=tf.placeholder('float') #label of data

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#2 and 3rd one mean the convolutional window is moving 1 pixel at a time

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#2,2 mean size of pool window is 2x2 and it i moving 2 steps at a time and hence there is no overlapping      
#I found out what it is. It went through two max pooling process. And each time, it reduces the dimension by half because of its stride of 2 and size of 2. Hence, 28/2/2 = 7.﻿
def convolutional_neural_network(x):
    weights={'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])), #it will be 5X5 convolution,and it will take 1 input and will produce 32 output  
             'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
             'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),#compressing image to 7*7 pixel ie a feature map,we started with 28x28 pixel and we want 1024nodes and 64 he said came from previous one!!!!!! 
             'out':tf.Variable(tf.random_normal([1024,n_classes]))}#those 124 output come in output layer and n_classes implies nodes in output layer

    biases={'b_conv1':tf.Variable(tf.random_normal([32])),   
             'b_conv2':tf.Variable(tf.random_normal([64])),
             'b_fc':tf.Variable(tf.random_normal([1024])), 
             'out':tf.Variable(tf.random_normal([n_classes]))}
    

    x=tf.reshape(x,shape=[-1,28,28,1]) #reshaping our long 784 pixel image back to 28*28
    conv1=conv2d(x,weights['W_conv1'])
    conv1=maxpool2d(conv1) #pooling on the convolutional window and we got ist hidden layer

    conv2=conv2d(conv1,weights['W_conv2']) #input to 2nd layer
    conv2=maxpool2d(conv2)#we got 2nd hiden layer

    fc=tf.reshape(conv2,[-1,7*7*64]) # a long vector  (to be honest didn't got this 7*7*64 thing!!!!!!!!!)
    fc=tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

    output=tf.matmul(fc,weights['out'])+biases['out']
    return output #it is that one_hot array!! #computation graph completed for our tensorflow model (neural neural_network_model)




def train_neural_network(x):
    prediction=convolutional_neural_network(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) 
    optimizer=tf.train.AdamOptimizer().minimize(cost) 

    hm_epochs=10 

    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):       #now our data will be processed in batch sizes
                #print(mnist.train.num_examples)
                epoch_x,epoch_y=mnist.train.next_batch(batch_size) 
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y}) 
                epoch_loss +=c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss) #

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))  #checking index pf 1 for our prediction and actual label as both are of the form 0000001000
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        
        print("accuracy",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))




train_neural_network(x)