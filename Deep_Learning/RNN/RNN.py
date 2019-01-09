''' In this tutorial we will talk about the main type of algorithms
people are running on deep learning.Most people are
not doing your conventional multi layer perceptron.
People generally use cnn and rnn
Now tranditional neural network does not understand order
(Harrison killed billy or billy killed harison cannot be differentiated
Now we are going to built an LSTM cell ie long short term memory cell,
it is one of the most common cell used by Recurrent neural networks
RNN are used with language data and CNN is use for imaginary data
solve sequential temproral problem
'''

'''
Now what is RNN?
Now basically in NN we feed the data, say a sentence and if we alter
the words ,they still it will be considered same this is not what we
want , hence we have RNN which are used when we want the order of input
data to be important and say a self driving car and like each neuron
in RNN which is called an LSTM as well has this reccurence like
the output of each lstm cell goes into input or as input of other
lstm cells
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell
from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn
mnist=input_data.read_data_sets("/tmp/data/" ,one_hot=True)
#mnist data set is have 60,000 rows where each row is a image of 28x28 pixels and each pixel is either 0 or 1 which is hand written script and output is (see below)
'''
This is output of our data set
now it is a multi classification problem with 10 classes from 0-9
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
etc..
'''

hm_epochs=3
n_classes=10
batch_size=128 #something related to that data will be processed in batch sizes!!

chunk_size=28
n_chunks=28
rnn_size=128


x=tf.placeholder('float',[None,n_chunks,chunk_size]) #a long straing tensor with one row or 784 pixels wide
y=tf.placeholder('float') #label of data

def recurrent_neural_network(x):
    #(input*weights)+biases - biases in case all features are zero so we want some neuorns to fire !
    layer={'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}#weights will be a tensorflow variable ie a giants tensor of size(784,n_nodes_hl1) with random numbers!! because we have to start with something so random

    x=tf.transpose(x,[1,0,2])
    x=tf.reshape(x,[-1,chunk_size])
    x=tf.split(0,n_chunks,x)

    lstm = BasicLSTMCell(rnn_size, state_is_tuple=True,reuse=True)
    (outputs, states) = static_rnn(lstm, x, dtype=tf.float32)﻿﻿

    output=tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output #it is that one_hot array!! #computation graph completed for our tensorflow model (neural neural_network_model)

#till now we have defined our computation graph
#now what left is ,we have to specify how we want to run that data through our model in the session and what we want to do that model in session


def train_neural_network(x):
    prediction=recurrent_neural_network(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #we are using cross_entropy_with_logits as out cost function which is calculating the differnce between prediction and the label as cost
    optimizer=tf.train.AdamOptimizer().minimize(cost) #and we are using AdamOptimizer as our optimizer function which will minimize the cost

    with tf.Session() as sess:                  #now our session is running
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):       #now our data will be processed in batch sizes
                #print(mnist.train.num_examples)
                epoch_x,epoch_y=mnist.train.next_batch(batch_size) #now 60000 rows are dividied in batch sizes and each time it is assigning the batch data to x and y
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y}) #now we are running our optimizer and it is minimizing the cost be changing weights and we are passing our epoch_x and epoch_y
                epoch_loss +=c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss) #each time cost is being minimized

        #like how our accuracy is supposed to be defined!
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))  #checking index pf 1 for our prediction and actual label as both are of the form 0000001000
        accuracy=tf.reduce_mean(tf.cast(correct,'float')) #whatever the result came converting it into float
        '''
        with tf.Session() as sess:
            output=sess.run(accuracy)
            print(output)

        #print(accuracy)
        '''
        print("accuracy",accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)),y:mnist.test.labels}))




train_neural_network(x)
