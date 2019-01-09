'''
What we will be doing :
Feed forward =input>weight>hiddenlayer1(activation function)>weights>hiddenlayer2(activation function)>weights>output layer
then we will compare output to intended output by using some cost function
then we will have a an optimization function(optimizer) to minimize the costfunction (eg of optimization function AdamOptimizer...SGD(dradient descent,AdaGrad))
then we will do backpropogation

feed forward+backpropogation=epoch
1 epoch is like a cycle , and we will do it maybe 10 20 0r 30 times untill cost function is minimized
so like after ist epoch cost will be high,then it might get lower down after furthur cycles etc..
'''
from senti import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x=tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    #(input*weights)+biases
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}#weights will be a tensorflow variable ie a giants tensor of size(784,n_nodes_hl1) with random numbers!! because we have to start with something so random
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    #now our model is  #(input*weights)+biases
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases']) #that summation box
    l1=tf.nn.relu(l1) #threshold function rel

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases']) #that summation box
    l2=tf.nn.relu(l1)

    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases']) #that summation box
    l3=tf.nn.relu(l1)

    output=tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output #it is that one_hot array!! #computation graph completed for our tensorflow model (neural neural_network_model)


def train_neural_network(x):
    prediction=neural_network_model(x)
    cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) #we are using cross_entropy_with_logits as out cost function which is calculating the prediction and the label
    optimizer=tf.train.AdamOptimizer().minimize(cost) #and we are using AdamOptimizer as our optimizer function which will minimize the cost

    hm_epochs=10 #we want 10 cycles

    with tf.Session() as sess:                  #now our session is running
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss=0
            i=0
            while i<len(train_x):
                start=i
                end=i+batch_size
                batch_x=np.array(train_x[start:end])   #batch_x contain one batch of training data
                batch_y=np.array(train_y[start:end])
                _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y}) #now we are passing each batch for optimization
                epoch_loss +=c #epich loss for first time forward feed and backprop being calculated
                i+=batch_size

            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss) #each time cost is being minimized

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))  #checking index max index ie 1 for our prediction and actual label as both are
        #same or not as this will decide if prediction matched to positive or negative (i hope u got it!!! lol)
        accuracy=tf.reduce_mean(tf.cast(correct,'float')) #whatever the result came converting it into float

        print("accuracy",accuracy.eval({x:test_x,y:test_y}))


train_neural_network(x)
