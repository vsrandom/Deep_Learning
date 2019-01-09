'''
What we will be doing :
Feed forward =input>weight>hiddenlayer1(activation function)>weights>hiddenlayer2(activation function)>weights>output layer
then we will compare output to intended output by using some cost function
then we will have a an optimization function(optimizer) to minimize the costfunction (eg of optimization function AdamOptimizer...SGD(dradient descent,AdaGrad))
Stochastic Gradient Descent !
then we will do backpropogation

feed forward+backpropogation=epoch
1 epoch is like a cycle , and we will do it maybe 10 20 0r 30 times untill cost function is minimized
so like after ist epoch cost will be high,then it might get lower down after furthur cycles etc..
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/" ,one_hot=True)
#mnist data set is have 60,000 rows where each row is a image of 28x28 pixels and each pixel is either 0 or 1 which is hand written script and output is (see below)
'''
This is output of our data set
now it is a multi classification problem with 10 classes from 0-9
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
etc..
'''
n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10
batch_size=100 #something related to that data will be processed in batch sizes!!

x=tf.placeholder('float',[None,784]) #a long straing tensor with one row or 784 pixels wide
y=tf.placeholder('float') #label of data

def neural_network_model(data):
    #(input*weights)+biases - biases in case all features are zero so we want some neuorns to fire !
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}#weights will be a tensorflow variable ie a giants tensor of size(784,n_nodes_hl1) with random numbers!! because we have to start with something so random
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    #now our model is  #(input*weights)+biases
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases']) #that summation box
    l1=tf.nn.relu(l1) #threshold/Activation function rectified linear to see weather a neuron will fire or not!

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases']) #that summation box
    l2=tf.nn.relu(l1)

    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases']) #that summation box
    l3=tf.nn.relu(l1)

    output=tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output #it is that one_hot array!! #computation graph completed for our tensorflow model (neural neural_network_model)

#till now we have defined our computation graph
#now what left is ,we have to specify how we want to run that data through our model in the session and what we want to do that model in session


def train_neural_network(x):
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #we are using cross_entropy_with_logits as out cost function which is calculating the differnce between prediction and the label as cost
    optimizer=tf.train.AdamOptimizer().minimize(cost) #and we are using AdamOptimizer as our optimizer function which will minimize the cost
    #an epoch is a feed forward + backprop cycle
    hm_epochs=10 #we want 10 cycles

    with tf.Session() as sess:                  #now our session is running
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):       #now our data will be processed in batch sizes
                #print(mnist.train.num_examples)
                epoch_x,epoch_y=mnist.train.next_batch(batch_size) #now 60000 rows are dividied in batch sizes and each time it is assigning the batch data to x and y
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
        print("accuracy",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))




train_neural_network(x)
