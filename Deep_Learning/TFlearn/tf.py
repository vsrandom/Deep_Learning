'''
TFLearn- High level abstraction layer in library on top of tensor flow
eg keres tflearn etc..
'''
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


X,Y,test_x,test_y=mnist.load_data(one_hot=True)

X=X.reshape([-1,28,28,1]) #-1 implies it will automatically decide / only one row(maybe) , and size of vector is 28*28 straignht and it has only one channel
test_x=test_x.reshape([-1,28,28,1])

convnet=input_data(shape=[None,28,28,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu') #32 size and 2 is windows size or maybe movemet? Activation function= rectify linear
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8) #What the heck in this world this dropout do? IDK i guess it was something related to firing of neurons

#Now output layer is also a fully connected layer
convnet=fully_connected(convnet,10,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy',name='targets') #calculating loss and from inputs and loss ,how close are we to our targets
model=tflearn.DNN(convnet)

#training the CNN model !
model.fit({'input':X},{'targets':Y},n_epoch=2,
		validation_set=({'input':X},{'targets':Y}),
		snapshot_step=500,show_metric=True,run_id='mnist')


model.save('tflearncnn.model') #it only save our weights ie line from 32-34
