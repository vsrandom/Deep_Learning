#Tensorflow is an array manipulation libraries, and it is a deep learning library and it is more like numpy and close to sicket Learn
#python is slow compare to c++
#to activate virtual enviornment --> $source ~/tensorflow/venv/bin/activate
#basically we build our model in tensorflow in two steps , first we build the computation graph and then we write what is supposed to be write in session and
#then we run our model
import tensorflow as tf

x1=tf.constant(5)
x2=tf.constant(6)
result=tf.multiply(x1,x2)
print(result)

with tf.Session() as sess:
    output=sess.run(result)
    print(output)
'''
In tensorflow we have a computation graph in which we define how many layers and nodes we are goona have in our network and all and in Session we have the Optimizer with which
we want to run our sessison and all!

'''
#Tensor("Mul:0", shape=(), dtype=int32) it is a abstract tensor and it does not have a value


#sess=tf.Session()
#print(sess.run(result))
