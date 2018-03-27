#coding:utf-8

isTrain = False 
train_steps = 1100  
checkpoint_steps = 50
Mini_batch=100

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1)*tf.sqrt(1/in_size),
                        name='Weights')
    biases=tf.Variable(tf.random_normal([1,out_size]),name='biases')
    Wx_plus_b=tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
prediction=layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
saver=tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
<<<<<<< HEAD
    if isTrain:
        sess.run(tf.global_variables_initializer())
=======
    sess.run(tf.global_variables_initializer())
    if isTrain:
>>>>>>> 765e3b52cb734f0098625c8592830ffaea173c44
        max_acc=0
        for i in range(train_steps):
            batch_xs,batch_ys=mnist.train.next_batch(Mini_batch)
            sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
            if i%checkpoint_steps==0:
                accuracy=compute_accuracy(mnist.test.images,mnist.test.labels)
                print(accuracy)
                if accuracy>max_acc:
                    max_acc=accuracy
                    saver.save(sess,r'C:\Users\zylzlh\Desktop\my_net\save_net.ckpt',global_step=i+1)
    #else:
        #model_file=tf.train.latest_checkpoint(r'C:\Users\zylzlh\Desktop\my_net\save_net.ckpt')
        #saver.restore(sess,model_file)
        #print( sess.run(prediction,feed_dict={xs:mnist.train.next_batch(10)}))
        #print(sess.run(Weights))  
        #print(sess.run(biases))
            
            
            
            
