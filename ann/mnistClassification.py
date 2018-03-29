#coding:utf-8
isTrain = True
train_steps=1000  
checkpoint_steps = 50
batch_size=100

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

with tf.variable_scope('Input'):
    xs=tf.placeholder(tf.float32,[None,784],name='x_input')/255
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.variable_scope('Net'):
    prediction=tf.layers.dense(xs,10,tf.nn.softmax,name='pre_layer')
    
cross_entropy=tf.losses.softmax_cross_entropy(ys,prediction,scope='loss')
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

saver=tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if isTrain:
        max_acc=0
        for i in range(train_steps):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
            if i%checkpoint_steps==0:
                correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
                accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                result=sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels})
                print(result)
                if result>max_acc:
                    max_acc=result
                    saver.save(sess,r'C:\Users\zylzlh\Desktop\my_model',global_step=i+1)
    else:
       # Restore model weights from previously saved model 
        new_saver = tf.train.import_meta_graph(r'C:\Users\zylzlh\Desktop\my_model-501.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(r'C:\Users\zylzlh\Desktop'))
        graph = tf.get_default_graph()
        print(sess.run('Weights:0'))
        print(sess.run('biases:0'))
        print(sess.run(prediction,feed_dict={xs:mnist.test.images}))
        
        
       
            
            
            
