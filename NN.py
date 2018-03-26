import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function==None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)+noise-0.5

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,n_layer=2)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                      reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('tarin'):
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter("c:/logs/",sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            rs=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(rs,i)
