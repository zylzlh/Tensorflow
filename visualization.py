# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:44:33 2018

@author: zylzlh
"""

import tensorflow as tf
import numpy as np

x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)+noise-0.5

with tf.variable_scope('Inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x')
    ys=tf.placeholder(tf.float32,[None,1],name='y')

with tf.variable_scope('Net'):
    l1=tf.layers.dense(xs,10,tf.nn.relu,name='hidden_layer')
    prediction=tf.layers.dense(l1,1,name='output_layer')
    #add to histgram summary
    tf.summary.histgram('h_out',l1)
    tf.summary.histgram('pred',prediction)

loss=tf.mean_squared_error(ys,prediction,scope='loss')
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
tf.summary.scalar('loss',loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter(r'C:\log',sess.graph)
    merge_op=tf.summary.merge_all()
    for i in range(1000):
        _,result=sess.run([train_step,merge_op],feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        
