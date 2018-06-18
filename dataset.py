# -*- coding: utf-8 -*-
"""
Created on Mon May  7 03:06:30 2018

@author: zylzlh
"""
import tensorflow as tf

num_epochs = 10
batch_size = 256
shuffle_buffer=2000
#解析一个TFrecord的方法。record是从文件中读取的一个样例。
def parser1(record):
    #解析读入的一个样例（来自训练集或验证集）。
    features = tf.parse_single_example(record,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'image_raw' : tf.FixedLenFeature([], tf.string)
                                   })  #取出包含image和label的feature对象
    #从原始图像数据解析出像素矩阵。
    decode_img=tf.decode_raw(features['image_raw'],tf.uint8)
    decode_img=tf.reshape(decode_img,[227,227,1])
    decode_img=tf.subtract(tf.div(tf.cast(decode_img,tf.float32),255.),0.5)*2
    lable=features['label']
    return decode_img,lable

train_files=tf.train.match_filenames_once('/home/zyl/Folder1/train.tfrecords-*')
train_files=tf.train.match_filenames_once('/home/zyl/Folder1/train.tfrecords-*')

#从TFrecord文件创建训练数据集.
dataset=tf.contrib.data.TFRecordDataset(train_files)

#利用map()函数来调用parser1()对训练数据集中的每一条数据进行解析。
dataset=dataset.map(parser1)

dataset=dataset.shuffle(shuffle_buffer).batch(batch_size)
dataset=dataset.repeat(num_epochs)

iterator=dataset.make_initializable_iterator()
img_batch,label_batch=iterator.get_next()

with tf.Session() as sess:
    # Initialize
    sess.run([tf.global_variables_initializer(),
             tf.local_variables_initializer()])
      
    sess.run(iterator.initializer)

    #遍历所有数据，结束时抛出OutofRangeError错误。
    while True:
        try:
            sess.run([img_batch,label_batch])
        except tf.errors.OutOfRangeError:
            break
