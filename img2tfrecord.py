#-*-coding:utf-8-*-
"""
Created on Thu May  3 08:54:37 2018

@author: zylzlh
"""
import tensorflow as tf
from skimage import transform,io
import numpy as np
import os

clas=[]
image_list=[]
label_list=[]
i=0

for root,dirs,files in os.walk('/home/zyl/train'):
    if dirs!=[]:
        clas=clas+dirs
    if files!=[]:
        for file in files:
            file_path=os.path.join(root,file)
            image_list.append(file_path)
            label_list.append(i)
        i+=1
temp = np.array([image_list, label_list]) #转换成2维矩阵  
temp = temp.transpose() #转置  
np.random.shuffle(temp) #按行随机打乱顺序  
  
#从打乱的temp中再取出list（img和lab）  
image_trainlist = list(temp[:30000, 0])  #取出第0列数据前30000个作为训练集，即图片路径  
label_trainlist = list(temp[:30000, 1]) #取出第1列数据前30000个，即训练集图片标签  
label_trainlist = [int(i) for i in label_trainlist] #转换成int数据类型

image_vallist = list(temp[30000:, 0])  #取出第0列数据后10000个作为验证集，即图片路径   
label_vallist = list(temp[30000:, 1]) #取出第1列数据后10000，即验证集的图片标签  
label_vallist = [int(i) for i in label_vallist] #转换成int数据类型
                   

print(len(image_trainlist),len(label_trainlist),len(image_vallist),len(label_vallist))
print(clas)

def convert_to_tfrecord(images, labels, save_dir, name):    
    '''convert all images and labels to one tfrecord file.  
    Args:  
        images: list of image directories, string type  
        labels: list of labels, int type  
        save_dir: the directory to save tfrecord file, e.g.: '/home/Folder1/' or '/home/Folder2/'
        name: the name of tfrecord file, string type, e.g.: 'train' or 'val' 
    Return:  
        no return  
    Note:  
        converting needs some time, be patient...  
    '''
    bestnum=1000
    filenum=0
    num=0
    n_samples = len(labels)  
    filename = os.path.join(save_dir, name + '.tfrecords-{0:03d}'.format(filenum))    
    writer = tf.python_io.TFRecordWriter(filename)  
    print('\nTransform start......')  
    for i in np.arange(0, n_samples):
        num+=1
        if num>bestnum:
            num=1
            filenum+=1
            filename = os.path.join(save_dir, name + '.tfrecords-{0:03d}'.format(filenum))    
            writer = tf.python_io.TFRecordWriter(filename)
            
        image = io.imread(images[i]) # type(image) must be array!    
        image =transform.resize(image, (227, 227))  
        img = image * 255   
        img = img.astype(np.uint8)     
        image_raw = img.tostring()  
        label = labels[i]
        
        example = tf.train.Example(features=tf.train.Features(feature={  
                        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
                        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}))  
        writer.write(example.SerializeToString())
    
    writer.close()  
if __name__ =='__main__':
    convert_to_tfrecord(image_trainlist,label_trainlist,'/home/zyl/Folder1','train')
    print('trainTf done!')
    convert_to_tfrecord(image_vallist,label_vallist,'/home/zyl/Folder2','val')
    print('valTf done!')
   
    

