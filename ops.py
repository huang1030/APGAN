from __future__ import print_function
import random
import os
import tensorflow as tf
import numpy as np
import glob

def flatten(x) :
    return tf.layers.flatten(x)
    

def save(saver, sess, logdir, step): #保存模型的save函数
    model_name = 'model' #保存的模型名前缀
    checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
    if not os.path.exists(logdir): #如果路径不存在即创建
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step) #保存模型
    print('The checkpoint has been created.')
 
    
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像
 

def get_write_picture(x_image, fake_y, y_image): #get_write_picture函数得到训练过程中的可视化结果
    x_image = cv_inv_proc(x_image) #还原x域的图像
    y_image = cv_inv_proc(y_image) #还原y域的图像
    fake_y = cv_inv_proc(fake_y[0])
    output = np.concatenate((x_image, fake_y, y_image), axis=1)
    return output
 
    
def make_train_data_list(x_data_path, y_data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    x_input_images_raw = glob.glob(os.path.join(x_data_path, "*")) #读取全部的x域图像路径名称列表
    y_input_images_raw = glob.glob(os.path.join(y_data_path, "*")) #读取全部的y域图像路径名称列表
    return x_input_images_raw, y_input_images_raw
    
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
 
    
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src-dst)**2)
 
        
def make_test_data_list(x_data_path): #make_test_data_list函数得到测试中的x域和y域的图像路径名称列表
    x_input_images = glob.glob(os.path.join(x_data_path, "*")) #读取全部的x域图像路径名称列表
    return x_input_images
 
    
def get_picture(x_image, fake_y): 
    x_image = cv_inv_proc(x_image) 
    fake_y = cv_inv_proc(fake_y[0])
    output = np.concatenate((x_image, fake_y), axis=1)
    return output


def var_mean_close(input1, input2, epsilon=1e-5):
    axes = [2,3] 
    c_mean, c_var = tf.nn.moments(input1, axes=axes, keep_dims=True)
    s_mean, s_var = tf.nn.moments(input2, axes=axes, keep_dims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)
    return  l1_loss(c_mean, s_mean) + l1_loss(c_std, s_std)

