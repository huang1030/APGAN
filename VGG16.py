import tensorflow as tf
import numpy as np 

data_dict = np.load('vgg16.npy', encoding='latin1').item()

def conv(x, d_out, name, fineturn=False, xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:

        if fineturn:
            kernel = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
        elif not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),trainable=True, name='bias')
        else:
            kernel = tf.get_variable(scope+'weights', shape=[3, 3, d_in, d_out], dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]), trainable=True, 
                                                name='bias')

        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        return activation


def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name) 
    return activation


def VGG16(images):

    conv1_1 = conv(images, 64, 'conv1_1', fineturn=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', fineturn=True)
    pool1   = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1', fineturn=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', fineturn=True)
    pool2   = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1', fineturn=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', fineturn=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', fineturn=True)
    pool3   = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1', fineturn=True)
    conv4_2 = conv(conv4_1, 512, 'conv4_2', fineturn=True)
    conv4_3 = conv(conv4_2, 512, 'conv4_3', fineturn=True)
    pool4   = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1', fineturn=True)
    conv5_2 = conv(conv5_1, 512, 'conv5_2', fineturn=True)
    conv5_3 = conv(conv5_2, 512, 'conv5_3', fineturn=True)
    pool5   = maxpool(conv5_3, 'pool5')

    return conv1_2, conv2_2, conv3_3, conv4_3
