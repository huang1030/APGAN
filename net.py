import tensorflow as tf
import tensorflow.contrib as tf_contrib
from initialization import parse_args

args = parse_args()
    
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)
 

def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 

def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output
 

def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 

def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)
 

def residule_block(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = True, name = "res"):
    if not atrous:
        conv2dc0 = atrous_conv2d(input_, output_dim, kernel_size, dilation, name = (name + '_c0'))
        conv2dc0_relu = relu(input_ = conv2dc0)
        conv2dc1 = atrous_conv2d(conv2dc0_relu, output_dim, kernel_size, dilation, name = (name + '_c1'))
    else:
        conv2dc0 = conv2d(input_, output_dim, kernel_size, stride, name = (name + '_c0'))
        conv2dc0_relu = relu(input_ = conv2dc0)
        conv2dc1 = conv2d(conv2dc0_relu, output_dim, kernel_size, stride, name = (name + '_c1'))

    add_raw = input_ + conv2dc1
    output = relu(add_raw)
    return output
 
def flatten(x) :
    return tf.layers.flatten(x)
  
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output   


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

 
def global_max_pooling(x):
    gmp = tf.reduce_max(x, axis=[1, 2])
    return gmp


def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('InstanceNorm'):
        x = x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
    return x   

    
def generator(image, gf_dim=64, reuse=False, name="generator"): 
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        x = relu(batch_norm(conv2d(image, gf_dim, 7, 1, name = 'g_e0'), name='batch_norm_1'))
        x = relu(batch_norm(conv2d(x, gf_dim * 2, 3, 2, name = 'g_e1'), name='batch_norm_2'))
        
        x_ = tf.image.resize_images(image, [128, 256])      
        x_ = relu(batch_norm(conv2d(x_, gf_dim * 2, 3, 2, name = 'g_e1_'), name='batch_norm_3'))
        
        for i in range(9):
            name = 'res_stage'+ str(i)
            x_ = residule_block(x_, gf_dim*2, atrous = False, name=name)
            
        x_ = relu(batch_norm(deconv2d(x_, gf_dim * 2, 3, 2, name = 'g_d1_dc'),name='batch_norm_4')) 
         
        x = tf.concat([x, x_], axis=-1)

        for i in range(9):
            name = 'res_con'+ str(i)
            x = residule_block(x, gf_dim*4, atrous = False, name=name)

        x = relu(batch_norm(deconv2d(x, gf_dim * 2, 3, 2, name = 'g_d2_dc'),name='batch_norm_5'))        
        d3 = conv2d(x, 3, 7, 1, name = 'g_e')
        output = tf.nn.tanh(d3)
        
        return output
        

def discriminator(image, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        x = lrelu(conv2d(image, df_dim, 4, 2, name='d_h0_conv'))
        x = lrelu(batch_norm(conv2d(x, df_dim*2, 4, 2, name='d_h1_conv'), 'd_bn1'))
        x = lrelu(batch_norm(conv2d(x, df_dim*4, 4, 2, name='d_h2_conv'), 'd_bn2'))
        x = lrelu(batch_norm(conv2d(x, df_dim*8, 4, 1, name='d_h3_conv'), 'd_bn3'))
        output = conv2d(x, 1, 4, 1, name='d_h4_conv')
        
        return output
