import tensorflow as tf
import numpy as np
import math

DROPOUT = 0.5

def discriminator(image, alpha=0.2, reuse=False):

    def conv(inputs, filters):
        return tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=[5, 5],
                              strides=2,
                              padding="same",
                              activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = conv(image, 128)
        h1 = tf.maximum(alpha*h1, h1)

        h2 = conv(h1, 256)
        h2 = tf.layers.batch_normalization(h2, training=True)
        h2 = tf.maximum(alpha*h2, h2)        

        h3 = conv(h2, 512)
        h3 = tf.layers.batch_normalization(h3, training=True)
        h3 = tf.maximum(alpha*h3, h3)

        h4 = tf.contrib.layers.flatten(h3)
        logits = tf.layers.dense(inputs=h4, 
                                 units=1, 
                                 activation=None, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                )
        out = tf.sigmoid(logits)
        
        return out, logits
    
    
def generator(z, out_shape, alpha=0.2, is_train=True):
    
    slice_w = out_shape[0]
    slice_h = out_shape[1]
    for _ in range(4):
        slice_w = math.ceil(slice_w / 2.)
        slice_h = math.ceil(slice_h / 2.)

    def conv_t(inputs, filters):
        return tf.layers.conv2d_transpose(inputs=inputs, 
                                          filters=filters,
                                          kernel_size=[5, 5], 
                                          strides=2, 
                                          padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation=None)

    with tf.variable_scope('generator', reuse=(not is_train)):
        
        h1 = tf.layers.dense(inputs=z, 
                             units=slice_w*slice_h*1024,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=None)
        h1 = tf.reshape(h1, [-1, slice_w, slice_h, 1024])
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.relu(h1)

        h2 = conv_t(h1, 512)
        h2 = tf.layers.dropout(h2, rate=DROPOUT, training=is_train)
        h2 = tf.layers.batch_normalization(h2, training=is_train)
        h2 = tf.nn.relu(h2)
        
        h3 = conv_t(h2, 256)
        h3 = tf.layers.dropout(h3, rate=DROPOUT, training=is_train)
        h3 = tf.layers.batch_normalization(h3, training=is_train)
        h3 = tf.nn.relu(h3)
        
        h4 = conv_t(h3, 128)
        h4 = tf.layers.dropout(h4, rate=DROPOUT, training=is_train)
        h4 = tf.layers.batch_normalization(h4, training=is_train)
        h4 = tf.nn.relu(h4)
        
        logits = conv_t(h4, out_shape[2])
        
        out = tf.tanh(logits)
        return out
    
def model_inputs(image_width, image_height, image_channels, z_dim):

    real_inputs = tf.placeholder(dtype=tf.float32, 
                                 shape = [None, image_width, image_height, image_channels], 
                                 name="real_inputs")
    z_inputs = tf.placeholder(dtype=tf.float32, shape = [None, z_dim], name="z_inputs")
    lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
    
    return real_inputs, z_inputs, lr
    
    
def model_loss(input_real, input_z, out_shape):
    
    fake_images = generator(input_z, out_shape, is_train=True)
    d_model_real, d_logits_real = discriminator(input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(fake_images, reuse=True)
    
    # use label smoothing
    d_labels_real = tf.ones_like(d_model_real)
    d_labels_real += tf.random.uniform(tf.shape(d_labels_real), minval=-0.3, maxval=0)
    
    d_labels_fake = tf.zeros_like(d_model_fake)
    d_labels_fake += tf.random.uniform(tf.shape(d_labels_fake), minval=0, maxval=0.3)
    
    g_labels_fake = tf.ones_like(d_model_fake)
    g_labels_fake += tf.random.uniform(tf.shape(g_labels_fake), minval=-0.3, maxval=0)
    
    # randomly flip 5% of the labels for the discriminator
    mask_real = tf.random.uniform(tf.shape(d_labels_real), minval=0., maxval=20.)
    mask_real = tf.math.floordiv(mask_real, 19) 
    mask_fake = tf.random.uniform(tf.shape(d_labels_fake), minval=0., maxval=20.)
    mask_fake = tf.math.floordiv(mask_fake, 19) 
    d_labels_real -= (2. * (d_labels_real - 0.5)) * mask_real
    d_labels_fake -= (2. * (d_labels_fake - 0.5)) * mask_fake
    
    d_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=d_labels_real))
    d_loss += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                labels=d_labels_fake)) 
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                labels=g_labels_fake))
    
    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def generate_z(batch_size, z_dim):
    x = np.random.normal(size = [batch_size, z_dim])
    return x
