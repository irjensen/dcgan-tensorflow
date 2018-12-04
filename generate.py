import tensorflow as tf
import numpy as np
from model import *
import os
from PIL import Image
from glob import glob
from argparse import ArgumentParser

BETA1 = 0.5
IMAGE_NAME = 'image'

def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint', help='path to checkpoint',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--output-dir', type=str,
                        dest='output_dir', help='dir to place generated images',
                        metavar='OUTPUT_DIR', required=True)
    
    parser.add_argument('--num-images', type=int,
                        dest='num_images', help='number of images to output',
                        metavar='NUM_IMAGES', default=1)
    
    parser.add_argument('--seed', type=int,
                        dest='seed', help='seed for random generation of z',
                        metavar='SEED', default=None)
    
    parser.add_argument('--name', type=str,
                        dest='name', help='name for the generated image(s)',
                        metavar='NAME', default=IMAGE_NAME)

    parser.add_argument('--grid-size', type=int,
                        dest='grid_size', help='generate a square grid of images',
                        metavar='GRID_SIZE', default=1)
    return parser


def generate_z(batch_size, z_dim, seed=None):
    np.random.seed(seed)
    #return np.random.uniform(-1.0, 1.0, size=(batch_size, z_dim))
    return np.ones([batch_size, z_dim])*-1.0

    z_dim = input_z.get_shape()[-1]
    sample_z = generate_z(25, z_dim)
    
    sample_images = sess.run(
        generator(input_z, data_shape[1:], is_train=False), 
        feed_dict = {input_z: sample_z})
    
    square_images = sample_images.reshape([5, 5, data_shape[1], data_shape[2], data_shape[3]])
    square_images = square_images.swapaxes(1, 2)
    square_images = square_images.reshape([5*data_shape[1], 5*data_shape[2], data_shape[3]])
    #square_images = 255. * (square_images + 1.) / 2.
    return Image.fromarray(square_images, "RGB")


def get_sample(sess, input_z, data_shape, size, seed=None):
    z_dim = input_z.get_shape()[-1]
    sample_z = generate_z(size*size, z_dim)
    
    sample_images = sess.run(
        generator(input_z, data_shape[1:], is_train=False), 
        feed_dict = {input_z: sample_z})
    
    square_images = sample_images.reshape([size, size, data_shape[1], data_shape[2], data_shape[3]])
    square_images = square_images.swapaxes(1, 2)
    square_images = square_images.reshape([size*data_shape[1], size*data_shape[2], data_shape[3]])
    square_images = (((square_images - square_images.min()) * 255) / (square_images.max() - square_images.min())).astype(np.uint8)
    return Image.fromarray(square_images, 'RGB')
     
    
def main():
    parser = build_parser()
    opt = parser.parse_args()
    
    data_shape = None
    z_dim = None
    with tf.Graph().as_default():
        saver0 = tf.train.import_meta_graph(os.path.join(opt.checkpoint, '.meta'))
        with tf.Session() as sess:
            saver0.restore(sess, opt.checkpoint)
            input_real = tf.get_collection('vars')[0]
            input_z = tf.get_collection('vars')[1]
            
            data_shape = input_real.get_shape().as_list()
            z_dim = input_z.get_shape().as_list()[-1]
        
    tf.reset_default_graph()
    with tf.Graph().as_default():
        input_real, input_z, lr = model_inputs(data_shape[1], 
                                               data_shape[2], 
                                               data_shape[3],
                                               z_dim)
        d_loss, g_loss = model_loss(input_real, input_z, data_shape[1:])
        d_opt, g_opt = model_opt(d_loss, g_loss, lr, BETA1)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver1 = tf.train.Saver(allow_empty=True)
            saver1.restore(sess, opt.checkpoint)
            
            os.makedirs(os.path.dirname(opt.output_dir), exist_ok=True)
            for i in range(opt.num_images):
                image = get_sample(sess, input_z, data_shape, opt.grid_size, opt.seed)
                if opt.num_images == 1:
                    image.save(os.path.join(opt.output_dir, '{}.png'.format(opt.name)))
                else:
                    image.save(os.path.join(opt.output_dir, '{}_{}.png'.format(opt.name, i)))


                    
                  
    

if __name__ == '__main__':
    main()