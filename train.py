import tensorflow as tf
import numpy as np
from model import *
import os
from PIL import Image
from glob import glob
from argparse import ArgumentParser

LEARNING_RATE = 0.0002
NUM_EPOCHS = 5
BATCH_SIZE = 64
Z_DIM = 100
BETA1 = 0.5

def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', default="./ckpt")
    
    parser.add_argument('--data-dir', type=str,
                        dest='data_dir', help='dir containing training images',
                        metavar='DATA_DIR', default="./data")

    parser.add_argument('--sample-dir', type=str,
                        dest='sample_dir', help='dir to place sample images while training',
                        metavar='SAMPLE_DIR', default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)
    
    parser.add_argument('--rescale-width', type=int,
                        dest='rescale_width', help='set to rescale data image width before training',
                        metavar='RESCALE_WIDTH', default=False)
    
    parser.add_argument('--rescale-height', type=int,
                        dest='rescale_height', help='set to rescale data image height before training',
                        metavar='RESCALE_HEIGHT', default=False)

    parser.add_argument('--crop-width', type=int,
                        dest='crop_width', help='set to crop data image width before training (Always applied before rescale)',
                        metavar='CROP_WIDTH', default=False)
    
    parser.add_argument('--crop-height', type=int,
                        dest='crop_height', help='set to rescale data image width before training',
                        metavar='CROP_HEIGHT', default=False)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    
    parser.add_argument('--z-dim', type=int,
                        dest='z_dim', help='generator z-dimension',
                        metavar='Z_DIM', default=Z_DIM)
    
    parser.add_argument('--constant-z', help='use the same z for all training samples',
                        dest='constant_z', action="store_true")
    return parser
    

def get_sample(sess, input_z, data_shape, sample_z=None):
    z_dim = input_z.get_shape()[-1]
    if sample_z is None:
        sample_z = generate_z(25, z_dim)
    
    sample_images = sess.run(
        generator(input_z, data_shape[1:], is_train=False), 
        feed_dict = {input_z: sample_z})
    
    square_images = sample_images.reshape([5, 5, data_shape[1], data_shape[2], data_shape[3]])
    square_images = square_images.swapaxes(1, 2)
    square_images = square_images.reshape([5*data_shape[1], 5*data_shape[2], data_shape[3]])
    square_images = (((square_images - square_images.min()) * 255) / (square_images.max() - square_images.min())).astype(np.uint8)
    return Image.fromarray(square_images, 'RGB')

    

def train(epoch_count, batch_size, z_dim, learning_rate, get_batches, data_shape, sample_dir, ckpt_dir, constant_z):
    if constant_z:
        sample_z = generate_z(25, z_dim)
    else:
        sample_z = None
    
    input_real, input_z, lr = model_inputs(data_shape[1], 
                                           data_shape[2], 
                                           data_shape[3],
                                           z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[1:])
    d_opt, g_opt = model_opt(d_loss, g_loss, lr, BETA1)
    
    d_loss_data = []
    g_loss_data = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("total trainable parameters:", 
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        for epoch_i in range(epoch_count):
            batch_i = 0
            for batch_images in get_batches(batch_size, 'RGB'): 
                batch_z = generate_z(batch_size, z_dim) 
                
                # run optimizers (generator runs twice)
                _ = sess.run(d_opt, feed_dict={input_real:batch_images, input_z:batch_z, lr:learning_rate})
                _ = sess.run(g_opt, feed_dict={input_z:batch_z, input_real:batch_images, lr:learning_rate})
                _ = sess.run(g_opt, feed_dict={input_z:batch_z, input_real:batch_images, lr:learning_rate})
                
                train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                train_loss_g = g_loss.eval({input_z: batch_z})
                d_loss_data.append(train_loss_d)
                g_loss_data.append(train_loss_g)
                
                # display losses and generated output every 100 batches
                if batch_i % 100 == 0:
                    if sample_dir:
                        samples = get_sample(sess, input_z, data_shape, sample_z)
                        samples.save(os.path.join(sample_dir, 'sample_{:02d}_{:04d}.png'.format(epoch_i+1, batch_i))) 
                    # print training loss
                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Batch {}/{} ...".format(batch_i, data_shape[0]//batch_size),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g),
                          flush=True)  
                    
                    # save a checkpoint
                    tf.add_to_collection("vars", input_real)
                    tf.add_to_collection("vars", input_z)
                    saver = tf.train.Saver()
                    saver.save(sess, ckpt_dir, write_meta_graph=True)
                    #saver.export_meta_graph(os.path.join(ckpt_dir, '.meta'))


                batch_i += 1
                
        # print when finished training
        if sample_dir:
            samples = get_sample(sess, input_z, data_shape, sample_z)
            samples.save(os.path.join(sample_dir, 'sample_{:02d}_{:04d}.png'.format(epoch_count, batch_i)))
        train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
        print("Finished training!",
              "Epoch {}/{}...".format((epoch_i+1)%epoch_count, epoch_count),
              "Batch {} ...".format(batch_i),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
        
        tf.add_to_collection("vars", input_real)
        tf.add_to_collection("vars", input_z)
        saver = tf.train.Saver()
        saver.save(sess, ckpt_dir, write_meta_graph=True)
        
class Dataset():
    def __init__(self, data_files, crop_w=None, crop_h=None, rescale_w=None, rescale_h=None):
        self.data_files = data_files
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.rescale_w = rescale_w
        self.rescale_h = rescale_h
        
        image = self.get_image(data_files[0], 'RGB')
        self.shape = [len(data_files), image.shape[0], image.shape[1], image.shape[2]]
        
    def get_image(self, image_path, mode):
        image = Image.open(image_path)
        x1 = 0
        y1 = 0
        x2 = image.size[0]
        y2 = image.size[1]
        
        if self.crop_w:
            x1 = (image.size[0] - self.crop_w) // 2
            x2 = x1 + self.crop_w
        if self.crop_h:
            y1 = (image.size[1] - self.crop_h) // 2
            y2 = y1 + self.crop_h
        
        if self.crop_w or self.crop_h:
            image = image.crop([x1, y1, x2, y2])
        if self.rescale_w or self.rescale_h:
            image = image.resize([self.rescale_w, self.rescale_h], Image.BILINEAR)
        return 2. * np.array(image.convert(mode)) / 255. - 1.0
    
    def get_batches(self, batch_size, mode):
        indx = 0
        while indx + batch_size <= len(self.data_files):
            batch = np.array([self.get_image(file, mode) for file in self.data_files[indx:indx+batch_size]])
            indx += batch_size
            yield batch
    

def main():
    parser = build_parser()
    opt = parser.parse_args()
    
    print('TensorFlow Version: {}'.format(tf.__version__))
    print("Default GPU device: {}".format(tf.test.gpu_device_name()))
    
    dataset = Dataset(glob(os.path.join(opt.data_dir, '*.jpg')),
                      opt.crop_width,
                      opt.crop_height, 
                      opt.rescale_width,
                      opt.rescale_height)
    
    test_image = dataset.get_image(dataset.data_files[0], 'RGB')
    assert test_image.shape[0] % 16 ==0 and test_image.shape[1] % 16 == 0,\
        "training images must have resolutions divisible by 16"
    
    os.makedirs(os.path.dirname(opt.checkpoint_dir), exist_ok=True)
    if opt.sample_dir:
        os.makedirs(os.path.dirname(opt.sample_dir), exist_ok=True)
                               
    with tf.Graph().as_default():
        train(opt.epochs, opt.batch_size, opt.z_dim, opt.learning_rate, dataset.get_batches,
              dataset.shape, opt.sample_dir, opt.checkpoint_dir, opt.constant_z)
        

if __name__ == '__main__':
    main()

