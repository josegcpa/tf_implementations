#TODO:Finish test and prediction

import os
import sys
import time
import argparse
import pickle
from math import floor
from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from PIL import Image

slim = tf.contrib.slim

_MODE_ERROR = "Mode {0!s} has to be one of ['train','test','predict']."
_NO_IMAGES_ERROR = "No images in {0!s} with the extension {1!s}."

_LOG_TRAIN = 'Step {0:d}: time = {1:f}, loss = {2:f}, '
_LOG_TRAIN += 'KL divergence = {3:f}, reconstruction error = {4:f}'
_LOG_TRAIN_FINAL = 'Finished training. {0:d} steps, '
_LOG_TRAIN_FINAL += 'with {1:d} images in the training set.'
_LOG_TRAIN_FINAL += 'Average time/image = {2:f}s, final loss = {3:f}'
_SUMMARY_TRAIN = 'Step {0:d}: Summary saved in {1!s}'
_CHECKPOINT_TRAIN = 'Step {0:d}: Checkpoint saved in {1!s}'

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

class VAE:
    """
    Variational autoencoder (VAE) class. This class does all the "heavy
    lifting" for the script. For now, only a fixed architecture was
    considered, since it is for now mostly experimental.

    Arguments [default]:
    * mode - whether to use the VAE to train, test or predict ['train']
    #Images
    * image_path - path to the directory containing the images ['.']
    * extension - extension for all images ['png']
    * height - height for the input images
    * width - width for the input images
    #Preprocessing
    * data_augmentation - whether data augmentation should be used [True]
    * convert_hsv - whether the images should be converted to HSV [False]
    #Training
    * learning_rate - learning rate for the training [0.001]
    * beta_l2_regularization - L2 regularization factor for the loss [None]
    * number_of_steps - number of iterations [1000]
    * epochs - number of training epochs (overrides number_of_steps) [None]
    * save_checkpoint_folder - folder where the checkpoints should be
    stored ['/tmp/checkpoint']
    * save_checkpoint_steps - how often should checkpoints be saved [100]
    * save_summary_folder - folder where the summary should be updated
    ['/tmp/summary']
    * save_summary_steps - how often should the summary be updated [100]
    #Testing/Prediction
    * checkpoint_path - path for the checkpoint to be restored [None]
    * save_predictions_path - path where to save the predictions, namely
    for the latent space representation ['./predictions.npy']
    #General parameters
    * config_arguments - arguments for the tf.ConfigProto that will act as
    the config argument for the tf.Session [{}]
    * log_every_n_steps - how often shold training log be produced [5]
    * batch_size - number of images in each mini-batch [32]
    """
    
    def __init__(self,
                 mode = 'train',
                 #Images
                 image_path = '.',
                 extension = '.png',
                 height = 331,
                 width = 331,
                 #Preprocessing
                 data_augmentation = True,
                 convert_hsv = False,
                 #Training
                 learning_rate = 0.001,
                 beta_l2_regularization = None,
                 number_of_steps = 1000,
                 epochs = None,
                 save_checkpoint_folder = '/tmp/checkpoint',
                 save_checkpoint_steps = 100,
                 save_summary_folder = '/tmp/summary',
                 save_summary_steps = 100,
                 #Testing/Prediction
                 checkpoint_path = None,
                 save_predictions_path = './predictions.npy',
                 #General parameters
                 config_arguments = {},
                 log_every_n_steps = 5,
                 batch_size = 32):

        self.mode = mode
        #Images
        self.image_path = image_path
        self.extension = extension
        self.height = height
        self.width = width
        #Preprocessing
        self.data_augmentation = data_augmentation
        self.convert_hsv = convert_hsv
        #Training
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_l2_regularization = beta_l2_regularization
        self.number_of_steps = number_of_steps
        self.epochs = epochs
        self.save_checkpoint_folder = save_checkpoint_folder
        self.save_checkpoint_steps = save_checkpoint_steps
        self.save_summary_folder = save_summary_folder
        self.save_summary_steps = save_summary_steps
        #Testing/Prediction
        self.checkpoint_path = checkpoint_path
        self.save_predictions_path = save_predictions_path
        #General parameters
        self.config_arguments = config_arguments
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size

        #Path-related operations
        self.image_path_list = glob(
            os.path.join(self.image_path, '*' + self.extension)
            )
        self.no_images = len(self.image_path_list)

        #Training step-related operations
        if self.epochs != None:
            self.number_of_steps = floor(self.no_images/self.batch_size) + 1
            self.number_of_steps *= self.number_of_steps * self.epochs
        if self.no_images == 0:
            raise Exception(
                _NO_IMAGES_ERROR.format(self.image_path,self.extension)
                )

        if self.mode == 'train':
            pass
        elif self.mode == 'test' or self.mode == 'predict':
            self.data_augmentation = False
            self.number_of_steps = floor(self.no_images / self.batch_size) + 1
        else:
            raise ValueError(_MODE_ERROR.format(self.mode))

        self.save_ckpt_path = os.path.join(
            self.save_checkpoint_folder, 'my-model.ckpt'
            )

        self.make_dirs()
        print('Filtering images by size...')
        self.filter_size()
        print('Done - {0:d} images.'.format(self.no_images))
        self.image_generator()
        self.prepare_images()
        self.variational_autoencoder()

        if self.mode = 'train':
            self.vae_loss()
            self.pre_train_operations()
            self.train()
        if self.mode = 'test':
            self.
            self.test()
        if self.mode = 'predict':

    def make_dirs(self):
        """
        Function to create directories where checkpoints and the summary will
        be stored/updated.
        """

        def make_dir(dir):
            try: os.makedirs(dir)
            except: pass

        make_dir(self.save_checkpoint_folder)
        make_dir(self.save_summary_folder)

    def filter_size(self):
        """
        Filter all images in the input folder that have dimensions different
        from self.height x self.width.
        """

        tmp = []
        for i,image_path in enumerate(self.image_path_list):
            image = np.array(Image.open(image_path))
            if image.shape[0] == self.height and image.shape[1] == self.width:
                tmp.append(image_path)
            if i % 1000 == 0:
                print('{0:d}/{1:d}'.format(i,self.no_images))
        self.image_path_list = tmp
        self.no_images = len(self.image_path_list)

    def variational_autoencoder(self):
        """
        Function to create the VAE.
        """

        weights_initializer = tf.contrib.layers.variance_scaling_initializer
        tensors = []
        tensors.append(self.inputs)

        if self.beta_l2_regularization != None:
            weights_regularizer = tf.contrib.layers.l2_regularizer(
                self.beta_l2_regularization
            )
        else:
            weights_regularizer = None

        with tf.variable_scope('VAE'):
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer = weights_initializer(),
                activation_fn = tf.nn.relu,
                weights_regularizer = weights_regularizer,
                padding = 'SAME'):

                with tf.variable_scope('Encoder'):

                    with tf.variable_scope('Block1'):
                        network = slim.conv2d(self.inputs,32,[3,3],
                                              scope = 'conv2d_3x3_1')
                        network = slim.conv2d(network,64,[5,5],
                                              scope = 'conv2d_5x5_2')
                        network = slim.conv2d(network,16,[1,1],
                                              scope = 'conv2d_1x1_3')
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                    with tf.variable_scope('Block2'):
                        network = slim.conv2d(network,64,[3,3],
                                              scope = 'conv2d_3x3_1')
                        network = slim.conv2d(network,128,[5,5],
                                              scope = 'conv2d_3x3_2')
                        network = slim.conv2d(network,32,[1,1],
                                              scope = 'conv2d_3x3_3')
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                    with tf.variable_scope('Block3'):
                        network = slim.conv2d(network,128,[3,3],
                                              scope = 'conv2d_3x3_1')
                        network = slim.conv2d(network,256,[5,5],
                                              scope = 'conv2d_3x3_2')
                        network = slim.conv2d(network,64,[1,1],
                                              scope = 'conv2d_3x3_3')
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                with tf.variable_scope('Latent_representation'):
                    tensors.append(network)
                    network = tf.reduce_max(network,axis = [1,2])
                    network = tf.reshape(network,
                                         [-1,64])
                    self.means = slim.fully_connected(
                        network,64,
                        activation_fn = None,
                        scope = 'means',
                        weights_regularizer = weights_regularizer,
                        weights_initializer = weights_initializer())
                    #These are not really the sd values, but log(sd ** 2)
                    self.sds = slim.fully_connected(
                        network,64,
                        activation_fn = None,
                        scope = 'sd',
                        weights_regularizer = weights_regularizer,
                        weights_initializer = weights_initializer())

                    eps = tf.random_normal(tf.shape(self.means),
                                           mean = 0,
                                           stddev = 1,
                                           dtype = tf.float32)
                    self.latent = tf.add(self.means,
                                         tf.sqrt(tf.exp(self.sds)) * eps,
                                         name = 'latent')

                with tf.variable_scope('Decoder'):
                    tensors.append(network)
                    fc_shape = int(
                        np.prod(tensors[4].get_shape().as_list()[1:])
                        )
                    network = slim.fully_connected(
                        self.latent,
                        fc_shape,
                        scope = 'full_decoder',
                        activation_fn = None,
                        weights_regularizer = weights_regularizer,
                        weights_initializer = weights_initializer())
                    network = tf.reshape(network,tf.shape(tensors[3]))

                    with tf.variable_scope('Block1'):
                        network = tf.image.resize_images(
                            network,
                            tf.shape(tensors[2])[1:3]
                            )
                        network = slim.conv2d(network,64,[1,1],
                                              scope = 'conv2d_1x1_1')
                        network = slim.conv2d(network,256,[5,5],
                                              scope = 'conv2d_5x5_2')
                        network = slim.conv2d(network,128,[3,3],
                                              scope = 'conv2d_3x3_3')

                    with tf.variable_scope('Block2'):
                        network = tf.image.resize_images(
                            network,
                            tf.shape(tensors[1])[1:3]
                            )
                        network = slim.conv2d(network,32,[1,1],
                                              scope = 'conv2d_1x1_1')
                        network = slim.conv2d(network,128,[5,5],
                                              scope = 'conv2d_5x5_2')
                        network = slim.conv2d(network,64,[3,3],
                                              scope = 'conv2d_3x3_3')

                    with tf.variable_scope('Block3'):
                        network = tf.image.resize_images(
                            network,
                            tf.shape(tensors[0])[1:3]
                            )
                        network = slim.conv2d(network,16,[1,1],
                                              scope = 'conv2d_1x1_1')
                        network = slim.conv2d(network,64,[5,5],
                                              scope = 'conv2d_5x5_2')
                        network = slim.conv2d(network,32,[3,3],
                                              scope = 'conv2d_3x3_3')
                        self.network = slim.conv2d(
                            network,3,[1,1],
                            scope = 'conv2d_1x1_4',
                            activation_fn = tf.nn.sigmoid)

    def vae_loss(self):
        """
        Implements the loss for the VAE (CrossEntropy + KLDivergence).
        Cross Entropy = -sum(ip + (1 - i) * log(1 - p + e))
        KL Divergence = -0.5 * sum(1 + 2SD - mu ** 2 - e ** 2SD)
        """

        with tf.name_scope('MSE'):
            mse = tf.subtract(self.inputs,self.network)
            mse = tf.square(mse)
            self.mse = tf.reduce_sum(mse,[1,2,3])
            self.mse_batch_mean = tf.reduce_mean(self.mse)

        with tf.name_scope('KL_divergence'):
            kl_div = 1 + 2 * self.sds - self.means ** 2 - tf.exp(2 * self.sds)
            self.kl_div = -0.5 * tf.reduce_sum(kl_div,1)
            self.kl_div_batch_mean = tf.reduce_mean(self.kl_div)

        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(self.mse + self.kl_div)
            self.loss = tf.cast(self.loss,tf.float64)
            if self.beta_l2_regularization != None:
                self.loss = self.loss + tf.add_n(
                    slim.losses.get_regularization_losses()
                    )

    def prepare_images(self):
        """
        Function used to prepare a batch of images for training/testing.
        Everything is implement in tensorflow, so it should be fairly fast.
        Apart from this, it also enables conversion to HSV from RGB.
        """

        def pp_image(image):
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            color_ordering = tf.random_uniform([1],0,4,tf.int32)
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(
                image,
                k = tf.squeeze(tf.random_uniform([1],0,4,tf.int32))
                )

            return image

        def convert_image_float32(image):
            return tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.data_augmentation == True:
            self.inputs = tf.map_fn(
                pp_image,
                self.inputs,
                dtype = tf.float32
            )

        else:
            self.inputs = tf.map_fn(
                convert_image_float32,
                self.inputs,
                dtype = tf.float32
            )

        if self.convert_hsv == True:
            self.inputs = tf.image.rgb_to_hsv(
                self.inputs
            )

    def image_generator(self):
        """
        Creates a tf native input pipeline. Makes everything quite a bit faster
        than using the feed_dict method.
        """

        image_path_tensor = ops.convert_to_tensor(self.image_path_list,
                                                  dtype=dtypes.string)
        train_input_queue = tf.train.slice_input_producer([image_path_tensor],
                                                          shuffle=False)[0]

        file_content = tf.read_file(train_input_queue)
        train_image = tf.image.decode_image(file_content, channels = 3)
        train_image.set_shape([self.height, self.width, 3])
        self.inputs, self.file_names = tf.train.batch(
            [train_image,train_input_queue],
            batch_size=self.batch_size,
            allow_smaller_final_batch=True
            )

    def pre_flight_operations(self):
        """
        Creates the summary operations, sets random seeds and creates the
        global variables initializer.
        """

        self.saver = tf.train.Saver()

        #Setting seeds for randomness
        tf.set_random_seed(1)
        np.random.seed(1)

        #Defining variables to be initialized
        self.init = tf.global_variables_initializer()

        if self.mode == 'train':

            #Optimizer, minimization and variable initiation
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            #Session configuration
            self.config = tf.ConfigProto(**self.config_arguments)

            #Metric
            self.mse,self.mse_op = tf.metrics.mean_squared_error(self.inputs,
                                                                 self.network)

            #Summaries
            self.summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            for variable in slim.get_model_variables():
                self.summaries.add(
                    tf.summary.histogram(variable.op.name, variable)
                    )
            self.summaries.add(tf.summary.scalar('loss', self.loss))
            self.summaries.add(tf.summary.scalar('kl_div_batch_mean',
                                                 self.kl_div_batch_mean))
            self.summaries.add(tf.summary.scalar('mse_batch_mean',
                                                 self.mse_batch_mean))
            self.summary_op = tf.summary.merge(list(self.summaries),
                                               name='summary_op')

        elif self.mode == 'test':

            #Metric
            self.mse,self.mse_op = tf.metrics.mean_squared_error(self.inputs,
                                                                 self.network)

        elif self.mode == 'predict':
            pass

    def train(self):
        """
        Function used to train the VAE.
        """

        print('Run parameters:')
        for var in vars(self):
            print('\t{0}={1}'.format(var,vars(self)[var]))

        print('\n')

        with tf.Session(config = self.config) as self.sess:

            #This allows training to continue
            if self.checkpoint_path != None:
                self.saver.restore(self.sess,checkpoint_path)
            self.writer = tf.summary.FileWriter(self.save_summary_folder,
                                                self.sess.graph)
            self.sess.run(self.init)

            #Sets everything up to run with the queue from image_generator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.time_list = []

            for i in range(1,self.number_of_steps + 1):
                self.sess.run(tf.local_variables_initializer())

                a = time.perf_counter()
                _, _, kl, loss = self.sess.run(
                    [self.train_op,
                    self.mse_op,
                    self.kl_div_batch_mean,
                    self.loss]
                    )
                b = time.perf_counter()
                self.time_list.append(b - a)

                if i % self.log_every_n_steps == 0 or\
                 i % self.number_of_steps == 0 or i == 1:
                    mse = self.sess.run(self.mse)
                    last_time = self.time_list[-1]
                    print(_LOG_TRAIN.format(
                        i,last_time,loss,kl,mse)
                          )

                if i % self.save_summary_steps == 0 or i == 1:
                    summary = self.sess.run(self.summary_op)
                    self.writer.add_summary(summary,i)
                    print(_SUMMARY_TRAIN.format(
                        i,
                        self.save_summary_folder)
                          )

                if i % self.save_checkpoint_steps == 0 or i == 1:
                    self.saver.save(self.sess, self.save_ckpt_path,
                                    global_step = i)
                    print(_CHECKPOINT_TRAIN.format(
                        i,
                        self.save_checkpoint_folder)
                          )

            self.saver.save(self.sess, self.save_ckpt_path,global_step = i)
            print(_CHECKPOINT_TRAIN.format(
                i,
                self.save_checkpoint_folder)
                  )
            self.writer.add_summary(summary,i)
            print(_SUMMARY_TRAIN.format(
                i,
                self.save_summary_folder)
                  )

            loss = self.sess.run(self.loss)
            print(_LOG_TRAIN_FINAL.format(self.number_of_steps,
                                          self.no_images,
                                          np.mean(self.time_list),
                                          loss))
            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def test(self):
        """
        Function used to train the VAE.
        """

        with tf.Session(self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            all_mse = []
            all_kl = []
            time_list = []

            for i in range(self.number_of_steps):
                self.sess.run(tf.local_variables_initializer())
                a = time.perf_counter()
                _, kl_div = self.sess.run([self.mse_op,
                                           self.kl_div])
                b = time.perf_counter()
                self.time_list.append(b - a)

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def predict(self):
        """
        Function used to predict the latent dimension of the VAE. The output is
        stored in a .npy file with information regardin the latent dimension,
        KL divergence and the file name.
        """

        with tf.Session(self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            all_mse = []
            all_kl = []
            time_list = []

            with open(self.save_predictions_path,'ab') as f_handle:
                for i in range(self.number_of_steps):
                    self.sess.run(tf.local_variables_initializer())
                    a = time.perf_counter()
                    latent, kl_div, file_names  = self.sess.run(
                        [self.latent,
                         self.kl_div,
                         self.file_names])

                    b = time.perf_counter()
                    self.time_list.append(b - a)

                    pickle.dump(
                        self.format_output(latent,kl_div,file_names),
                        f_handle
                        )

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def format_output(file_names,kl_div,latent):
        """
        Convenience function to format the output of the self.predict()
        function.
        """

        output = ''

        for file_name,features in zip(file_names,latent):
            features = ','.join(features.astype('str'))
            output += file_name + ',' + features + '\n'

        return(output)

parser = argparse.ArgumentParser(
    prog = 'vae.py',
    description = 'Variational autoencoder.'
)

parser.add_argument('--mode',dest = 'mode',
                    action = 'store',type = str,
                    choices=['train','test','prediction'],
                    default = 'train',
                    help = 'Defines the mode.')
#Images
parser.add_argument('--image_path',dest = 'image_path',
                    action = ToDirectory,type = str,
                    required = True,
                    help = 'Path for the folder containing the images.')
parser.add_argument('--extension',dest = 'extension',
                    action = 'store',type = str,
                    choices=['png','jpg','gif'],
                    default = 'png',
                    help = 'Extension for the image files.')
parser.add_argument('--height',dest = 'height',
                    action = 'store',type = int,
                    default = 256,
                    help = 'Image height.')
parser.add_argument('--width',dest = 'width',
                    action = 'store',type = int,
                    default = 256,
                    help = 'Image width.')
#Preprocessing
parser.add_argument('--data_augmentation',dest = 'data_augmentation',
                    action = 'store_true',
                    default = False,
                    help = 'Flag to set data augmentation.')
parser.add_argument('--convert_hsv',dest = 'convert_hsv',
                    action = 'store_true',
                    default = False,
                    help = 'Flag to convert images from RGB to HSV.')
#Training
parser.add_argument('--learning_rate',dest = 'learning_rate',
                    action = 'store',type = float,
                    default = 0.001,
                    help = 'Learning rate for training.')
parser.add_argument('--epsilon',dest = 'epsilon',
                    action = 'store',type = float,
                    default = 1e-9,
                    help = 'Small constant to add to avoid log(0) errors.')
parser.add_argument('--beta_l2_regularization',dest = 'beta_l2_regularization',
                    action = 'store',type = float,
                    default = None,
                    help = 'Small constant to add to avoid log(0) errors.')
parser.add_argument('--number_of_steps',dest = 'number_of_steps',
                    action = 'store',type = int,
                    default = 1000,
                    help = 'Number of training steps.')
parser.add_argument('--epochs',dest = 'epochs',
                    action = 'store',type = int,
                    default = None,
                    help = 'Number of epochs (overrides number_of_steps).')
parser.add_argument('--save_checkpoint_folder',dest = 'save_checkpoint_folder',
                    action = ToDirectory,type = str,
                    default = '/tmp/checkpoint',
                    help = 'Folder where checkpoints will be stored.')
parser.add_argument('--save_checkpoint_steps',dest = 'save_checkpoint_steps',
                    action = 'store',type = int,
                    default = 100,
                    help = 'Save checkpoint every n steps.')
parser.add_argument('--save_summary_folder',dest = 'save_summary_folder',
                    action = ToDirectory,type = str,
                    default = '/tmp/checkpoint',
                    help = 'Folder where summary will be stored.')
parser.add_argument('--save_summary_steps',dest = 'save_summary_steps',
                    action = 'store',type = int,
                    default = 100,
                    help = 'Save summary every n steps.')
#Testing/prediction
parser.add_argument('--checkpoint_path',dest = 'save_summary_steps',
                    action = ToDirectory,type = str,
                    default = 100,
                    help = 'Path to checkpoint.')
#General parameters
parser.add_argument('--log_every_n_steps',dest = 'log_every_n_steps',
                    action = 'store',type = int,
                    default = 5,
                    help = 'Print log every n steps.')
parser.add_argument('--batch_size',dest = 'batch_size',
                    action = 'store',type = int,
                    default = 32,
                    help = 'Number of images in each mini-batch.')

args = parser.parse_args()

vae = VAE(mode = args.mode,
          #Images
          image_path = args.image_path,
          extension = args.extension,
          height = args.height,
          width = args.width,
          #Preprocessing
          data_augmentation = args.data_augmentation,
          convert_hsv = args.convert_hsv,
          #Training
          learning_rate = args.learning_rate,
          batch_size = args.batch_size,
          epsilon = args.epsilon,
          number_of_steps = args.number_of_steps,
          epochs = args.epochs,
          save_checkpoint_folder = args.save_checkpoint_folder,
          save_checkpoint_steps = args.save_checkpoint_steps,
          save_summary_folder = args.save_summary_folder,
          save_summary_steps = args.save_summary_steps,
          #General parameters
          config_arguments = {},
          log_every_n_steps = args.log_every_n_steps)
