import os
import sys
import time
import argparse
import pickle
from math import floor
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import PIL

import tf_da

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
    * n_latent_layers - the number of autoregressive layers. Implemented as in
    Kingma et al. (2017, https://arxiv.org/pdf/1606.04934.pdf)
    [1]
    """

    def __init__(self,
                 mode = 'train',
                 random_seed = 1,
                 #Images
                 tfrecords=False,
                 image_path = '.',
                 extension = '.png',
                 height = 331,
                 width = 331,
                 resize_height = 256,
                 resize_width = 256,
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
                 #Data augmentation
                 data_augmentation_params={},
                 #Testing/Prediction
                 checkpoint_path = None,
                 save_predictions_path = './predictions.pkl',
                 #General parameters
                 depth_mult = 1,
                 latent_mult = 1,
                 config_arguments = {},
                 log_every_n_steps = 5,
                 batch_size = 32,
                 n_latent_layers = 1,
                 sparsity_amount = 0.1):

        self.mode = mode
        self.random_seed = random_seed
        #Images
        self.tfrecords = tfrecords
        self.image_path = image_path
        self.extension = extension
        self.height = height
        self.width = width
        self.resize_height = resize_height
        self.resize_width = resize_width
        #Preprocessing
        self.data_augmentation = data_augmentation
        self.convert_hsv = convert_hsv
        #Training
        self.learning_rate = learning_rate
        self.beta_l2_regularization = beta_l2_regularization
        self.number_of_steps = number_of_steps
        self.epochs = epochs
        self.save_checkpoint_folder = save_checkpoint_folder
        self.save_checkpoint_steps = save_checkpoint_steps
        self.save_summary_folder = save_summary_folder
        self.save_summary_steps = save_summary_steps
        #Data augmentation
        self.data_augmentation_params = data_augmentation_params
        #Testing/Prediction
        self.checkpoint_path = checkpoint_path
        self.save_predictions_path = save_predictions_path
        #General parameters
        self.depth_mult = depth_mult
        self.latent_mult = latent_mult
        self.config_arguments = config_arguments
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.n_latent_layers = n_latent_layers
        self.sparsity_amount = sparsity_amount
        self.discrete_pixel = False

        #Path-related operations
        if self.extension in self.image_path:
            self.image_path_list = glob(
                self.image_path
                )
            if self.tfrecords:
                s = 0
                for rec in self.image_path_list:
                    s += sum(1 for _ in tf.python_io.tf_record_iterator(rec))
                print(s)
                self.number_of_steps = s // self.batch_size
                if s % self.batch_size > 0:
                    self.number_of_steps += 1
                print(self.number_of_steps)
        else:
            self.image_path_list = glob(
                os.path.join(self.image_path, '*' + self.extension)
                )
        self.no_images = len(self.image_path_list)

        #Training step-related operations
        self.global_step = tf.train.get_or_create_global_step()
        if self.epochs != None:
            self.number_of_steps = floor(self.no_images/self.batch_size) + 1
            self.number_of_steps *= int(self.number_of_steps * self.epochs)
        if self.no_images == 0:
            raise Exception(
                _NO_IMAGES_ERROR.format(self.image_path,self.extension)
                )

        if self.mode == 'train':
            self.is_training = True
        elif self.mode == 'test' or self.mode == 'predict':
            self.is_training = False
            self.data_augmentation = False
            self.epochs = 1
        elif self.mode == 'predict_with_rotation':
            self.is_training = False
            self.data_augmentation = False
            self.number_of_steps = floor(
                self.no_images / self.batch_size + 1) * 4

        self.save_ckpt_path = os.path.join(
            self.save_checkpoint_folder, 'my-model.ckpt'
            )

        self.make_dirs()
        #print('Filtering images by size...')
        #self.filter_size()
        print('Done - {0:d} images.'.format(self.no_images))
        self.image_generator()
        self.prepare_images()
        self.variational_autoencoder()

        if self.mode == 'train':
            self.vae_loss()
            self.pre_flight_operations()
            self.train()
        if self.mode == 'test':
            self.pre_flight_operations()
            self.test()
        if self.mode == 'predict':
            self.vae_loss()
            self.pre_flight_operations()
            self.predict()
        if self.mode == 'predict_with_rotation':
            self.vae_loss()
            self.pre_flight_operations()
            self.predict()

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

    def sess_debugger(self,wtv,times = 1):
        """
        Convenience function used for debugging. It creates a self contained
        session and runs whatever its input is.
        """

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.initialize_local_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            result = sess.run(wtv)
            coord.request_stop()
            coord.join(threads)
        return result

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

        def weight_variable(name,shape):
            initializer = tf.truncated_normal_initializer(mean=0.0,
                                                          stddev=0.01,
                                                          dtype=tf.float32)
            return tf.get_variable(name, shape, initializer = initializer)

        def bias_variable(name,shape):
            initializer = tf.constant_initializer(value=0.0,
                                                  dtype=tf.float32)
            return tf.get_variable(name, shape,initializer=initializer)

        def autoregressive_layer(z,h,dim_z,n_units,scope):
            with tf.variable_scope('autoregressive_' + scope):
                w_m = weight_variable('w_m', [dim_z, dim_z])
                b_m = bias_variable('b_m', [dim_z])
                w_s = weight_variable('w_s', [dim_z, dim_z])
                b_s = bias_variable('b_s', [dim_z])
                w_h = weight_variable('w_h', [n_units, dim_z])

                w_m_masked = mask * w_m
                w_s_masked = mask * w_s

                w_m_h = tf.concat([w_m_masked, w_h], axis = 0)
                w_s_h = tf.concat([w_s_masked, w_h], axis = 0)

                z_h = tf.concat([z, h], axis = 1)

                m = tf.add(tf.matmul(z_h, w_m_h), b_m)
                s = tf.add(tf.matmul(z_h, w_s_h), b_s)
                return m,s

        weights_initializer = tf.contrib.layers.variance_scaling_initializer
        tensors = []
        tensors.append(self.inputs)
        self.convolutions = []

        if self.beta_l2_regularization != None:
            weights_regularizer = tf.contrib.layers.l2_regularizer(
                self.beta_l2_regularization
            )
        else:
            weights_regularizer = None

        with tf.variable_scope('VAE'):
            with slim.arg_scope(
                [slim.conv2d,slim.conv2d_transpose],
                weights_initializer = weights_initializer(),
                activation_fn = None,
                weights_regularizer = weights_regularizer,
                padding = 'SAME',
                normalizer_fn = slim.batch_norm,
                normalizer_params = {'is_training': self.is_training}):

                with tf.variable_scope('Encoder'):
                    with tf.variable_scope('Block1'):
                        network = slim.conv2d(self.inputs,
                                              int(16),
                                              [3,3],
                                              scope = 'conv2d_3x3_1')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)
                        network = slim.conv2d(network,
                                              int(32 * self.depth_mult),
                                              [3,3],
                                              scope = 'conv2d_3x3_2')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                    with tf.variable_scope('Block2'):
                        network = slim.conv2d(network,
                                              int(64 * self.depth_mult),
                                              [3,3],
                                              scope = 'conv2d_3x3_1')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                    with tf.variable_scope('Block3'):
                        network = slim.conv2d(network,
                                              int(128 * self.depth_mult),
                                              [3,3],
                                              scope = 'conv2d_3x3_1')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                        tensors.append(network)

                    with tf.variable_scope('Block4'):
                        depth = int(256 * self.depth_mult)
                        if self.n_latent_layers > 1:
                            depth += 128 * self.depth_mult * self.latent_mult
                            depth = int(depth)
                        network = slim.conv2d(network,
                                              depth,
                                              [3,3],
                                              scope = 'conv2d_3x3_1')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)
                        network = slim.max_pool2d(network,[2,2],
                                              scope = 'max_pool2d_3x3_1')
                    tensors.append(network)

                with tf.variable_scope('Latent_representation'):
                    tensors.append(network)
                    network = tf.reduce_max(network,axis = [1,2])
                    latent_size = int(128 * self.depth_mult * self.latent_mult)
                    network = tf.reshape(
                        network,
                        [-1,depth])
                    self.means = slim.fully_connected(
                        network[:,:latent_size],latent_size,
                        activation_fn = None,
                        scope = 'means',
                        weights_regularizer = weights_regularizer,
                        weights_initializer = weights_initializer())
                    #These are not really the sd values, but log(sd ** 2)
                    self.sds = slim.fully_connected(
                        network[:,latent_size:],latent_size,
                        activation_fn = None,
                        scope = 'sd',
                        weights_regularizer = weights_regularizer,
                        weights_initializer = weights_initializer())
                    if self.n_latent_layers > 1:
                        self.h = slim.fully_connected(
                            network[:,latent_size*2:],latent_size,
                            activation_fn = None,
                            scope = 'h',
                            weights_regularizer = weights_regularizer,
                            weights_initializer = weights_initializer())
                    eps = tf.random_normal(tf.shape(self.means),
                                           mean = 0,
                                           stddev = 1,
                                           dtype = tf.float32)
                    stds = tf.exp(0.5 * self.sds)
                    self.latent = tf.add(self.means,
                                         stds * eps,
                                         name = 'latent')

                    # Define loss function for the latent dimension
                    with tf.name_scope('KL_divergence'):
                        self.kl_div = 1.0 + self.sds
                        self.kl_div -= tf.square(self.means)
                        self.kl_div -= tf.exp(self.sds)
                        self.kl_div = -0.5 * tf.reduce_mean(self.kl_div,axis=1)

                    # Add the IAF layers
                    if self.n_latent_layers > 1:
                        mask = np.zeros([latent_size, latent_size],
                                        dtype=np.float32)
                        mask[np.triu_indices(latent_size)] = 1.0
                        mask = tf.constant(value = mask,
                                           dtype=tf.float32)

                        z = self.latent
                        for l in range(1,self.n_latent_layers):
                            m,s = autoregressive_layer(
                                z=z,h=self.h,dim_z=latent_size,
                                n_units=latent_size,scope=str(l))
                            sigma = tf.nn.sigmoid(s + 1) #+1 = forget gate bias
                            z = sigma * z + (1 - sigma) * m
                            self.kl_div -= tf.reduce_sum(tf.log(sigma + 1e-16))
                        self.latent = z
                    self.kl_div_decay = 1. - tf.train.linear_cosine_decay(
                        learning_rate=2.0,
                        global_step=self.global_step,
                        decay_steps=10000
                    )
                    self.kl_div_decay = tf.maximum(0.,self.kl_div_decay)
                    self.kl_div = self.kl_div * self.kl_div_decay
                    self.kl_div_batch_mean = tf.reduce_mean(self.kl_div)

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
                    network = tf.reshape(network,tf.shape(tensors[4]))

                    with tf.variable_scope('Block1'):
                        network = slim.conv2d_transpose(
                            network,
                            int(256 * self.depth_mult),
                            [3,3],
                            stride=2,
                            scope = 'conv2d_3x3_1')
                        network = slim.conv2d(
                            network,
                            int(128 * self.depth_mult),
                            [3,3],
                            scope = 'conv2d_3x3_2')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)

                    with tf.variable_scope('Block2'):
                        network = slim.conv2d_transpose(
                            network,
                            int(128 * self.depth_mult),
                            [3,3],
                            stride=2,
                            scope = 'conv2d_3x3_1')
                        network = slim.conv2d(
                            network,
                            int(64 * self.depth_mult),
                            [3,3],
                            scope = 'conv2d_3x3_2')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)

                    with tf.variable_scope('Block3'):
                        network = slim.conv2d_transpose(
                            network,
                            int(64 * self.depth_mult),
                            [3,3],
                            stride=2,
                            scope = 'conv2d_3x3_1')
                        network = slim.conv2d(
                            network,
                            int(64 * self.depth_mult),
                            [3,3],
                            scope = 'conv2d_3x3_2')
                        self.convolutions.append(network)
                        network = tf.nn.relu(network)

                    if self.discrete_pixel == True:
                        with tf.variable_scope('Block4'):
                            self.network = slim.conv2d_transpose(
                                network,
                                256,
                                [5,5],
                                activation_fn=None,
                                stride=2,
                                scope = 'conv2d_5x5_1')
                            self.network = slim.conv2d(
                                network,
                                256,
                                [1,1],
                                activation_fn=None,
                                scope = 'conv2d_3x3_discrete_pixel')
                    else:
                        with tf.variable_scope('Block4'):
                            network = slim.conv2d_transpose(
                                network,
                                int(64 * self.depth_mult),
                                [5,5],
                                activation_fn=tf.nn.relu,
                                stride=2,
                                scope = 'conv2d_5x5_1')
                            network = slim.conv2d(
                                network,
                                int(32 * self.depth_mult),
                                [3,3],
                                scope = 'conv2d_3x3_1')
                            self.convolutions.append(network)
                            network = tf.nn.relu(network)
                            network = slim.conv2d(network,
                                                  int(16),
                                                  [3,3],
                                                  scope = 'conv2d_3x3_2')
                            self.convolutions.append(network)
                            network = tf.nn.relu(network)
                            self.network = slim.conv2d(
                                network,3,[1,1],
                                scope = 'conv2d_1x1_4',
                                activation_fn = tf.nn.sigmoid)

        # Reconstruction loss for VAE
        with tf.name_scope('Binary_CE'):
            if self.discrete_pixel == True:
                self.inputs_discrete_pixel = tf.floor(self.inputs * 256)
                self.inputs_discrete_pixel = tf.one_hot(
                    indices=tf.cast(self.inputs_discrete_pixel,tf.uint8),
                    depth=255
                )
                ce = self.inputs_discrete_pixel * tf.log(self.network + 1e-16)
                ce += tf.multiply(
                    1 - self.inputs_discrete_pixel,
                    tf.log(self.network + 1e-16))
                self.ce = -tf.reduce_mean(ce,[1,2,3])
                self.ce_batch_mean = tf.reduce_mean(self.ce)
            else:
                ce = self.inputs * tf.log(self.network + 1e-16)
                ce += (1 - self.inputs) * tf.log(1 - self.network + 1e-16)
                self.ce = - tf.reduce_mean(ce,[1,2,3])
                self.ce_batch_mean = tf.reduce_mean(self.ce)

    def vae_loss(self):
        """
        Implements the loss for the VAE (CrossEntropy + KLDivergence).
        It also enforces sparsity.
        """

        with tf.name_scope('Loss'):
            if self.sparsity_amount > 0:
                with tf.name_scope('SparsityEnforcement'):
                    kl_divs = []
                    for convs in self.convolutions:
                        convs = tf.nn.sigmoid(convs)
                        average_activation = tf.reduce_mean(convs,axis=0)
                        kl_div = tf.add(
                            tf.multiply(
                                self.sparsity_amount,
                                tf.log(self.sparsity_amount/average_activation)
                                ),
                            tf.multiply(
                                1 - self.sparsity_amount,
                                tf.log((1-self.sparsity_amount)/(1-average_activation))
                            )
                        )
                        kl_div = tf.reduce_mean(kl_div)
                        kl_divs.append(kl_div)
                    self.sparsity = tf.add_n(kl_divs) / len(kl_divs)
            else:
                self.sparsity = 0.

            if self.beta_l2_regularization != None:
                self.reg_loss = self.beta_l2_regularization * tf.add_n(
                    slim.losses.get_regularization_losses()
                    )
            else:
                self.reg_loss = 0.

            self.loss = tf.reduce_mean(
                tf.add(
                    self.ce,
                    0.1 * self.kl_div))
            self.loss += tf.add(
                tf.cast(self.reg_loss,tf.float32),
                tf.cast(self.sparsity * self.sparsity_amount,tf.float32))

    def prepare_images(self):
        """
        Function used to prepare a batch of images for training/testing.
        Everything is implement in tensorflow, so it should be fairly fast.
        Apart from this, it also enables conversion to HSV from RGB.
        """

        def pp_image(image):
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = image_augmenter.augment(image)
            image = tf.clip_by_value(image,0.,1.)
            image = tf.clip_by_value(image,0.,1.)
            return image

        def convert_image_float32(image):
            return tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.data_augmentation == True:
            image_augmenter = tf_da.ImageAugmenter(
                **self.data_augmentation_params)
            self.inputs = tf.image.convert_image_dtype(self.inputs,dtype=tf.float32)
            self.inputs = tf.map_fn(
                lambda x: image_augmenter.augment(x),
                self.inputs,
                dtype=tf.float32
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
        self.inputs = tf.image.resize_images(
            self.inputs,
            (self.resize_height,self.resize_width))

    def image_generator(self):
        """
        Creates a tf native input pipeline. Makes everything quite a bit faster
        than using the feed_dict method.
        """

        if self.tfrecords == True:
            def parse_example(serialized_example):
                feature = {
                    'image/encoded': tf.FixedLenFeature([], tf.string),
                    'image/filename': tf.FixedLenFeature([], tf.string),
                    'image/format': tf.FixedLenFeature([], tf.string),
                    'image/class/label': tf.FixedLenFeature([], tf.int64),
                    'image/tp': tf.FixedLenFeature([], tf.int64),
                    'image/height': tf.FixedLenFeature([], tf.int64),
                    'image/width': tf.FixedLenFeature([], tf.int64)
                }
                features = tf.parse_single_example(serialized_example,
                                                   features=feature)
                train_image = tf.image.decode_jpeg(
                    features['image/encoded'],channels=3)
                shape = features['image/height'],features['image/width']
                train_image = tf.reshape(train_image,
                                         [self.height, self.width, 3])

                train_input_queue = features['image/filename']
                class_input = features['image/class/label']
                purity_input = features['image/tp']

                return train_image,train_input_queue,class_input,purity_input

            all_files = glob('{}/*tfrecord*'.format(self.image_path))
            files = tf.data.Dataset.list_files(
                '{}/*tfrecord*'.format(self.image_path))
            dataset = files.interleave(
                tf.data.TFRecordDataset,
                np.minimum(len(all_files)/10,50)
            )
            if self.mode == 'train':
                dataset = dataset.repeat()
                dataset = dataset.shuffle(len(self.image_path_list))
            dataset = dataset.map(parse_example)
            dataset = dataset.batch(self.batch_size)
            if self.mode == 'train':
                dataset = dataset.shuffle(buffer_size=500)
            iterator = dataset.make_one_shot_iterator()

            next_element = iterator.get_next()
            self.train_image,self.train_input_queue,_,_ = next_element

        else:
            image_path_tensor = ops.convert_to_tensor(
                self.image_path_list,dtype=dtypes.string)

            self.train_input_queue = tf.train.slice_input_producer(
                [image_path_tensor],shuffle=False)[0]

            file_content = tf.read_file(self.train_input_queue)
            self.train_image = tf.image.decode_image(file_content,
                                                     channels=3)
            self.train_image.set_shape([self.height,self.width,3])

        if self.mode != 'predict_with_rotation':
            self.inputs = self.train_image
            self.file_names = self.train_input_queue

        else:
            self.inputs = tf.concat(
                [
                    self.train_image,
                    tf.image.rot90(self.train_image,k = 1),
                    tf.image.rot90(self.train_image,k = 2),
                    tf.image.rot90(self.train_image,k = 3)
                ],
                axis = 0
            )
            self.file_names = tf.concat(
                [
                    self.train_input_queue,
                    tf.string_join([self.train_input_queue,'_rot90']),
                    tf.string_join([self.train_input_queue,'_rot180']),
                    tf.string_join([self.train_input_queue,'_rot270'])
                ],
                axis = 0
            )

    def pre_flight_operations(self):
        """
        Creates the summary operations, sets random seeds and creates the
        global variables initializer.
        """

        self.inputs = tf.image.convert_image_dtype(self.inputs,tf.float32)

        self.saver = tf.train.Saver()

        #Setting seeds for randomness
        tf.set_random_seed(self.random_seed)
        np.random.seed(self.random_seed)

        #Session configuration
        self.config = tf.ConfigProto(
            #f,
            **self.config_arguments)

        if self.mode == 'train':

            #Optimizer, minimization and variable initiation
            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=0.5)
            tvars = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                grads_and_vars = optimizer.compute_gradients(self.loss, tvars)

                clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
                           for grad, tvar in grads_and_vars]
                self.train_op = optimizer.apply_gradients(
                    clipped,
                    global_step=self.global_step,
                    name="minimize_cost")

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
                                                 self.ce_batch_mean))
            self.summaries.add(tf.summary.image('inputs',
                                                self.inputs,
                                                max_outputs=4))
            if self.discrete_pixel == True:
                self.summaries.add(tf.summary.image(
                'outputs',
                tf.argmax(self.network,axis=-1) / 255,
                max_outputs=4))
            else:
                self.summaries.add(tf.summary.image('outputs',
                                                    self.network,
                                                    max_outputs=4))
            self.summary_op = tf.summary.merge(list(self.summaries),
                                               name='summary_op')

        elif self.mode == 'test':
            #Metric
            self.mse,self.mse_op = tf.metrics.mean_squared_error(self.inputs,
                                                                 self.network)

        elif self.mode == 'predict':
            pass
        #Defining variables to be initialized
        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Function used to train the VAE.
        """

        print('Run parameters:')
        for var in vars(self):
            pass
            #print('\t{0}={1}'.format(var,vars(self)[var]))

        print('\n')

        with tf.Session(config = self.config) as self.sess:

            #This allows training to continue
            if self.checkpoint_path != None:
                self.saver.restore(self.sess,self.checkpoint_path)
            self.writer = tf.summary.FileWriter(self.save_summary_folder,
                                                self.sess.graph)
            self.sess.run(self.init)

            #Sets everything up to run with the queue from image_generator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.time_list = []

            for i in range(1,int(self.number_of_steps) + 1):
                self.sess.run(tf.local_variables_initializer())

                a = time.time()

                _, _, kl, loss, _ = self.sess.run(
                    [self.train_op,
                     self.mse_op,
                     self.kl_div_batch_mean,
                     self.loss,
                     self.global_step]
                    )
                b = time.time()
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

        with tf.Session(config = self.config) as self.sess:
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

        with tf.Session(config = self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            all_mse = []
            all_kl = []
            self.time_list = []

            #Sets everything up to run with the queue from image_generator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            track = 0
            print(self.number_of_steps)
            with open(self.save_predictions_path,'w') as o:
                for i in range(self.number_of_steps):
                    self.sess.run(tf.local_variables_initializer())
                    a = time.perf_counter()
                    latent, kl_div, file_names  = self.sess.run(
                        [self.latent,
                         self.kl_div,
                         self.file_names])
                    output = self.format_output(file_names,kl_div,latent)
                    b = time.perf_counter()
                    self.time_list.append(b - a)
                    track = track + len(output)
                    print(i,len(output),track,b-a)
                    o.write('\n'.join(output) + '\n')

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def format_output(self,file_names,kl_div,latent):
        """
        Convenience function to format the output of the predict() function.
        """

        out = []
        for file_name,features in zip(file_names,latent):
            features = ','.join(features.astype('str'))
            out.append(file_name.decode('ascii') + ',' + features)

        return(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'vae.py',
        description = 'Variational autoencoder.'
    )

    parser.add_argument('--mode',
                        dest = 'mode',
                        action = 'store',type = str,
                        choices=['train','test','predict','predict_with_rotation'],
                        default = 'train',
                        help = 'Defines the mode.')
    parser.add_argument('--random_seed',
                        dest = 'random_seed',
                        action = 'store',type = int,
                        default = 1,
                        help = 'Random seed for np and tf.')
    #Images
    parser.add_argument('--tfrecords',
                        dest='tfrecords',
                        action='store_true',
                        default = False,
                        help = 'Flag to signal that input is tfrecords.')
    parser.add_argument('--image_path',
                        dest = 'image_path',
                        action = ToDirectory,type = str,
                        required = True,
                        help = 'Path for the folder containing the images.')
    parser.add_argument('--extension',
                        dest = 'extension',
                        action = 'store',type = str,
                        default = 'png',
                        help = 'Extension for the image files.')
    parser.add_argument('--height',
                        dest = 'height',
                        action = 'store',type = int,
                        default = 256,
                        help = 'Image height.')
    parser.add_argument('--width',
                        dest = 'width',
                        action = 'store',type = int,
                        default = 256,
                        help = 'Image width.')
    parser.add_argument('--resize_height',
                        dest = 'resize_height',
                        action = 'store',type = int,
                        default = 256,
                        help = 'Resized image height.')
    parser.add_argument('--resize_width',
                        dest = 'resize_width',
                        action = 'store',type = int,
                        default = 256,
                        help = 'Resized image width.')
    #Preprocessing
    parser.add_argument('--data_augmentation',
                        dest = 'data_augmentation',
                        action = 'store_true',
                        default = False,
                        help = 'Flag to set data augmentation.')
    parser.add_argument('--convert_hsv',
                        dest = 'convert_hsv',
                        action = 'store_true',
                        default = False,
                        help = 'Flag to convert images from RGB to HSV.')
    #Training
    parser.add_argument('--learning_rate',
                        dest = 'learning_rate',
                        action = 'store',type = float,
                        default = 0.001,
                        help = 'Learning rate for training.')
    parser.add_argument('--beta_l2_regularization',
                        dest = 'beta_l2_regularization',
                        action = 'store',type = float,
                        default = None,
                        help = 'Small constant to add to avoid log(0) errors.')
    parser.add_argument('--number_of_steps',
                        dest = 'number_of_steps',
                        action = 'store',type = int,
                        default = 1000,
                        help = 'Number of training steps.')
    parser.add_argument('--epochs',
                        dest = 'epochs',
                        action = 'store',type = int,
                        default = None,
                        help = 'Number of epochs (overrides number_of_steps).')
    parser.add_argument('--save_checkpoint_folder',
                        dest = 'save_checkpoint_folder',
                        action = ToDirectory,type = str,
                        default = '/tmp/checkpoint',
                        help = 'Folder where checkpoints will be stored.')
    parser.add_argument('--save_checkpoint_steps',
                        dest = 'save_checkpoint_steps',
                        action = 'store',type = int,
                        default = 100,
                        help = 'Save checkpoint every n steps.')
    parser.add_argument('--save_summary_folder',
                        dest = 'save_summary_folder',
                        action = ToDirectory,type = str,
                        default = '/tmp/checkpoint',
                        help = 'Folder where summary will be stored.')
    parser.add_argument('--save_summary_steps',
                        dest = 'save_summary_steps',
                        action = 'store',type = int,
                        default = 100,
                        help = 'Save summary every n steps.')

    #Data augmentation
    for arg in [
        ['brightness_max_delta',16. / 255.,float],
        ['saturation_lower',0.8,float],
        ['saturation_upper',1.2,float],
        ['hue_max_delta',0.2,float],
        ['contrast_lower',0.8,float],
        ['contrast_upper',1.2,float],
        ['salt_prob',0.1,float],
        ['pepper_prob',0.1,float],
        ['noise_stddev',0.05,float],
        ['blur_probability',0.1,float],
        ['blur_size',3,int],
        ['blur_mean',0,float],
        ['blur_std',0.05,float],
        ['discrete_rotation',True,'store_true'],
        ['continuous_rotation',True,'store_true'],
        ['min_jpeg_quality',30,int],
        ['max_jpeg_quality',70,int],
        ['elastic_transform_p',0.0,float]
    ]:
        print(arg[0])
        if arg[2] != 'store_true':
            parser.add_argument('--{}'.format(arg[0]),dest=arg[0],
                                action='store',type=arg[2],
                                default=arg[1])
        else:
            parser.add_argument('--{}'.format(arg[0]),dest=arg[0],
                                action='store_true',
                                default=False)

    #Testing/prediction
    parser.add_argument('--checkpoint_path',
                        dest = 'checkpoint_path',
                        action = ToDirectory,type = str,
                        default = None,
                        help = 'Path to checkpoint.')
    parser.add_argument('--save_predictions_path',
                        dest = 'save_predictions_path',
                        action = ToDirectory,type = str,
                        default = './predictions.pkl',
                        help = 'Path to sava predictions.')
    #General parameters
    parser.add_argument('--depth_mult',
                        dest = 'depth_mult',
                        action = 'store',type = float,
                        default = 1,
                        help = 'Multiplier for convolution depth.')
    parser.add_argument('--latent_mult',
                        dest = 'latent_mult',
                        action = 'store',type = float,
                        default = 1,
                        help = 'Multiplier for latent representation dimension.')
    parser.add_argument('--log_every_n_steps',
                        dest = 'log_every_n_steps',
                        action = 'store',type = int,
                        default = 5,
                        help = 'Print log every n steps.')
    parser.add_argument('--batch_size',
                        dest = 'batch_size',
                        action = 'store',type = int,
                        default = 32,
                        help = 'Number of images in each mini-batch.')
    parser.add_argument('--n_latent_layers',
                        dest = 'n_latent_layers',
                        action = 'store',type = int,
                        default = 32,
                        help = """
                        Number of latent layers. For n > 1, these are
                        implemented using inverted augmented flows.
                        """)
    parser.add_argument('--sparsity_amount',
                        dest = 'sparsity_amount',
                        action = 'store',type = float,
                        default = 0.1,
                        help = """
                        Amount of sparsity as described by Makhzani and Frey in
                        arXiv:1312.5663v2.
                        """)

    args = parser.parse_args()

    #Data augmentation
    data_augmentation_params = {
        'brightness_max_delta':args.brightness_max_delta,
        'saturation_lower':args.saturation_lower,
        'saturation_upper':args.saturation_upper,
        'hue_max_delta':args.hue_max_delta,
        'contrast_lower':args.contrast_lower,
        'contrast_upper':args.contrast_upper,
        'salt_prob':args.salt_prob,
        'pepper_prob':args.pepper_prob,
        'noise_stddev':args.noise_stddev,
        'blur_probability':args.blur_probability,
        'blur_size':args.blur_size,
        'blur_mean':args.blur_mean,
        'blur_std':args.blur_std,
        'discrete_rotation':args.discrete_rotation,
        'continuous_rotation':args.continuous_rotation,
        'min_jpeg_quality':args.min_jpeg_quality,
        'max_jpeg_quality':args.max_jpeg_quality,
        'elastic_transform_p':args.elastic_transform_p
    }

    vae = VAE(mode=args.mode,
              random_seed=args.random_seed,
              #Images
              tfrecords=args.tfrecords,
              image_path=args.image_path,
              extension=args.extension,
              height=args.height,
              width=args.width,
              resize_height=args.resize_height,
              resize_width=args.resize_width,
              #Preprocessing
              data_augmentation=args.data_augmentation,
              convert_hsv=args.convert_hsv,
              #Training
              learning_rate=args.learning_rate,
              beta_l2_regularization=args.beta_l2_regularization,
              batch_size=args.batch_size,
              number_of_steps=args.number_of_steps,
              epochs=args.epochs,
              save_checkpoint_folder=args.save_checkpoint_folder,
              save_checkpoint_steps=args.save_checkpoint_steps,
              save_summary_folder=args.save_summary_folder,
              save_summary_steps=args.save_summary_steps,
              #Data augmentation
              data_augmentation_params=data_augmentation_params,
              #Testing/prediction
              checkpoint_path=args.checkpoint_path,
              save_predictions_path=args.save_predictions_path,
              #General parameters
              depth_mult=args.depth_mult,
              config_arguments={},
              log_every_n_steps=args.log_every_n_steps,
              n_latent_layers=args.n_latent_layers,
              sparsity_amount=args.sparsity_amount)
