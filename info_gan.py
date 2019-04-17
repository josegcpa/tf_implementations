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

slim=tf.contrib.slim

_MODE_ERROR="Mode {0!s} has to be one of ['train','test','predict']."
_NO_IMAGES_ERROR="No images in {0!s} with the extension {1!s}."

_SUMMARY_TRAIN='Step {0:d}: Summary saved in {1!s}'
_CHECKPOINT_TRAIN='Step {0:d}: Checkpoint saved in {1!s}'

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

class InfoGAN:
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
                 mode='train',
                 #Images
                 image_path='.',
                 extension='.png',
                 height=128,
                 width=128,
                 resize_height = 64,
                 resize_width = 64,
                 n_channels=3,
                 #Preprocessing
                 data_augmentation=False,
                 convert_hsv=False,
                 #Training
                 learning_rate=0.001,
                 beta_l2_regularization=None,
                 number_of_steps=1000,
                 epochs=None,
                 save_checkpoint_folder='/tmp/checkpoint',
                 save_checkpoint_steps=100,
                 save_summary_folder='/tmp/summary',
                 save_summary_steps=100,
                 #Testing/Prediction
                 checkpoint_path=None,
                 save_predictions_path='./predictions.pkl',
                 #General parameters
                 depth_mult=1.,
                 config_arguments={},
                 log_every_n_steps=5,
                 batch_size=32,
                 noise_size=64,
                 cat_list=[],
                 n_cont=0):

        self.wgan_loss = True
        self.gradient_penalising = True
        self.gradient_clipping = False

        self.mode = mode
        #Images
        self.image_path = image_path
        self.extension = extension
        self.height = height
        self.width = width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.n_channels = n_channels
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
        #Testing/Prediction
        self.checkpoint_path = checkpoint_path
        self.save_predictions_path = save_predictions_path
        #General parameters
        self.depth_mult = depth_mult
        self.config_arguments = config_arguments
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.cat_list = cat_list
        self.n_cont = n_cont

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
            self.is_training = True
        elif self.mode == 'test' or self.mode == 'predict':
            self.is_training = False
            self.data_augmentation = False
            self.number_of_steps = floor(self.no_images / self.batch_size) + 1
        else:
            raise ValueError(_MODE_ERROR.format(self.mode))

        self.save_ckpt_path = os.path.join(
            self.save_checkpoint_folder, 'my-model.ckpt'
            )

        #Getting the categorical list in a Python-readable format
        self.cat_list = self.cat_list.split(',')
        try:
            self.cat_list = [int(i) for i in self.cat_list]
        except:
            self.cat_list = []
        #Begin serious operations
        self.make_dirs()
        print('Filtering images by size...')
        #self.filter_size()
        print('Done - {0:d} images.'.format(self.no_images))
        self.image_generator()
        self.prepare_images()
        self.info_gan()

        if self.mode == 'train':
            self.pre_flight_operations()
            self.train()
        if self.mode == 'test':
            self.pre_flight_operations()
            self.test()
        if self.mode == 'predict':
            self.vae_loss()
            self.pre_flight_operations()
            self.predict()

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
        train_image = tf.image.decode_image(file_content,
                                            channels=self.n_channels)
        train_image.set_shape([self.height, self.width, self.n_channels])
        self.inputs, self.file_names = tf.train.shuffle_batch(
            [train_image,train_input_queue],
            batch_size=self.batch_size,
            capacity=60000,
            min_after_dequeue=30000,
            allow_smaller_final_batch=True
            )
        self.inputs = tf.image.resize_images(
            self.inputs,[self.resize_height,self.resize_width])
        self.n_images_batch = tf.shape(self.inputs)[0]

    def info_gan(self):
        """
        Function to create the InfoGAN.
        """

        def pixel_normalization(inputs):
            """
            The pixel normalization presented in the ProGAN paper. This
            normalization is basically a local response normalization performed
            depth-wise. It offers the advantage of having no learnable
            parameters, which makes training faster, and allows the usage of
            the WGAN-GP loss function.
            This is used to normalize the layers from both the generator (as
            per the ProGAN paper) and the discriminator (as the WGAN-GP paper
            suggests).
            """
            norm_factor = tf.sqrt(
                tf.reduce_mean(tf.square(inputs),axis=-1) + 1e-16
            )
            norm_factor = tf.expand_dims(norm_factor,-1)
            return inputs / norm_factor

        def generator(noise):
            if self.beta_l2_regularization != None:
                weights_regularizer = tf.contrib.layers.l2_regularizer(
                    self.beta_l2_regularization
                )
            else:
                weights_regularizer = None

            with tf.name_scope('Generator') and tf.variable_scope('Generator'):
                with slim.arg_scope(
                    [slim.convolution2d],
                    activation_fn=tf.nn.relu,
                    kernel_size=[3,3],
                    padding='SAME',
                    normalizer_fn=pixel_normalization,
                    weights_regularizer=weights_regularizer
                ):
                    noise_shape = tf.shape(noise)
                    fc_noise = slim.fully_connected(
                        noise,4 * 4 * 256,
                        activation_fn=tf.nn.relu,
                        scope='fc_noise')

                    reshaped_noise = tf.reshape(
                        fc_noise,
                        [-1,4,4,256],
                        name='reshaped_noise')

                    conv_1 = slim.convolution2d(
                        reshaped_noise,
                        num_outputs=128,
                        scope='conv_1')
                    conv_1 = tf.depth_to_space(
                        conv_1,2,name='resize_1')

                    conv_2 = slim.convolution2d(
                        conv_1,
                        num_outputs=64,
                        scope='conv_2')
                    conv_2 = tf.depth_to_space(
                        conv_2,2,name='resize_2')

                    conv_3 = slim.convolution2d(
                        conv_2,
                        num_outputs=32,
                        scope='conv_3')
                    conv_3 = tf.depth_to_space(
                        conv_3,2,name='resize_3')

                    conv_4 = slim.convolution2d(
                        conv_3,
                        num_outputs=16,
                        scope='conv_4')
                    conv_4 = tf.depth_to_space(
                        conv_4,2,name='resize_4')

                    conv_5 = slim.convolution2d(
                        conv_4,
                        num_outputs=self.inputs.get_shape().as_list()[-1],
                        biases_initializer=None,
                        activation_fn=tf.nn.tanh,
                        normalizer_fn=None,
                        scope='conv_5')
            return conv_5

        def discriminator(image, cat_list, n_cont, reuse=False):
            if self.beta_l2_regularization != None:
                weights_regularizer = tf.contrib.layers.l2_regularizer(
                    self.beta_l2_regularization * 10
                )
            else:
                weights_regularizer = None
            print(image)
            initializer = tf.initializers.truncated_normal(stddev=0.02)
            with tf.name_scope('Discriminator') and\
             tf.variable_scope('Discriminator'):
                with slim.arg_scope(
                    [slim.convolution2d],
                    activation_fn=tf.nn.leaky_relu,
                    kernel_size=[3,3],
                    padding='SAME',
                    normalizer_fn=pixel_normalization,
                    weights_regularizer=weights_regularizer
                ):
                    conv_1 = slim.convolution2d(
                        image,16,
                        biases_initializer=None,
                        normalizer_fn=None,
                        reuse=reuse,scope='conv_1')
                    conv_1 = tf.space_to_depth(
                        conv_1,2,name='resize_1')
                    conv_1 = slim.dropout(
                        conv_1,
                        keep_prob=0.3,
                        is_training=self.is_training,
                        scope='dropout_1'
                    )

                    conv_2 = slim.convolution2d(
                        conv_1,32,
                        biases_initializer=None,
                        reuse=reuse,scope='conv_2')
                    conv_2 = tf.space_to_depth(
                        conv_2,2,name='resize_2')
                    conv_2 = slim.dropout(
                        conv_2,
                        keep_prob=0.3,
                        is_training=self.is_training,
                        scope='dropout_2'
                    )

                    conv_3 = slim.convolution2d(
                        conv_2,64,
                        biases_initializer=None,
                        reuse=reuse,scope='conv_3')
                    conv_3 = tf.space_to_depth(
                        conv_3,2,name='resize_3')
                    conv_3 = slim.dropout(
                        conv_3,
                        keep_prob=0.3,
                        is_training=self.is_training,
                        scope='dropout_3'
                    )

                    conv_4 = slim.convolution2d(
                        conv_3,128,
                        biases_initializer=None,
                        reuse=reuse,scope='conv_4')
                    conv_4 = tf.space_to_depth(
                        conv_4,2,name='resize_4')
                    conv_4 = slim.dropout(
                        conv_4,
                        keep_prob=0.3,
                        is_training=self.is_training,
                        scope='dropout_4'
                    )

                    flat = slim.flatten(conv_4)
                    fc_1 = slim.fully_connected(
                        flat,
                        1024,
                        activation_fn=tf.nn.leaky_relu,
                        reuse=reuse,
                        scope='fc_1',
                        weights_initializer=initializer)

                    cl = slim.fully_connected(
                        fc_1,
                        1,
                        activation_fn=None,
                        reuse=reuse,
                        scope='classification',
                        weights_initializer=initializer)

            with tf.name_scope('Latent_Representations') and\
             tf.variable_scope('Latent_Representations'):
                pre_latent = slim.fully_connected(
                    fc_1,
                    128,
                    activation_fn=tf.nn.leaky_relu,
                    reuse=reuse,
                    scope='pre_latent_1',
                    weights_initializer=initializer)
                pre_latent = slim.fully_connected(
                    pre_latent,
                    256,
                    activation_fn=tf.nn.leaky_relu,
                    reuse=reuse,
                    scope='pre_latent_2',
                    weights_initializer=initializer)
                cat_outs = []
                for i,c in enumerate(cat_list):
                    cat_out = slim.fully_connected(
                        pre_latent,
                        c,
                        activation_fn=None,
                        reuse=reuse,
                        scope='cat_out_' + str(i),
                        weights_initializer=initializer)

                    cat_outs.append(cat_out)

                if self.n_cont > 0:
                    cont_outs = slim.fully_connected(
                        pre_latent,
                        self.n_cont,
                        activation_fn=tf.nn.tanh,
                        reuse=reuse,
                        scope='cont_out_' + str(self.n_cont),
                        weights_initializer=initializer)
                else:
                    cont_outs = None

            return cl,cat_outs,cont_outs

        self.discriminator = discriminator

        with tf.variable_scope('InfoGAN'):
            self.noise_input = tf.random_normal(
                [self.n_images_batch,self.noise_size],
                name='noise_input')
            self.cat_inputs = []
            for i,c in enumerate(self.cat_list):
                cat_input = tf.random_uniform(
                    shape=[self.n_images_batch],
                    minval=0,
                    maxval=c,
                    dtype=tf.int32,
                    name='cat_in_' + str(i))
                cat_input = tf.one_hot(cat_input,
                                       depth=c)
                self.cat_inputs.append(cat_input)
            self.cont_input = tf.random_uniform(
                shape=[self.n_images_batch,self.n_cont],
                minval=-1,
                maxval=1,
                dtype=tf.float32,
                name='cont_in_' + str(self.n_cont))
            self.full_gen_input = tf.concat(
                [self.noise_input,
                 *self.cat_inputs,
                 self.cont_input],
                axis=1
            )
            self.shake_sd = tf.Variable(0.,trainable=False)
            self.shake = tf.random_normal(
                tf.shape(self.inputs),
                mean=0.0,
                stddev=self.shake_sd)
            self.generated_images = generator(self.full_gen_input)

            self.cl_real,_,_ = self.discriminator(self.inputs + self.shake,
                self.cat_list,self.n_cont)
            self.cl_gen,self.cat_gen,self.cont_gen = self.discriminator(
                self.generated_images,self.cat_list,self.n_cont,
                reuse=True)
            self.sess_debugger(self.cl_gen)

    def prepare_images(self):
        """
        Function used to prepare a batch of images for training/testing.
        Everything is implement in tensorflow, so it should be fairly fast.
        Apart from this, it also enables conversion to HSV from RGB.
        """

        def pp_image(image):
            def distort_colors_0(image):
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                #image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                return image

            def distort_colors_1(image):
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                #image = tf.image.random_hue(image, max_delta=0.05)
                return image

            def distort_colors_2(image):
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                #image=tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                return image

            def distort_colors(image,color_ordering):
                image = tf.cond(
                    tf.equal(color_ordering,0),
                    lambda: distort_colors_0(image),
                    lambda: tf.cond(
                        tf.equal(color_ordering,1),
                        lambda: distort_colors_1(image),
                        lambda: tf.cond(tf.equal(color_ordering,2),
                            lambda: distort_colors_2(image),
                            lambda: image)
                    )
                )
                return image

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            color_ordering = tf.random_uniform([],0,6,tf.int32)
            if self.n_channels == 3 and self.mode == 'train':
                image = distort_colors(image,color_ordering)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(
                image,
                k=tf.squeeze(tf.random_uniform([1],0,4,tf.int32))
                )
            image = tf.clip_by_value(image,0,1)
            image = tf.divide(
                image - tf.reduce_min(image),
                tf.reduce_max(image) - tf.reduce_min(image)
            ) * 2
            image = image - 2

            return image

        def convert_image_float32(image):
            return tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.data_augmentation == True:
            self.inputs = tf.map_fn(
                pp_image,
                self.inputs,
                dtype=tf.float32
            )

        else:
            self.inputs = tf.map_fn(
                convert_image_float32,
                self.inputs,
                dtype=tf.float32
            )

        if self.convert_hsv == True:
            self.inputs = tf.image.rgb_to_hsv(
                self.inputs
            )

    def pre_flight_operations(self):
        """
        Defines the loss functions, creates the gradient update and summary
        operations, sets random seeds and creates the global variables
        initializer.
        """

        def clip_gradients(gradients):
            return [(tf.clip_by_norm(v[0],0.5),v[1]) for v in gradients]

        def random_like(tensor):
            return tf.cast(tf.random_uniform(tf.shape(tensor),0.,1.) > 0.5,
                           tf.float32)
        def maxmin(tensor,maximum,minimum):
            return tf.maximum(tf.minimum(tensor,minimum),maximum)

        self.saver = tf.train.Saver()

        #Setting seeds for randomness
        tf.set_random_seed(42)
        np.random.seed(42)

        #Session configuration
        self.config = tf.ConfigProto(**self.config_arguments)

        if self.mode == 'train':

            #Optimizer, minimization and variable initiation
            trainable_var = tf.trainable_variables()
            generator_variables = []
            discriminator_variables = []
            latent_variables = []

            for var in trainable_var:
                print(var)
                if 'Generator' in var.name:
                    generator_variables.append(var)
                elif 'Discriminator' in var.name:
                    discriminator_variables.append(var)
                elif 'Latent_Representation' in var.name:
                    latent_variables.append(var)
                if 'Discriminator/conv_4' in var.name:
                    latent_variables.append(var)
                elif 'Discriminator/fc_1' in var.name:
                    latent_variables.append(var)

            self.opt_cl = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate * 0.2,beta1=0.5)
            self.opt_gen = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,beta1=0.9)
            self.opt_latent = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,beta1=0.5)

            # Defining all losses
            self.smooth = tf.Variable(0.,trainable=False)
            self.random_prob = tf.Variable(0.,trainable=False)
            ins = tf.cond(
                tf.random_uniform([],0.,1.) < (1. - self.random_prob),
                lambda: tf.zeros_like(self.cl_real),
                lambda: random_like(self.cl_real))
            self.labels_real = maxmin(
                tf.ones_like(self.cl_real) - ins - self.smooth,1,0)
            self.labels_gen = tf.zeros_like(self.cl_gen)

            if self.wgan_loss == True:
                # Earthmover's distance loss from the WGAN paper
                base_disc_loss = tf.subtract(
                    tf.reduce_mean(self.cl_real),
                    tf.reduce_mean(self.cl_gen)
                    )
                self.gen_loss = tf.reduce_mean(tf.reduce_mean(self.cl_gen))
            else:
                # Vanilla GAN loss
                base_disc_loss = tf.subtract(
                    tf.reduce_mean(tf.log(self.cl_real + 1e-16)),
                    tf.reduce_mean(tf.log(1 - self.cl_gen + 1e-16))
                    )

            if self.gradient_penalising == True:
                # Gradient penalising from the WGAN-GP paper. This should be
                # used with the WGAN loss. The gradient extraction step
                # was adapted from https://github.com/changwoolee/WGAN-GP-tensorflow/blob/master/model.py
                self.epsilon = tf.random_uniform(
                    shape=[self.batch_size, 1, 1, 1],
                    minval=0.,
                    maxval=1.)
                sample_x_hat = tf.add(
                    self.inputs,
                    self.epsilon * tf.subtract(self.generated_images,
                                               self.inputs)
                    )
                with tf.variable_scope('InfoGAN'):
                    class_sample_x_hat = self.discriminator(
                        sample_x_hat,
                        cat_list=[],
                        n_cont=0,
                        reuse=True)
                grad_sample_x_hat = tf.gradients(
                    class_sample_x_hat,[sample_x_hat])[0]
                grad_reg = tf.reduce_mean(
                    tf.square(tf.norm(grad_sample_x_hat) - 1.)
                )
                self.disc_loss = base_disc_loss + grad_reg
            else:
                self.disc_loss = base_disc_loss

            self.cat_losses = []
            for i,c in enumerate(self.cat_inputs):
                cat_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=c,
                    logits=self.cat_gen[i]
                )
                self.cat_losses.append(cat_loss)
            self.cat_loss = tf.reduce_mean(self.cat_losses)
            if self.n_cont > 0:
                self.cont_loss=tf.losses.mean_squared_error(
                    labels=self.cont_input,
                    predictions=self.cont_gen
                )
                self.cont_loss = tf.reduce_mean(self.cont_loss)
            else:
                self.cont_loss = 0.0

            self.latent_loss = tf.abs(
                self.gen_loss - (self.cont_loss + self.cat_loss))

            self.loss = tf.add_n(
                [self.latent_loss,
                self.disc_loss,
                self.gen_loss,
                tf.reduce_mean(tf.losses.get_regularization_losses())]
            )

            if self.gradient_clipping == True:
                cl_grads = clip_gradients(cl_grads)

            # Compute gradients
            cl_grads = self.opt_cl.compute_gradients(
                self.disc_loss,discriminator_variables)
            gen_grads = self.opt_gen.compute_gradients(
                self.gen_loss,generator_variables)
            latent_grads = self.opt_latent.compute_gradients(
                self.latent_loss,latent_variables)

            # Apply gradients
            self.cl_train_op = self.opt_cl.apply_gradients(cl_grads)
            self.gen_train_op = self.opt_gen.apply_gradients(gen_grads)

            if len(self.cat_list) > 0 or self.n_cont > 0:
                self.latent_train_op = self.opt_latent.apply_gradients(latent_grads)
                self.train_op = tf.group(
                    self.cl_train_op,
                    self.gen_train_op,
                    self.latent_train_op)
            else:
                # This option enables the training of a vanilla GAN by simply
                # not adding any continuous or categorical latent dimensions.
                self.train_op = tf.group(
                    self.cl_train_op,
                    self.gen_train_op)

            #Metrics
            self.sensitivity = tf.reduce_sum(
                tf.cast(tf.nn.sigmoid(self.cl_real) > 0.5,tf.float32)
            ) / tf.cast(self.n_images_batch,tf.float32)
            self.specificity = tf.reduce_sum(
                tf.cast(tf.nn.sigmoid(self.cl_gen) < 0.5,tf.float32)
            ) / tf.cast(self.n_images_batch,tf.float32)

            #Summaries
            self.summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            for variable in slim.get_model_variables():
                self.summaries.add(
                    tf.summary.histogram(variable.op.name, variable)
                    )
            self.summaries.add(tf.summary.scalar('loss', self.loss))
            self.summaries.add(tf.summary.scalar('disc_loss',
                                                 self.disc_loss))
            self.summaries.add(tf.summary.scalar('gen_loss',
                                                 self.gen_loss))
            self.summaries.add(tf.summary.scalar('latent_loss',
                                                 self.latent_loss))
            self.summaries.add(
                tf.summary.image('real_images',
                                 self.inputs,
                                 max_outputs=4))
            self.summaries.add(
                tf.summary.image('generated_images',
                                 self.generated_images,
                                 max_outputs=4)
            )

            self.generated_images
            self.summary_op=tf.summary.merge(list(self.summaries),
                                               name='summary_op')


        elif self.mode == 'test':
            pass

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

        with tf.Session(config=self.config) as self.sess:

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
            for i in range(1,self.number_of_steps + 1):
                decay = tf.divide(self.number_of_steps + 1 - i,
                                  self.number_of_steps + 1)
                self.sess.run(tf.local_variables_initializer())

                a = time.perf_counter()

                l1,l2,l3,sen,spe,r,g = self.sess.run(
                    [self.gen_loss,
                     self.disc_loss,
                     self.latent_loss,
                     self.sensitivity,
                     self.specificity,
                     self.labels_real,
                     self.labels_gen]
                    )

                reassignments = tf.group([
                    tf.assign(self.smooth,spe * 0.1),
                    tf.assign(self.shake_sd,0.05 * decay),
                    tf.assign(self.random_prob,(0.1 * np.exp(decay))/np.exp(1))
                ])

                self.sess.run(reassignments)

                self.sess.run(self.gen_train_op)
                self.sess.run(self.latent_train_op)
                self.sess.run(self.cl_train_op)

                b = time.perf_counter()
                self.time_list.append(b - a)
                if i % self.log_every_n_steps == 0 or\
                 i % self.number_of_steps == 0 or i == 1:
                    last_time = self.time_list[-1]
                    print("""Step: {0}
                    \tLoss: {1}""".format(i,[l1,l2,l3]))
                    print("""
                    \tSensitivity: {0}
                    \tSpecificity: {1}""".format(sen,spe))

                if i % self.save_summary_steps == 0 or i == 1:
                    summary = self.sess.run(self.summary_op)
                    self.writer.add_summary(summary,i)
                    print("""Summary saved in: {0} (step {1})""".format(
                        self.save_summary_folder,i)
                        )

                if i % self.save_checkpoint_steps == 0 or i == 1:
                    self.saver.save(self.sess, self.save_ckpt_path,
                                    global_step=i)
                    print("""Checkpoint saved in: {0} (step {1})""".format(
                        self.save_checkpoint_folder,i)
                        )

            self.saver.save(self.sess, self.save_ckpt_path,global_step=i)
            print("""Checkpoint saved in: {0} (step {1})""".format(
                self.save_checkpoint_folder,i)
                )
            self.writer.add_summary(summary,i)
            print("""Checkpoint saved in: {0} (step {1})""".format(
                self.save_summary_folder,i)
                )

            loss = self.sess.run(self.loss)
            print("""
            Training finished. Av. time: {}. Last loss: {}""".format(
                np.mean(self.time_list),loss))
            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def predict(self):
        """
        Function used to predict the latent dimension of the VAE. The output is
        stored in a .npy file with information regardin the latent dimension,
        KL divergence and the file name.
        """

        with tf.Session(config=self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            self.time_list=[]

            #Sets everything up to run with the queue from image_generator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            output = []

            for i in range(self.number_of_steps):
                self.sess.run(tf.local_variables_initializer())
                a = time.perf_counter()
                file_names,cl, cat_code, cont_code =self.sess.run(
                    [self.file_names,
                     self.cl_gen,
                     self.cat_gen,
                     self.cont_gen])

                b = time.perf_counter()
                self.time_list.append(b - a)
                output.extend(self.format_output(
                    file_names,cl,cat_code,cont_code))

            np.save(self.save_predictions_path,
                    np.array(output,dtype=object))

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def format_output(self,file_names,cl,cat_code,cont_code):
        """
        Convenience function to format the output of the self.predict()
        function.
        """

        output = []
        for i in range(file_names.size):
            features = ','.join(features.astype('str'))
            output.append(','.join([
                                   file_names[i].decode('ascii'),
                                   cl[i],
                                   cat_code[i,:],
                                   cont_code[i,:]]))

        return output

parser = argparse.ArgumentParser(
    prog='vae.py',
    description='Variational autoencoder.'
)

parser.add_argument('--mode',dest='mode',
                    action='store',type=str,
                    choices=['train','test','predict'],
                    default='train',
                    help='Defines the mode.')
#Images
parser.add_argument('--image_path',dest='image_path',
                    action=ToDirectory,type=str,
                    required=True,
                    help='Path for the folder containing the images.')
parser.add_argument('--extension',dest='extension',
                    action='store',type=str,
                    choices=['png','jpg','gif'],
                    default='png',
                    help='Extension for the image files.')
parser.add_argument('--height',dest='height',
                    action='store',type=int,
                    default=28,
                    help='Image height.')
parser.add_argument('--width',dest='width',
                    action='store',type=int,
                    default=28,
                    help='Image width.')
parser.add_argument('--resize_height',dest='resize_height',
                    action='store',type=int,
                    default=64,
                    help='Resize image height.')
parser.add_argument('--resize_width',dest='resize_width',
                    action='store',type=int,
                    default=64,
                    help='Resize image widht.')
parser.add_argument('--n_channels',dest='n_channels',
                    action='store',type=int,
                    default=3,
                    help='Number of channels in image.')

#Preprocessing
parser.add_argument('--data_augmentation',dest='data_augmentation',
                    action='store_true',
                    default=False,
                    help='Flag to set data augmentation.')
parser.add_argument('--convert_hsv',dest='convert_hsv',
                    action='store_true',
                    default=False,
                    help='Flag to convert images from RGB to HSV.')
#Training
parser.add_argument('--learning_rate',dest='learning_rate',
                    action='store',type=float,
                    default=0.001,
                    help='Learning rate for training.')
parser.add_argument('--beta_l2_regularization',dest='beta_l2_regularization',
                    action='store',type=float,
                    default=None,
                    help='Small constant to add to avoid log(0) errors.')
parser.add_argument('--number_of_steps',dest='number_of_steps',
                    action='store',type=int,
                    default=1000,
                    help='Number of training steps.')
parser.add_argument('--epochs',dest='epochs',
                    action='store',type=int,
                    default=None,
                    help='Number of epochs (overrides number_of_steps).')
parser.add_argument('--save_checkpoint_folder',dest='save_checkpoint_folder',
                    action=ToDirectory,type=str,
                    default='/tmp/checkpoint',
                    help='Folder where checkpoints will be stored.')
parser.add_argument('--save_checkpoint_steps',dest='save_checkpoint_steps',
                    action='store',type=int,
                    default=100,
                    help='Save checkpoint every n steps.')
parser.add_argument('--save_summary_folder',dest='save_summary_folder',
                    action=ToDirectory,type=str,
                    default='/tmp/checkpoint',
                    help='Folder where summary will be stored.')
parser.add_argument('--save_summary_steps',dest='save_summary_steps',
                    action='store',type=int,
                    default=100,
                    help='Save summary every n steps.')
#Testing/prediction
parser.add_argument('--checkpoint_path',dest='checkpoint_path',
                    action=ToDirectory,type=str,
                    default=None,
                    help='Path to checkpoint.')
parser.add_argument('--save_predictions_path',dest='save_predictions_path',
                    action=ToDirectory,type=str,
                    default='./predictions.pkl',
                    help='Path to sava predictions.')
#General parameters
parser.add_argument('--log_every_n_steps',dest='log_every_n_steps',
                    action='store',type=int,
                    default=5,
                    help='Print log every n steps.')
parser.add_argument('--batch_size',dest='batch_size',
                    action='store',type=int,
                    default=32,
                    help='Number of images in each mini-batch.')
parser.add_argument('--noise_size',dest='noise_size',
                    action='store',type=int,
                    default=64,
                    help='Size of the noise vector.')
parser.add_argument('--cat_list',dest='cat_list',
                    action='store',type=str,
                    default='',
                    help='List of categorical latent dimensions (comma-separated).')
parser.add_argument('--n_cont',dest='n_cont',
                    action='store',type=int,
                    default=0,
                    help='Number of continuous latent dimensions.')

args=parser.parse_args()

vae=InfoGAN(mode=args.mode,
              #Images
              image_path=args.image_path,
              extension=args.extension,
              height=args.height,
              width=args.width,
              resize_height=args.resize_height,
              resize_width=args.resize_width,
              n_channels=args.n_channels,
              #Preprocessing
              data_augmentation=args.data_augmentation,
              convert_hsv=args.convert_hsv,
              #Training
              learning_rate=args.learning_rate,
              beta_l2_regularization=args.beta_l2_regularization,
              number_of_steps=args.number_of_steps,
              epochs=args.epochs,
              save_checkpoint_folder=args.save_checkpoint_folder,
              save_checkpoint_steps=args.save_checkpoint_steps,
              save_summary_folder=args.save_summary_folder,
              save_summary_steps=args.save_summary_steps,
              #Testing/prediction
              checkpoint_path=args.checkpoint_path,
              save_predictions_path=args.save_predictions_path,
              #General parameters
              config_arguments={},
              log_every_n_steps=args.log_every_n_steps,
              batch_size=args.batch_size,
              noise_size=args.noise_size,
              cat_list=args.cat_list,
              n_cont=args.n_cont)
