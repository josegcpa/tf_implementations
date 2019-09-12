import os
import numpy as np
import argparse
import random
from glob import glob
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

slim = tf.contrib.slim

N_ITER_D = 1
N_ITER_G = 1

LOG ="""
Step: {}/{}
    Generator_IdealToNonIdealLoss = {}
    Discriminator_NonIdealLoss = {}
    CycleConsistencyLossIdeal = {}
    GradientPenaltyNonIdeal = {}

    Generator_NonIdealToIdealLoss = {}
    Discriminator_IdealLoss = {}
    CycleConsistencyLossNonIdeal = {}
    GradientPenaltyIdeal = {}

    Full_loss = {}"""

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

class CycleGAN():

    def __init__(
        self,
        mode='train',
        ideal_image_path=None,
        nonideal_image_path=None,
        extension='png',
        height=512,
        resize_height=512,
        width=512,
        resize_width=512,
        n_channels=3,
        n_classes=2,
        batch_size=4,
        beta_l2_regularization=0.005,
        lambda_cycle=10,
        wasserstein=False,
        compound_loss=False):

        self.mode = mode
        self.ideal_image_path = ideal_image_path
        self.nonideal_image_path = nonideal_image_path
        self.extension = extension
        self.height = height
        self.resize_height = resize_height
        self.width = width
        self.resize_width = resize_width
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.beta_l2_regularization = beta_l2_regularization
        self.lambda_cycle = lambda_cycle
        self.wasserstein = wasserstein
        self.compound_loss = compound_loss

        if self.mode == 'train':
            if self.ideal_image_path == None:
                raise Exception(
                    'ideal_image_path is necessary for training.')
            if self.nonideal_image_path == None:
                raise Exception(
                    'nonideal_image_path is necessary for training.')
            self.is_training = True
            self.data_augmentation = True

        else:
            if self.ideal_image_path == None:
                self.ideal_image_path = self.nonideal_image_path
            elif self.nonideal_image_path == None:
                self.nonideal_image_path = self.ideal_image_path

        self.ideal_image_path_list = glob(
            os.path.join(self.ideal_image_path,'*' + self.extension))
        self.nonideal_image_path_list = glob(
            os.path.join(self.nonideal_image_path,'*' + self.extension))

        self.image_generator()
        self.prepare_images()
        self.cycle_gan()
        self.get_loss()

    def image_generator(self):
        """
        Creates a tf native input pipeline. Makes everything quite a bit faster
        than using the feed_dict method.
        """

        def paths_to_images(path_list):
            image_path_tensor = ops.convert_to_tensor(
                path_list,
                dtype=dtypes.string)
            train_input_queue = tf.train.slice_input_producer(
                [image_path_tensor],
                shuffle=False)[0]

            file_content = tf.read_file(train_input_queue)
            train_image = tf.image.decode_image(file_content,
                                                channels=self.n_channels)
            train_image.set_shape([self.height, self.width, self.n_channels])
            inputs, file_names = tf.train.shuffle_batch(
                [train_image,train_input_queue],
                batch_size=self.batch_size,
                capacity=100,
                min_after_dequeue=50,
                allow_smaller_final_batch=True
                )
            inputs = tf.image.resize_images(
                inputs,[self.resize_height,self.resize_width])

            return inputs,file_names

        self.inputs_ideal,self.file_names_ideal = paths_to_images(
            self.ideal_image_path_list)
        self.inputs_non_ideal,self.file_names_nonideal = paths_to_images(
            self.nonideal_image_path_list)

    def prepare_images(self):
        """
        Function used to prepare a batch of images for training/testing.
        Everything is implement in tensorflow, so it should be fairly fast.
        Apart from this, it also enables conversion to HSV from RGB.
        """

        def pp_image(image):
            def distort_colors_0(image):
                image = tf.image.random_brightness(image, max_delta=16. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                return image

            def distort_colors_1(image):
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=16. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                return image

            def distort_colors_2(image):
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image=tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=16. / 255.)
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
                #image = distort_colors(image,color_ordering)
                image = image
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(
                image,
                k=tf.squeeze(tf.random_uniform([1],0,4,tf.int32))
                )
            image = tf.clip_by_value(image,0,1) * 2
            image = image - 1

            return image

        def convert_image_float32(image):
            return tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.data_augmentation == True:
            self.inputs_ideal = tf.map_fn(
                pp_image,
                self.inputs_ideal,
                dtype=tf.float32
            )
            self.inputs_non_ideal = tf.map_fn(
                pp_image,
                self.inputs_non_ideal,
                dtype=tf.float32
            )

        else:
            self.inputs_ideal = tf.map_fn(
                convert_image_float32,
                self.inputs,
                dtype=tf.float32
            )
            self.inputs_non_ideal = tf.map_fn(
                convert_image_float32,
                self.inputs_non_ideal,
                dtype=tf.float32
            )

    def cycle_gan(self):
        """
        Function to create the CycleGAN.
        """

        def layer_normalization(inputs):
            mean = tf.reduce_mean(inputs)
            std = tf.sqrt(
                tf.reduce_mean(
                    tf.square(inputs - mean)
                    )
                )
            return (inputs - mean) / std

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
                tf.reduce_mean(tf.square(inputs),axis=-1)
            )
            norm_factor = tf.expand_dims(norm_factor,-1)
            norm_factor = tf.where(norm_factor > 0,
                                   tf.ones_like(norm_factor)/norm_factor,
                                   tf.zeros_like(norm_factor))

            return inputs * norm_factor

        def residual_block(inputs,scope):
            depth = inputs.shape.as_list()[-1]
            sc = 'Residual_Block_{}'.format(scope)
            with tf.name_scope(sc) and tf.variable_scope(sc):
                conv_1 = slim.convolution2d(inputs,num_outputs=depth,
                                            scope='conv_1')
                conv_2 = slim.convolution2d(conv_1,num_outputs=depth,
                                            scope='conv_2')
                output = tf.add(inputs,conv_2,name='addition')
            return output

        def generator(image,reuse=False,scope=''):
            if self.beta_l2_regularization != None:
                weights_regularizer = tf.contrib.layers.l2_regularizer(
                    self.beta_l2_regularization
                )
            else:
                weights_regularizer = None

            initializer = tf.initializers.random_normal(stddev=0.02)

            with tf.name_scope('Generator_{}'.format(scope)) and\
             tf.variable_scope('Generator_{}'.format(scope)):
                with slim.arg_scope(
                    [slim.convolution2d,slim.conv2d_transpose],
                    activation_fn=tf.nn.relu,
                    kernel_size=[3,3],
                    padding='SAME',
                    weights_initializer=initializer,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training':self.is_training},
                    reuse=reuse
                ):
                    conv = image
                    for i,n in enumerate([(32,1),(64,2),(128,2)]):
                        conv = slim.convolution2d(
                            conv,
                            num_outputs=n[0],
                            stride=n[1],
                            scope='conv_{}'.format(i))

                    res_1 = residual_block(conv,0)
                    res_2 = residual_block(res_1,1)
                    res_3 = residual_block(res_2,2)
                    res_4 = residual_block(res_3,3)
                    res_5 = residual_block(res_4,4)

                    conv = slim.conv2d_transpose(
                        res_5,
                        num_outputs=64,
                        stride=2,
                        scope='conv_4')

                    conv = slim.conv2d_transpose(
                        conv,
                        num_outputs=32,
                        stride=2,
                        scope='conv_5')

                    conv = slim.convolution2d(
                        conv,
                        stride=1,
                        kernel_size=[9,9],
                        num_outputs=3,
                        activation_fn=tf.nn.tanh,
                        normalizer_fn=None,
                        scope='conv_3')

            return conv

        def discriminator(images,n_classes,reuse=False,scope=''):
            if self.beta_l2_regularization != None:
                weights_regularizer = tf.contrib.layers.l2_regularizer(
                    self.beta_l2_regularization
                )
            else:
                weights_regularizer = None

            initializer = tf.initializers.random_normal(stddev=0.02)

            with tf.name_scope('Discriminator_{}'.format(scope)) and\
             tf.variable_scope('Discriminator_{}'.format(scope)):
                batch,h,w,c = images.shape.as_list()
                patch_h = h // 4
                patch_w = w // 4
                patch_stride_h = patch_h // 2
                patch_stride_w = patch_w // 2
                patches = tf.image.extract_image_patches(
                    images,
                    [1,patch_h,patch_w,1],
                    [1,patch_stride_h,patch_stride_w,1],
                    [1,1,1,1],
                    'SAME'
                )
                reshaped_patches = tf.reshape(
                    patches,
                    [tf.reduce_prod(tf.shape(patches)[:3]),
                     patch_h,
                     patch_w,c])
                with slim.arg_scope(
                    [slim.convolution2d],
                    activation_fn=tf.nn.relu,
                    kernel_size=[3,3],
                    padding='SAME',
                    weights_initializer=initializer,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training':self.is_training},
                    reuse=reuse
                ):
                    conv = images
                    for i,n in enumerate([32,64,128,256,512]):
                        conv = slim.convolution2d(
                            conv,n,
                            reuse=reuse,scope='conv_A_{}'.format(i))
                        conv = slim.convolution2d(
                            conv,n*2,
                            reuse=reuse,scope='conv_B_{}'.format(i))
                        conv = slim.max_pool2d(
                            conv,[2,2],stride=2,
                            scope='max_pool_{}'.format(i))

                    _,h,w,c = conv.shape.as_list()

                    features = slim.max_pool2d(conv,[h,w],stride=1,
                                               scope='global_pooling',
                                               padding='VALID')

                    features = tf.layers.flatten(features)
                    features = slim.fully_connected(
                        features,
                        256,
                        activation_fn=tf.nn.relu,
                        reuse=reuse,
                        weights_initializer=initializer,
                        scope='pre_classification')
                    cl = slim.fully_connected(
                        features,
                        1,
                        activation_fn=None,
                        weights_initializer=initializer,
                        reuse=reuse,
                        scope='classification')
            return cl

        with tf.variable_scope('CycleGAN'):
            # Non-ideal to Ideal Mapping
            self.g_ideal = generator(
                self.inputs_non_ideal,
                reuse=False,
                scope='NonIdeal_to_Ideal')
            self.d_ideal_truth = discriminator(
                self.inputs_ideal,
                n_classes=2,
                reuse=False,
                scope='Ideal')
            self.d_ideal_generated = discriminator(
                self.g_ideal,
                n_classes=2,
                reuse=True,
                scope='Ideal')

            # Ideal to Non-ideal Mapping
            self.g_non_ideal = generator(
                self.inputs_ideal,
                reuse=False,
                scope='Ideal_to_NonIdeal')
            self.d_nonideal_truth = discriminator(
                self.inputs_non_ideal,
                n_classes=self.n_classes,
                reuse=False,
                scope='NonIdeal')
            self.d_nonideal_generated = discriminator(
                self.g_non_ideal,
                n_classes=self.n_classes,
                reuse=True,
                scope='NonIdeal')

            # Cycle consistency
            self.g_non_ideal_cycle = generator(
                self.g_ideal,
                reuse=True,
                scope='Ideal_to_NonIdeal')
            self.g_ideal_cycle = generator(
                self.g_non_ideal,
                reuse=True,
                scope='NonIdeal_to_Ideal')

            ideal_t = tf.random_uniform(
                [tf.shape(self.inputs_ideal)[0],1,1,1],
                minval=0.,maxval=1.)
            non_ideal_t = tf.random_uniform(
                [tf.shape(self.inputs_non_ideal)[0],1,1,1],
                minval=0.,maxval=1.)
            self.ideal_hat = tf.add(
                ideal_t * self.g_ideal,
                (1 - ideal_t) * self.inputs_ideal)
            self.nonideal_hat = tf.add(
                non_ideal_t * self.g_non_ideal,
                (1 - non_ideal_t) * self.inputs_non_ideal)
            self.d_ideal_hat = discriminator(
                self.ideal_hat,
                n_classes=2,
                reuse=True,
                scope='Ideal')
            self.d_nonideal_hat = discriminator(
                self.nonideal_hat,
                n_classes=self.n_classes,
                reuse=True,
                scope='NonIdeal')

    def get_loss(self):
        def random_like(tensor):
            rand = tf.random_uniform(tf.shape(tensor),maxval=1.)
            random_binary = tf.cast(rand > 0.5,tf.float32)
            return random_binary

        with tf.name_scope('Loss') and tf.variable_scope('Loss'):
            reg_losses = tf.reduce_mean(tf.losses.get_regularization_losses())

            if self.wasserstein == True:
                self.g_ideal_to_nonideal_loss = tf.reduce_mean(
                    - self.d_nonideal_generated)
                self.d_nonideal_loss_generated = tf.reduce_mean(
                    self.d_nonideal_generated)
                self.d_nonideal_loss_truth = - tf.reduce_mean(
                    self.d_nonideal_truth)
                self.d_nonideal_loss = tf.add(
                    self.d_nonideal_loss_generated,
                    self.d_nonideal_loss_truth
                )

                self.g_nonideal_to_ideal_loss = tf.reduce_mean(
                    - self.d_ideal_generated)
                self.d_ideal_loss_generated = tf.reduce_mean(
                    self.d_ideal_generated)
                self.d_ideal_loss_truth = - tf.reduce_mean(
                    self.d_ideal_truth)
                self.d_ideal_loss = tf.add(
                    self.d_ideal_loss_generated,
                    self.d_ideal_loss_truth
                )

            else:
                self.g_ideal_to_nonideal_loss = tf.reduce_mean(
                    tf.square(
                        tf.sigmoid(self.d_nonideal_generated) - 1))
                self.g_nonideal_to_ideal_loss = tf.reduce_mean(
                    tf.square(
                        tf.sigmoid(self.d_ideal_generated) - 1))
                self.d_ideal_loss = tf.add(
                    tf.reduce_mean(
                        tf.square(
                            tf.sigmoid(self.d_ideal_truth) - 1)),
                    tf.reduce_mean(
                        tf.square(
                            tf.sigmoid(self.d_ideal_generated)))
                )
                self.d_nonideal_loss = tf.add(
                    tf.reduce_mean(
                        tf.square(
                            tf.sigmoid(self.d_nonideal_truth) - 1)),
                    tf.reduce_mean(
                        tf.square(
                            tf.sigmoid(self.d_nonideal_generated)))
                )

            self.cycle_consistency_loss_nonideal = tf.reduce_mean(
                tf.abs(self.g_non_ideal_cycle - self.inputs_non_ideal))
            self.cycle_consistency_loss_ideal = tf.reduce_mean(
                tf.abs(self.g_ideal_cycle - self.inputs_ideal))

            self.g_ideal_to_nonideal_loss = tf.add(
                self.g_ideal_to_nonideal_loss,
                self.lambda_cycle * self.cycle_consistency_loss_nonideal)
            self.g_nonideal_to_ideal_loss = tf.add(
                self.g_nonideal_to_ideal_loss,
                self.lambda_cycle * self.cycle_consistency_loss_ideal)

def train(ideal_image_path=None,
          nonideal_image_path=None,
          extension='png',
          height=512,
          resize_height=512,
          width=512,
          resize_width=512,
          n_channels=3,
          n_classes=2,
          batch_size=4,
          beta_l2_regularization=0.005,
          save_summary_steps=1000,
          save_summary_folder='',
          save_checkpoint_steps=1000,
          save_checkpoint_folder='',
          log_every_n_steps=50,
          learning_rate=0.001,
          epochs=100,
          lambda_cycle=10,
          wasserstein=False,
          compound_loss=False):

    def clip_gradients(gradients):
        return [(tf.clip_by_norm(v[0],0.5),v[1]) for v in gradients]

    def l2_norm(tensor,axis=[1,2,3]):
        pre_out = tf.reduce_sum(tf.square(tensor),axis=axis)
        return tf.sqrt(pre_out)

    cycle_gan = CycleGAN(
        mode='train',
        ideal_image_path=ideal_image_path,
        nonideal_image_path=nonideal_image_path,
        extension=extension,
        height=height,
        resize_height=resize_height,
        width=width,
        resize_width=resize_width,
        n_classes=n_classes,
        n_channels=n_channels,
        batch_size=batch_size,
        beta_l2_regularization=beta_l2_regularization,
        lambda_cycle=lambda_cycle,
        wasserstein=wasserstein)

    try: os.makedirs(save_summary_folder)
    except: pass

    try: os.makedirs(save_checkpoint_folder)
    except: pass

    save_ckpt_path = os.path.join(
        save_checkpoint_folder, 'my-model.ckpt'
        )

    n_ideal_images = len(cycle_gan.ideal_image_path_list)
    n_nonideal_images = len(cycle_gan.nonideal_image_path_list)
    n_steps = np.minimum(n_ideal_images/batch_size * epochs,
                         n_nonideal_images/batch_size * epochs)
    tf.set_random_seed(0)
    np.random.seed(42)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.cond(
        global_step < n_steps/2,
        lambda: global_step,
        lambda: tf.cast(global_step * n_steps/(2*global_step),tf.int64))

    vars = {
        'Generator_NonIdeal_to_Ideal':[],
        'Generator_Ideal_to_NonIdeal':[],
        'Discriminator_Ideal':[],
        'Discriminator_NonIdeal':[]
    }

    clip_op = []
    all_weights = []

    for var in tf.trainable_variables():
        if 'weight' in var.name:
            all_weights.append(var)
        if 'Generator_NonIdeal_to_Ideal' in var.name:
            vars['Generator_NonIdeal_to_Ideal'].append(var)
        elif 'Generator_Ideal_to_NonIdeal' in var.name:
            vars['Generator_Ideal_to_NonIdeal'].append(var)
        elif 'Discriminator_Ideal' in var.name:
            vars['Discriminator_Ideal'].append(var)
            if 'weights' in var.name:
                clip_op.append(var.assign(tf.clip_by_value(var, -0.01, 0.01)))
        elif 'Discriminator_NonIdeal' in var.name:
            vars['Discriminator_NonIdeal'].append(var)
            if 'weights' in var.name:
                clip_op.append(var.assign(tf.clip_by_value(var, -0.01, 0.01)))

    gradients_ideal = tf.gradients(cycle_gan.d_ideal_hat,
                                   cycle_gan.ideal_hat)[0]+1e-16
    gradients_nonideal = tf.gradients(cycle_gan.d_nonideal_hat,
                                      cycle_gan.nonideal_hat)[0]+1e-16
    gp_ideal = tf.reduce_mean(
        tf.square(l2_norm(gradients_ideal,axis=[1,2,3]) - 1.0))

    gp_non_ideal = tf.reduce_mean(
        tf.square(l2_norm(gradients_nonideal,axis=[1,2,3]) - 1.0))

    loss = tf.add_n(
        [cycle_gan.d_ideal_loss,
         cycle_gan.d_nonideal_loss,
         cycle_gan.g_ideal_to_nonideal_loss,
         cycle_gan.g_nonideal_to_ideal_loss]
        )

    if compound_loss == False:
        g_opt_ideal = tf.train.AdamOptimizer(learning_rate,
                                             beta1=0.5,beta2=0.9)
        g_opt_nonideal = tf.train.AdamOptimizer(learning_rate,
                                                beta1=0.5,beta2=0.9)
        d_opt_ideal = tf.train.AdamOptimizer(learning_rate,
                                             beta1=0.5,beta2=0.9)
        d_opt_nonideal = tf.train.AdamOptimizer(learning_rate,
                                                beta1=0.5,beta2=0.9)

        g_train_op = tf.group(
            g_opt_ideal.minimize(
                cycle_gan.g_ideal_to_nonideal_loss,
                global_step=global_step,
                var_list=vars['Generator_Ideal_to_NonIdeal']),
            g_opt_nonideal.minimize(
                cycle_gan.g_nonideal_to_ideal_loss,
                global_step=global_step,
                var_list=vars['Generator_NonIdeal_to_Ideal'])
            )

        if wasserstein == True:
            d_train_nonideal_loss = tf.add(
                cycle_gan.d_nonideal_loss/2,gp_non_ideal)
            d_train_ideal_loss = tf.add(
                cycle_gan.d_ideal_loss/2,gp_ideal)

        else:
            d_train_nonideal_loss = cycle_gan.d_nonideal_loss / 2
            d_train_ideal_loss = cycle_gan.d_ideal_loss / 2
        d_train_op_nonideal = d_opt_nonideal.minimize(
            d_train_nonideal_loss,
            global_step=global_step,
            var_list=vars['Discriminator_NonIdeal'])
        d_train_op_ideal = d_opt_ideal.minimize(
            d_train_ideal_loss,
            global_step=global_step,
            var_list=vars['Discriminator_Ideal'])
        d_train_op = tf.group(
            d_train_op_nonideal,
            d_train_op_ideal
            )
    else:
        opt = tf.train.AdamOptimizer(learning_rate,beta1=0.0,beta2=0.9)
        train_op = opt.minimize(loss)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    summaries.add(
        tf.summary.image(
            '1_nonideal_inputs',
            cycle_gan.inputs_non_ideal,max_outputs = 4))
    summaries.add(
        tf.summary.image(
            '2_nonideal_to_ideal',
            cycle_gan.g_ideal,max_outputs = 4))
    summaries.add(
        tf.summary.image(
            '3_nonideal_to_ideal_to_nonideal',
            cycle_gan.g_non_ideal_cycle,max_outputs = 4))

    summaries.add(
        tf.summary.image(
            '4_ideal_inputs',
            cycle_gan.inputs_ideal,max_outputs = 4))
    summaries.add(
        tf.summary.image(
            '5_ideal_to_nonideal',
            cycle_gan.g_non_ideal,max_outputs = 4))
    summaries.add(
        tf.summary.image(
            '6_ideal_to_nonideal_to_ideal',
            cycle_gan.g_ideal_cycle,max_outputs = 4))

    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    summaries.add(tf.summary.scalar('g_ideal_to_nonideal_loss',
                                    cycle_gan.g_ideal_to_nonideal_loss))
    summaries.add(tf.summary.scalar('d_ideal_loss',
                                    cycle_gan.d_ideal_loss))
    summaries.add(tf.summary.scalar('g_nonideal_to_ideal_loss',
                                    cycle_gan.g_nonideal_to_ideal_loss))
    summaries.add(tf.summary.scalar('d_nonideal_loss',
                                    cycle_gan.d_nonideal_loss))
    summaries.add(tf.summary.scalar('cycle_consistency_loss',
                                    cycle_gan.cycle_consistency_loss_ideal))
    summaries.add(tf.summary.scalar('cycle_consistency_loss',
                                    cycle_gan.cycle_consistency_loss_nonideal))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(save_summary_folder,
                                       sess.graph)
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.local_variables_initializer())

        all_losses = {
            'G_IdealToNonIdeal':cycle_gan.g_ideal_to_nonideal_loss,
            'D_Ideal':cycle_gan.d_ideal_loss,
            'G_NonIdealToIdeal':cycle_gan.g_nonideal_to_ideal_loss,
            'D_NonIdeal':cycle_gan.d_nonideal_loss,
            'GP_Ideal':gp_ideal,
            'GP_NonIdeal':gp_non_ideal,
            'Cycle_NonIdeal':cycle_gan.cycle_consistency_loss_nonideal,
            'Cycle_Ideal':cycle_gan.cycle_consistency_loss_ideal,
            'FullLoss':loss}

        for i in range(1,int(n_steps) + 1):
            if compound_loss == False:
                if wasserstein == True:
                    for _ in range(N_ITER_D):
                        sess.run(d_train_op_nonideal)
                        sess.run(d_train_op_ideal)
                else:
                    sess.run(d_train_op_nonideal)
                    sess.run(d_train_op_ideal)
                for _ in range(N_ITER_G):
                    sess.run(g_train_op)
                l = sess.run(all_losses)
            else:
                sess.run(train_op)
                l = sess.run(all_losses)
            if i % log_every_n_steps == 0 or i == 1:
                print(LOG.format(
                          i,
                          int(n_steps),
                          l['G_IdealToNonIdeal'],
                          l['D_NonIdeal'],
                          l['Cycle_NonIdeal'],
                          l['GP_NonIdeal'],
                          l['G_NonIdealToIdeal'],
                          l['D_Ideal'],
                          l['Cycle_Ideal'],
                          l['GP_Ideal'],
                          l['FullLoss']
                      ))
            if i % save_summary_steps == 0 or i == 1:
                summary = sess.run(summary_op)
                writer.add_summary(summary,i)
                print("""Summary saved in: {0} (step {1})""".format(
                    save_summary_folder,i)
                    )

            if i % save_checkpoint_steps == 0 or i == 1:
                saver.save(sess, save_ckpt_path,global_step=i)
                print("""Checkpoint saved in: {0} (step {1})""".format(
                    save_checkpoint_folder,i)
                    )
def test():
    pass
def predict():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='cycle_gan.py',
        description='Variational autoencoder.'
    )

    parser.add_argument('--mode',dest='mode',
                        action='store',type=str,
                        choices=['train','test','predict'],
                        default='train',
                        help='Defines the mode.')
    parser.add_argument('--ideal_image_path',dest='ideal_image_path',
                        action='store',type=str,
                        default=None,
                        help='Path to ideal images.')
    parser.add_argument('--nonideal_image_path',dest='nonideal_image_path',
                        action='store',type=str,
                        default=None,
                        help='Path to non-ideal images.')
    parser.add_argument('--extension',dest='extension',
                        action='store',type=str,
                        default='png',
                        help='Extension for the images.')
    parser.add_argument('--height',dest='height',
                        action='store',type=int,
                        default=512,
                        help='Height of the image files.')
    parser.add_argument('--width',dest='width',
                        action='store',type=int,
                        default=512,
                        help='Width of the image files.')
    parser.add_argument('--resize_height',dest='resize_height',
                        action='store',type=int,
                        default=512,
                        help='Final height for the images.')
    parser.add_argument('--resize_width',dest='resize_width',
                        action='store',type=int,
                        default=512,
                        help='Final width for the images.')
    parser.add_argument('--n_channels',dest='n_channels',
                        action='store',type=int,
                        default=3,
                        help='Number of channels in the image.')
    parser.add_argument('--n_classes',dest='n_classes',
                        action='store',type=int,
                        default=3,
                        help='Number of classes in the non-ideal images.')
    parser.add_argument('--batch_size',dest='batch_size',
                        action='store',type=int,
                        default=4,
                        help='Batch size for training.')
    parser.add_argument('--beta_l2_regularization',
                        dest='beta_l2_regularization',
                        action='store',type=float,
                        default=0.005,
                        help='Beta value for L2 regularization.')
    parser.add_argument('--save_summary_steps',dest='save_summary_steps',
                        action='store',type=int,
                        default=100,
                        help='Save summary every n steps.')
    parser.add_argument('--save_summary_folder',dest='save_summary_folder',
                        action=ToDirectory,type=str,
                        default='/tmp/checkpoint',
                        help='Folder where summary will be stored.')
    parser.add_argument('--save_checkpoint_steps',dest='save_checkpoint_steps',
                        action='store',type=int,
                        default=100,
                        help='Save checkpoint every n steps.')
    parser.add_argument('--save_checkpoint_folder',dest='save_checkpoint_folder',
                        action=ToDirectory,type=str,
                        default='/tmp/checkpoint',
                        help='Folder where checkpoints will be stored.')
    parser.add_argument('--log_every_n_steps',dest='log_every_n_steps',
                        action='store',type=int,
                        default=5,
                        help='Print log every n steps.')
    parser.add_argument('--learning_rate',dest='learning_rate',
                        action='store',type=float,
                        default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--epochs',dest='epochs',
                        action='store',type=int,
                        default=100,
                        help='Number of epochs.')
    parser.add_argument('--lambda_cycle',dest='lambda_cycle',
                        action='store',type=float,
                        default=10,
                        help='Factor for the cycle consistency loss.')
    parser.add_argument('--wasserstein',dest='wasserstein',
                        action='store_true',
                        default=False,
                        help='Flag to use WGAN-GP loss.')
    parser.add_argument('--compound_loss',dest='compound_loss',
                        action='store_true',
                        default=False,
                        help='Flag to train all losses with one optimizer.')

    args = parser.parse_args()

    if args.mode == 'train':
        print(args.ideal_image_path)
        train(ideal_image_path=args.ideal_image_path,
              nonideal_image_path=args.nonideal_image_path,
              extension=args.extension,
              height=args.height,
              width=args.width,
              resize_height=args.resize_height,
              resize_width=args.resize_width,
              n_channels=args.n_channels,
              n_classes=args.n_classes,
              batch_size=args.batch_size,
              beta_l2_regularization=args.beta_l2_regularization,
              save_summary_steps=args.save_summary_steps,
              save_summary_folder=args.save_summary_folder,
              save_checkpoint_steps=args.save_checkpoint_steps,
              save_checkpoint_folder=args.save_checkpoint_folder,
              log_every_n_steps=args.log_every_n_steps,
              learning_rate=args.learning_rate,
              epochs=args.epochs,
              lambda_cycle=args.lambda_cycle,
              wasserstein=args.wasserstein,
              compound_loss=args.compound_loss)
