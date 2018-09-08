"""
This script features a class implementing a region proposal network (RPN)
mostly based on the different flavours of You Only Look Once (YOLO) [1,2,3] and
Regions with Convolutional Neural Networks (RCNN) [4,5,6,7] and Feature Pyramid
Networks (FPN) [8]. It builds on the concept of FPN for multi-level predictions
using anchors. This was used primarily to assign bounding boxes to WBC in whole
blood slides, but can easily be generalized to any other bounding box
prediction task, given appropriate input.

[1] https://arxiv.org/abs/1506.02640
[2] https://arxiv.org/abs/1506.02640
[3] https://pjreddie.com/media/files/papers/YOLOv3.pdf
[4] https://arxiv.org/abs/1311.2524
[5] https://arxiv.org/abs/1504.08083
[6] https://arxiv.org/abs/1506.01497
[7] https://arxiv.org/abs/1703.06870
[8] https://arxiv.org/abs/1612.03144
"""

import os
import time
import argparse
from itertools import product
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
from math import ceil,floor
import mobilenet_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

slim = tf.contrib.slim

_NO_BB_CSV_ERROR = "--bounding_box_csv should be provided for {0!s} mode."

_LOG_TRAIN = 'Step {0:d}: \n\ttime = {1:f}s \n\tLoss = {2:f} '
_LOG_TRAIN += '\n\t\tLoss(class) = {3:f} '
_LOG_TRAIN += '\n\t\tLoss(Huber) = {4:f} \n\tMSE(t) = {5:.2E} '
_LOG_TRAIN += '\n\tAUC(P) = {6:f}'
_LOG_TRAIN_FINAL = 'Finished training. {0:d} steps, '
_LOG_TRAIN_FINAL += 'with {1:d} images in the training set.'
_LOG_TRAIN_FINAL += 'Average time/image = {2:f}s, final loss = {3:f}'

_LOG_TEST = 'Time = {0:f}/image with {1:d} images; AUC(p) = {2:f}; '
_LOG_TEST += 'Sen(p) = {3:f}; Spe(p) = {4:f}; MSE = {5:f}'
_LOG_TEST_FINAL = 'Averages: ' + _LOG_TEST
_SUMMARY_TRAIN = 'Step {0:d}: summary saved in {1!s}'
_CHECKPOINT_TRAIN = 'Step {0:d}: checkpoint saved in {1!s}'

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """
    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

class RPN():
    """
    The main working class for the RPN. It takes several different inputs that
    enable training (including realtime image augmentation), testing and
    prediction.

    The arguments are hierarchically organized under subtitles (signed with a
    leading hash). After each definition, the default argument (if any) is made
    clear between square brackets:
    # Data
        * data_directory - directory containing the image files
        * extension - the extension to look for in the directory ['png']
        * bounding_box_csv - a .csv file containing the bounding box
        information. This should be coded as one line *per* image and one
        column containing the image name and another column containing the
        bounding boxes, coded as x_center:y_center:height:width (with : as
        coord_sep) and each bounding box should be separated by bb_sep [None]
        * bb_sep - the character used to separate each bounding box on the
        bounding_box_csv [';']
        * coord_sep - the character used to separate each bounding box
        coordinate [':']
        * input_width - the input_width of the images [1024]
        * input_height - the input_height of the images [1024]
        * batch_size - the size of each mini-batch [32]
        * truth_only - a flag to exclude any images with no bounding boxes.
        truth_only = True is the only supported mode since having images with
        no bounding boxes will lead to a tensor of [n,n,0] dimensions. Apart
        from this, it is not extremely useful to use images without bounding
        boxes, so this is likely to stay this way [False]
        * mode - the mode for the algorithm. Can either be 'train', 'test' or
        'predict' ['train']
    # Network
        * confidence_threshold - the value used to consider an anchor as positive
        [0.7]
        * checkpoint_path - path to a checkpoint to be restored. This can be done
        in any mode [None]
    # Preprocessing
        * smart_preprocessing - a flag to precalculate some pixel-wise image
        features, namely HSV values and Sobel gradients [False]
    # Train
        * learning_rate - learning rate for the training [0.001]
        * weight_decay - multiplication factor for the L2 term in the loss
        function [0.0001]
        * class_factor - multiplication factor for the classification term in the
        loss function [1]
        * reg_factor - multiplication factor for the regression term in the loss
        function [1]
        * training_nms - flag to suppress all non-maximum anchors in each cell
        * save_summary_folder - folder where the summary will be stored (this
        folder will be created when RPN() is called) ['summary']
        * save_checkpoint_folder - folder where the checkpoints will be stored
        (this folder will be created when RPN() is called) ['checkpoint']
        * save_summary_steps - how often should the summary be updated [500]
        * save_checkpoint_steps - how often should checkpoints be saved [500]
        * number_of_steps - maximum number of steps [1000]
        * epochs - maximum number of epochs (overrides number of steps) [None]
        * log_every_n_steps - how often should the code produce an output.
        This contains information on the total loss, the loss terms for
        classification (loss(class)) and regression (loss(Huber)), the MSE for
        the bounding box dimensions and the AUC, Sensitivity and Specificity
        for the bounding box classification [50]
    # Prediction
        save_predictions_path - Where should the predictions be stored
        ['prediction.csv']

    NOTE: the predict() method is not fully implemented (yet)
    """

    def __init__(self,
                 #Data
                 data_directory,
                 extension = 'png',
                 bounding_box_csv = None,
                 bb_sep = ';',
                 coord_sep = ':',
                 input_width = 1024,
                 input_height = 1024,
                 batch_size = 32,
                 truth_only = False,
                 mode = 'train',
                 #Network
                 confidence_threshold = 0.7,
                 checkpoint_path = None,
                 #Preprocessing
                 smart_preprocessing = False,
                 #Train
                 learning_rate = 0.001,
                 weight_decay = 0.0001,
                 class_factor = 1,
                 reg_factor = 1,
                 training_nms = False,
                 save_summary_folder = 'summary',
                 save_checkpoint_folder = 'checkpoint',
                 save_summary_steps = 500,
                 save_checkpoint_steps = 500,
                 number_of_steps = 1000,
                 epochs = None,
                 log_every_n_steps = 50,
                 #Prediction
                 save_predictions_path = 'prediction.csv'
                 ):

        #Data
        self.config_arguments = {}
        self.data_directory = data_directory
        self.extension = extension
        self.bounding_box_csv = bounding_box_csv
        self.bb_sep = bb_sep
        self.coord_sep = coord_sep
        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = batch_size
        self.truth_only = truth_only
        self.mode = mode
        #Network
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        #Preprocessing
        self.smart_preprocessing = smart_preprocessing
        #Train
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_factor = class_factor
        self.reg_factor = reg_factor
        self.training_nms = training_nms
        self.save_summary_folder = save_summary_folder
        self.save_summary_steps = save_summary_steps
        self.save_checkpoint_folder = save_checkpoint_folder
        self.save_checkpoint_steps = save_checkpoint_steps
        self.number_of_steps = number_of_steps
        self.epochs = epochs
        self.log_every_n_steps = log_every_n_steps
        #Prediction
        self.save_predictions_path = save_predictions_path

        self.image_path_list = glob(
            os.path.join(data_directory,'*' + self.extension)
            )
        self.image_path_dict = {
            p.split(os.sep)[-1]:p for p in self.image_path_list
            }
        self.save_ckpt_path = os.path.join(
            self.save_checkpoint_folder,
            'my-rpn.ckpt'
            )

        if self.mode == 'train':
            self.is_training = True
        else:
            self.is_training = False
        if self.mode == 'train' or self.mode == 'test':
            if bounding_box_csv == None:
                raise Exception(_NO_BB_CSV_ERROR.format(self.mode))

        self.levels = [
            'P5',
            'P4',
            'P3'
            ]
        self.factors = {
            'P5':32,
            'P4':16,
            'P3':8
            }
        self.anchor_sizes = {
            'P5':[2,3],
            'P4':[2,3],
            'P3':[5,7]
            }
        self.level_sizes = {
            'P5':self.input_height / 32,
            'P4':self.input_height / 16,
            'P3':self.input_height / 8
            }
        self.no_anchors = {
            'P5':4,
            'P4':4,
            'P3':4
            }
        self.no_anchors_list = [4,4,4]

        self.make_dirs()
        self.data_generator()
        self.shape = self.inputs.get_shape().as_list()
        if self.epochs != None:
            self.number_of_steps = ceil(self.no_images/self.batch_size)
            print("No of iterations per epoch:",
                  self.number_of_steps)
            self.number_of_steps *= self.epochs
        self.get_feature_map()
        self.get_feature_pyramid()
        self.get_object_vectors()
        self.draw_boxes()
        self.get_loss_metrics()
        self.pre_flight_operations()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            all_var = [np.prod(v.get_shape()) for v in tf.trainable_variables()]
            print('VARIABLES',np.sum(all_var))

        if mode == 'train':
            #pass
            self.train()
        if mode =='test':
            self.number_of_steps = floor(self.no_images / self.batch_size) + 1
            self.test()

    def make_dirs(self):
        """Creates the save_checkpoint_folder and save_summary_folder
        directories.
        """

        def make_dir(dir):
            try: os.makedirs(dir)
            except: pass

        make_dir(self.save_checkpoint_folder)
        make_dir(self.save_summary_folder)

    def sess_debugger(self,wtv):
        """
        Convenience function used for debugging. It creates a self contained
        session and runs whatever its input is.
        """

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            return sess.run(wtv)
            coord.request_stop()
            coord.join(threads)

    def data_generator(self):
        """
        This creates an input pipeline.

        Information on each file and each bounding box is retrieved from
        self.bounding_box_csv, which should structured as a comma-separated
        values file, with n columns, one for the filename and one for each
        bounding box, which will be used to build a dict with a list of b.b.
        per file name:
            {file_name:[x1:y1:w1:h1,x2:y2:w2:h2,...]}
        """

        def anchor_to_mask(factor,
                           mask_height,
                           mask_width,
                           anchor_height,
                           anchor_width,
                           truth_string):

            def true_div_floor(a,b):
                a = tf.cast(a,tf.float32)
                b = tf.cast(b,tf.float32)
                return tf.floor(tf.truediv(a,b))

            def sparse_resizer(tensor):
                sparse_tensor = tf.contrib.layers.dense_to_sparse(tensor)
                new_indices = true_div_floor(sparse_tensor.indices,factor)
                new_indices = tf.cast(new_indices,tf.int32)

                new_shape = true_div_floor(sparse_tensor.dense_shape,factor)
                new_shape = tf.floor(new_shape)
                new_shape = tf.cast(new_shape,tf.int32)

                #This gets rid of duplicate entries by keeping the
                #maximum-valued bounding-box. Found this on stackoverflow (duh)
                #here: https://stackoverflow.com/questions/38233821/merge-duplicate-indices-in-a-sparse-tensor
                #it is kind of a clunky solution for a problem that tbh should
                #be solved
                ctt = new_shape[0]
                linearized = tf.squeeze(
                    tf.matmul(new_indices, [[ctt], [1]]))
                top = tf.shape(linearized)[0]

                linearized,order = tf.nn.top_k(linearized,
                                               k = top,sorted = True)
                values = tf.gather(sparse_tensor.values,order)

                y, idx = tf.unique(tf.squeeze(linearized))
                values = tf.segment_mean(values,idx)
                y = tf.expand_dims(y, 1)
                new_indices = tf.concat([y // ctt, y % ctt], axis=1)

                dense_tensor = tf.sparse_to_dense(
                    new_indices,
                    new_shape,
                    values,
                    validate_indices = False
                )

                return dense_tensor

            def unravel_argmax(argmax,shape):
                output_list = []
                output_list.append(argmax // (shape[2]))
                output_list.append(argmax % (shape[2]))
                return tf.stack(output_list)

            def coords_to_mask(truth):
                truth = tf.string_split(truth,
                                        delimiter = self.coord_sep)
                truth = tf.sparse_to_dense(truth.indices,
                                           truth.dense_shape,
                                           truth.values,
                                           '')
                truth = tf.string_to_number(truth)

                truth_x,truth_y,truth_h,truth_w = tf.split(truth[0],4)
                x3 = truth_x
                y3 = truth_y
                x4 = x3 + truth_h
                y4 = y3 + truth_w
                box_x1 = tf.maximum(x1,x3)
                box_x2 = tf.minimum(x2,x4)
                box_h = tf.maximum(0.,box_x2 - box_x1)
                box_y1 = tf.maximum(y1,y3)
                box_y2 = tf.minimum(y2,y4)
                box_w = tf.maximum(0.,box_y2 - box_y1)
                intersection_area = box_h * box_w
                anchor_area = anchor_tensor[:,:,2] * anchor_tensor[:,:,3]
                truth_area = truth_h * truth_w
                union_area = anchor_area + truth_area - intersection_area
                iou = tf.where(union_area > 0,
                               tf.truediv(intersection_area,union_area),
                               tf.zeros((true_mask_height,true_mask_width)))
                t_x = (anchor_tensor[:,:,0] - truth_x)/truth_h
                t_y = (anchor_tensor[:,:,1] - truth_y)/truth_w
                t_h = tf.log(anchor_tensor[:,:,2]/truth_h + 1e-7)
                t_w = tf.log(anchor_tensor[:,:,3]/truth_w + 1e-7)

                binary_iou = tf.cast(iou > self.confidence_threshold,
                                     tf.float32)
                iou,argmax = tf.nn.max_pool_with_argmax(
                    tf.expand_dims(
                        tf.expand_dims(iou * binary_iou,0),3),
                    ksize = [1,factor,factor,1],
                    strides = [1,factor,factor,1],
                    padding = 'VALID')
                iou = tf.squeeze(iou)
                argmax = tf.cast(argmax,tf.int32)
                argmax_x,argmax_y = (argmax // self.input_height,
                                     argmax % self.input_height)
                argmax_fix_x = tf.tile(
                    tf.expand_dims(tf.range(mask_height),1),
                    [1,mask_width]
                    ) * factor
                argmax_fix_y = tf.tile(
                    tf.expand_dims(tf.range(mask_width),0),
                    [mask_height,1]
                    ) * factor

                argmax_x = tf.expand_dims(
                    tf.squeeze(argmax_x) + argmax_fix_x,
                    0)
                argmax_y = tf.expand_dims(
                    tf.squeeze(argmax_y) + argmax_fix_y,
                    0)

                argmax = tf.concat(
                    [tf.layers.flatten(argmax_x),
                     tf.layers.flatten(argmax_y)],
                    axis = 0
                )
                argmax = tf.transpose(argmax,[1,0])

                t_x = tf.gather_nd(t_x,argmax)
                t_y = tf.gather_nd(t_y,argmax)
                t_h = tf.gather_nd(t_h,argmax)
                t_w = tf.gather_nd(t_w,argmax)
                adj_x = tf.gather_nd(anchor_tensor[:,:,0],argmax)
                adj_y = tf.gather_nd(anchor_tensor[:,:,1],argmax)

                t_x = tf.reshape(t_x,[mask_height,mask_width])
                t_y = tf.reshape(t_y,[mask_height,mask_width])
                t_h = tf.reshape(t_h,[mask_height,mask_width])
                t_w = tf.reshape(t_w,[mask_height,mask_width])
                adj_x = tf.reshape(argmax_x,[mask_height,mask_width])
                adj_y = tf.reshape(argmax_y,[mask_height,mask_width])

                adj_x = tf.truediv(tf.cast(adj_x,tf.float32),
                                   tf.cast(factor,tf.float32))
                adj_x = adj_x - tf.floor(adj_x)
                adj_y = tf.truediv(tf.cast(adj_y,tf.float32),
                                   tf.cast(factor,tf.float32))
                adj_y = (adj_y - tf.floor(adj_y))

                output = [iou,t_x,t_y,t_h,t_w,adj_x,adj_y]
                return output

            factor = int(factor)
            mask_height = int(mask_height)
            mask_width = int(mask_width)
            anchor_height = int(anchor_height)
            anchor_width = int(anchor_width)
            true_mask_height = mask_height * factor
            true_mask_width = mask_width * factor
            true_anchor_height = anchor_height * factor
            true_anchor_width = anchor_width * factor

            fix_x = tf.tile(
                tf.expand_dims(tf.range(0,true_mask_height),1),
                [1,true_mask_width]
                )
            fix_x = tf.cast(fix_x,dtype = tf.float32)
            fix_y = tf.tile(
                tf.expand_dims(tf.range(0,true_mask_width),0),
                [true_mask_height,1]
                )
            fix_y = tf.cast(fix_y,dtype = tf.float32)
            anchor_tensor = tf.stack(
                (fix_x,
                 fix_y,
                 tf.ones((true_mask_height,true_mask_width)) * true_anchor_height,
                 tf.ones((true_mask_height,true_mask_width)) * true_anchor_width),
                axis = 2
            )

            x1 = tf.maximum(0.,anchor_tensor[:,:,0] - anchor_tensor[:,:,2]/2)
            y1 = tf.maximum(0.,anchor_tensor[:,:,1] - anchor_tensor[:,:,3]/2)
            x2 = x1 + anchor_tensor[:,:,2]
            y2 = y1 + anchor_tensor[:,:,3]

            truth_string = tf.string_split(tf.reshape(truth_string,[1]),
                                           self.bb_sep)
            truth_string = tf.sparse_to_dense(truth_string.indices,
                                              truth_string.dense_shape,
                                              truth_string.values,
                                              '')
            truth_string = tf.transpose(truth_string,[1,0])

            iou,t_x,t_y,t_h,t_w,adj_x,adj_y = tf.map_fn(
                coords_to_mask,truth_string,
                dtype = [tf.float32 for i in range(7)])

            #These next few steps perform non-maximum suppression
            iou_mask = tf.transpose(iou,[1,2,0])
            argmax = tf.argmax(iou_mask,2)

            xy_ind = tf.transpose(
                np.mgrid[:mask_height, :mask_width], [1,2,0])
            xy_ind = tf.squeeze(xy_ind)
            gather_ind = tf.concat([xy_ind, argmax[..., None]], axis=-1)

            iou_mask = tf.gather_nd(iou_mask, gather_ind)
            t_x = tf.gather_nd(tf.transpose(t_x,[1,2,0]),
                               gather_ind)
            t_y = tf.gather_nd(tf.transpose(t_y,[1,2,0]),
                               gather_ind)
            t_h = tf.gather_nd(tf.transpose(t_h,[1,2,0]),
                               gather_ind)
            t_w = tf.gather_nd(tf.transpose(t_w,[1,2,0]),
                               gather_ind)
            adj_x = tf.gather_nd(tf.transpose(adj_x,[1,2,0]),
                                 gather_ind)
            adj_y = tf.gather_nd(tf.transpose(adj_y,[1,2,0]),
                                 gather_ind)

            x = adj_x
            y = adj_y
            h = tf.ones((mask_height,mask_width)) * anchor_height
            w = tf.ones((mask_height,mask_width)) * anchor_width

            t_mask = tf.stack((t_x,t_y,t_h,t_w),2)
            anchor_tensor = tf.stack((x,y,h,w),2)

            iou_mask = tf.expand_dims(iou_mask,2)
            output = tf.concat((iou_mask,anchor_tensor,t_mask),axis = 2)

            return output

        def get_dict_elem(str_array):
            key = str_array[0]
            key = key.decode('ascii')
            return self.bounding_box_dict[key]

        def paired_data_augmentation(image_masks_tuple):

            def distort_colors_0(image):
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                return image

            def distort_colors_1(image):
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.05)
                return image

            def distort_colors_2(image):
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                return image

            def distort_colors_3(image):
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                return image

            def distort_colors(image,color_ordering):
                image = tf.cond(tf.equal(color_ordering,0),
                                lambda: distort_colors_0(image),
                                lambda: image)
                image = tf.cond(tf.equal(color_ordering,1),
                                lambda: distort_colors_1(image),
                                lambda: image)
                image = tf.cond(tf.equal(color_ordering,2),
                                lambda: distort_colors_2(image),
                                lambda: image)
                image = tf.cond(tf.equal(color_ordering,3),
                                lambda: distort_colors_3(image),
                                lambda: image)
                return image

            def set_shape(images):
                images[0].set_shape([self.input_height,self.input_width,3])
                for m,level in zip(images[1:],self.levels):
                    m.set_shape([
                        self.level_sizes[level],
                        self.level_sizes[level],
                        9 * self.no_anchors[level]
                    ])

            def flip_mask(mask,mode = 'lr'):
                if mode == 'ud':
                    split_mask = tf.split(mask,9,axis = 2)
                    split_mask = [tf.squeeze(m) for m in split_mask]
                    mask = tf.stack(
                        [split_mask[0],
                         1. - split_mask[1],
                         *split_mask[2:5],
                        split_mask[5] + (2 * split_mask[1] - 1)/split_mask[3],
                        *split_mask[6:]],
                        axis = 2
                    )
                elif mode == 'lr':
                    split_mask = tf.split(mask,9,axis = 2)
                    split_mask = [tf.squeeze(m) for m in split_mask]
                    mask = tf.stack(
                        [*split_mask[0:2],
                         1. - split_mask[2],
                         *split_mask[3:6],
                        split_mask[6] + (2 * split_mask[2] - 1)/split_mask[4],
                        *split_mask[7:]],
                        axis = 2
                    )

                return mask

            def flipper_lr(image_masks):
                #image_masks = [
                #    tf.reverse(m,[0]) for m in image_masks]
                image_masks = [
                    tf.py_func(lambda x: np.flip(x,0),[m],
                               tf.float32) for m in image_masks]
                image = image_masks[0]
                masks = image_masks[1:]
                masks = [tf.split(m,n,axis = 2) for m,n in zip(
                    masks,self.no_anchors_list)]
                new_masks = []
                for mask in masks:
                    mask = [flip_mask(m,'lr') for m in mask]
                    mask = tf.concat(mask,axis = 2)
                    new_masks.append(mask)
                return [image,*new_masks]

            def flipper_ud(image_masks):
                #image_masks = [
                #    tf.reverse(m,[1]) for m in image_masks]
                image_masks = [
                    tf.py_func(lambda x: np.flip(x,1),[m],
                               tf.float32) for m in image_masks]
                image = image_masks[0]
                masks = image_masks[1:]
                masks = [tf.split(m,n,axis = 2) for m,n in zip(
                    masks,self.no_anchors_list)]
                new_masks = []
                for mask in masks:
                    mask = [flip_mask(m,'ud') for m in mask]
                    mask = tf.concat(mask,axis = 2)
                    new_masks.append(mask)
                return [image,*new_masks]

            image = image_masks_tuple[0]
            masks = image_masks_tuple[1:]
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            color_ordering = tf.random_uniform([],0,4,tf.int32)
            image = distort_colors(image,color_ordering)

            flip_lr = tf.cast(
                tf.random_uniform([],0,2,tf.int32),
                tf.bool)
            flip_ud = tf.cast(
                tf.random_uniform([],0,2,tf.int32),
                tf.bool)

            tmp = [image,*masks]
            """
            tmp = tf.cond(
                flip_lr,
                lambda: flipper_lr(tmp),
                lambda: tmp,
                strict = True)
            set_shape(tmp)
            tmp = tf.cond(
                flip_ud,
                lambda: flipper_ud(tmp),
                lambda: tmp,
                strict = True)
            """
            set_shape(tmp)
            return tmp

        def anchors_to_mask_nms(factor,
                                curr_size,
                                anchor_sizes,
                                curr_bb,
                                permute,
                                name):
            anchor_list = []
            if permute == True:
                anchors = list(product(anchor_sizes,repeat = 2))
            else:
                anchors = [(i,i) for i in anchor_sizes]
            no_anchors = len(anchors)
            with tf.name_scope(name):
                for a,b in anchors:
                    anchor_ten = anchor_to_mask(
                        factor,
                        curr_size,
                        curr_size,
                        a,b,
                        curr_bb
                        )
                    anchor_list.append(anchor_ten)

                if self.training_nms == True:
                    anchor_tensor = tf.stack(anchor_list,axis = 3)
                    iou_tensor_max = tf.reduce_max(anchor_tensor[:,:,0,:],
                                                   axis = -1)
                    iou_mask = tf.greater_equal(
                        anchor_tensor[:,:,0,:],
                        tf.stack(
                            [iou_tensor_max for i in range(len(anchors))],
                            axis = -1)
                        )
                    iou_mask = tf.cast(iou_mask,tf.float32)
                    iou_mask = tf.split(iou_mask,len(anchors),axis = -1)
                    output = [tf.concat(
                        [iou_mask[i]*tf.expand_dims(anchor_list[i][:,:,0],-1),
                         anchor_list[i][:,:,1:]],
                        axis = -1) for i in range(no_anchors)
                     ]
                    output = tf.concat(
                        output,
                        axis = -1
                    )
                else:
                    output = tf.concat(anchor_list,axis = 2)

            return output

        def expand_truth(tensor,n):
            """
            Takes a tensor of concat truth tensors (c,x,y,h,w,tx,ty,th,tw) and
            creates an additional dimension to stack them.
            """
            stack_list = []
            for i in range(n):
                a = i * 9
                b = (i + 1) * 9
                stack_list.append(tensor[:,:,a:b])
            return tf.stack(stack_list,-1)

        def sobel_edges(image_tensor):
            gray_image_tensor = tf.map_fn(
                lambda x: tf.reduce_mean(x,axis = 2),
                image_tensor
            )
            sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  tf.float32)
            sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

            filtered_x = tf.nn.conv2d(gray_image_tensor, sobel_x_filter,
                                      strides=[1, 1, 1, 1], padding='SAME')
            filtered_y = tf.nn.conv2d(gray_image_tensor, sobel_y_filter,
                                      strides=[1, 1, 1, 1], padding='SAME')

            return tf.sqrt(tf.square(filtered_x) + tf.square(filtered_y))

        def rgb2hsv(image_tensor):
            return tf.map_fn(lambda x: tf.image.rgb_to_hsv(x),
                             image_tensor)

        with open(self.bounding_box_csv) as o:
            lines = o.readlines()
        lines = [line.strip().split(self.bb_sep) for line in lines]

        tmp = []
        for line in lines:
            tmp_bb = []
            bb = line[1:]
            for b in bb:
                _,_,h,w = b.split(self.coord_sep)
                if int(h) > 10 and int(w) > 10:
                    tmp_bb.append(b)
            tmp.append([line[0]] + tmp_bb)
        lines = tmp

        if self.truth_only == True:
            tmp_lines = [line[0] for line in lines if len(line) > 1]
            tmp = {}
            for key in self.image_path_dict:
                key = key.strip()
                if key in tmp_lines:
                    tmp[key] = self.image_path_dict[key]
                    print(key)
            self.image_path_dict = tmp
            self.image_path_list = list(self.image_path_dict.values())

        self.image_path_dict = tmp
        self.no_images = len(self.image_path_list)
        self.bounding_box_dict = {
            x[0]:[j.strip() for j in x[1:]] for x in lines
            }

        self.bounding_box_dict_tensor = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                tf.convert_to_tensor(
                    [*self.bounding_box_dict.keys()]
                    ),
                tf.convert_to_tensor(
                    [';'.join(v) for v in self.bounding_box_dict.values()]
                    ),
                key_dtype = tf.string,
                value_dtype = tf.string

            ),
            default_value = ''
        )

        image_path_tensor = tf.convert_to_tensor(self.image_path_list,
                                                 dtype=tf.string)
        train_input_queue = tf.train.slice_input_producer([image_path_tensor],
                                                          shuffle=False)[0]

        file_content = tf.read_file(train_input_queue)
        train_image = tf.image.decode_image(file_content, channels = 3)

        curr_key = tf.string_split(
            tf.reshape(train_input_queue,[1]),
            os.sep)
        self.curr_key = tf.sparse_tensor_to_dense(curr_key,
                                                  default_value = '')[-1][-1]
        self.curr_bb = self.bounding_box_dict_tensor.lookup(self.curr_key)

        tmp_levels = []

        train_image.set_shape([self.input_height,
                               self.input_width,3])
        train_input_queue.set_shape([])
        orig_image = tf.cast(train_image,tf.float32)

        for level in self.levels:
            factor = self.factors[level]
            anchor_sizes = self.anchor_sizes[level]
            level_size = self.level_sizes[level]
            no_anchors = self.no_anchors[level]
            print(
                factor,
                anchor_sizes,
                level_size,
                no_anchors
            )
            tmp = anchors_to_mask_nms(
                factor = factor,
                curr_size = level_size,
                anchor_sizes = anchor_sizes,
                curr_bb = self.curr_bb,
                permute = True,
                name = level + "_anchors")
            tmp.set_shape(
                [level_size,level_size,9 * no_anchors])
            tmp_levels.append(tmp)

        if self.is_training == True:
            tmp = [tf.expand_dims(train_image,0),
                   *[tf.expand_dims(t,0) for t in tmp_levels]]
            tmp = tf.map_fn(
                paired_data_augmentation,
                tmp,
                dtype = [tf.float32,
                         *[tf.float32 for t in tmp_levels]]
            )
            train_image = tmp[0]
            tmp_levels = tmp[1:]
            train_image = tf.squeeze(train_image,0)
            tmp_levels = [tf.squeeze(t,0) for t in tmp_levels]

        else:
            train_image = tf.map_fn(
                lambda x: tf.image.convert_image_dtype(x, dtype=tf.float32),
                train_image,
                dtype = tf.float32
            )

        n_anchors = [self.no_anchors[l] for l in self.levels]
        tmp_levels = [expand_truth(t,l) for t,l in zip(tmp_levels,n_anchors)]

        for t,l in zip(tmp_levels,self.levels):
            size = self.level_sizes[l]
            n = self.no_anchors[l]
            t.set_shape([size,size,9,n])

        tmp = tf.train.shuffle_batch(
            [train_image,
             orig_image,
             train_input_queue,
             self.curr_bb,
             *tmp_levels],
            batch_size=self.batch_size,
            capacity = 32,
            min_after_dequeue = 16,
            allow_smaller_final_batch = True
            )

        self.inputs,self.orig_inputs,self.file_names = tmp[0:3]
        self.bounding_box = tmp[3]
        self.truth_boxes = tmp[4:]

        if self.smart_preprocessing == True:
            hsv = rgb2hsv(self.inputs)
            edges = sobel_edges(self.inputs)
            self.inputs = tf.concat(
                self.inputs,
                hsv,
                edges
            )

    def get_feature_map(self):
        """
        Uses a MobileNet-V2 to get a multi-layer feature map, which will then
        be used to build a feature pyramid network.
        """

        self.feature_dict = {}

        with slim.arg_scope(
            mobilenet_v2.training_scope(
                weight_decay = self.weight_decay)):
            _, end_points = mobilenet_v2.mobilenet(
                self.inputs,
                num_classes = False,
                is_training = self.is_training)

            self.feature_map = {
                'C3': end_points[
                    'layer_7'],
                'C4': end_points[
                    'layer_14'],
                'C5': end_points[
                    'layer_19']
                }

    def get_feature_pyramid(self):
        """
        Uses the self.feature_map to get a feature pyramid network.
        """

        def combine_layers(fp,fm,name):
            map_shape = fm.get_shape().as_list()
            fp_shape = fp.get_shape().as_list()
            stride = int(map_shape[1] / fp_shape[1])
            up_sample_fp = slim.conv2d_transpose(
                fp,
                128,
                [3,3],
                stride = (stride,stride),
                activation_fn = tf.nn.elu,
                scope = name + '_fp_resize')
            gate = slim.conv2d(fm,num_outputs = 128, kernel_size = [1,1],
                               scope = name + '_fm_gate_conv3x3',
                               activation_fn = tf.nn.sigmoid)
            fm = slim.conv2d(fm, num_outputs = 128, kernel_size = [1,1],
                             scope = name + '_fm_conv1x1') * gate
            fp = fm + up_sample_fp
            fp = slim.conv2d(fm,num_outputs = 128, kernel_size = [1,3],
                             scope = name + '_fp_conv1x3')
            fp = slim.conv2d(fp,num_outputs = 128, kernel_size = [3,1],
                             scope = name + '_fp_conv3x1')
            return fp

        with slim.arg_scope(
            [slim.conv2d],
            activation_fn = tf.nn.elu,
            weights_initializer = tf.contrib.layers.xavier_initializer(
                uniform = False
            ),
            weights_regularizer = slim.l2_regularizer(self.weight_decay)):

            self.feature_pyramid = {}

            self.feature_pyramid['P5'] = slim.conv2d(self.feature_map['C5'],
                                                     num_outputs = 128,
                                                     kernel_size = [1,3],
                                                     stride = 1,
                                                     scope = 'P5_1x3')
            self.feature_pyramid['P5'] = slim.conv2d(self.feature_pyramid['P5'],
                                                     num_outputs = 128,
                                                     kernel_size = [3,1],
                                                     stride = 1,
                                                     scope = 'P5_3x1') #s / 32

            self.feature_pyramid['P4'] = combine_layers(
                self.feature_pyramid['P5'],
                self.feature_map['C4'],
                'P4') #s / 16
            self.feature_pyramid['P3'] = combine_layers(
                self.feature_pyramid['P4'],
                self.feature_map['C3'],
                'P3') #s / 8

    def get_object_vectors(self):
        """
        Predicts the coordinates and confidence for the bounding boxes. To do
        so, it uses the layers from the self.feature_pyramid and it extracts
        the highest confidence box out of all the layers.
        This produces a tensor with shape:
        [None,self.height,self.width,5,self.no_anchors['level']]
        """

        self.box_vectors = []

        def prediction_layers(curr_level,n_predictions):
            layers = []
            for j in range(n_predictions):

                scores = slim.conv2d(
                    curr_level,
                    num_outputs = 256,
                    kernel_size = [1,1],
                    stride = 1,
                    activation_fn = tf.nn.elu,
                    scope = level + '_pre_box_scores_conv1x1_' + str(j))
                scores = slim.conv2d(
                    scores,
                    num_outputs = 1,
                    kernel_size = [1,1],
                    stride = 1,
                    activation_fn = None,
                    scope = level + '_box_scores_conv1x1_' + str(j))

                box_coords = slim.conv2d(
                    curr_level,
                    num_outputs = 256,
                    kernel_size = [1,1],
                    stride = 1,
                    activation_fn = tf.nn.elu,
                    scope = level + '_pre_box_coord_conv1x1_' + str(j))

                box_coords = slim.conv2d(
                    box_coords,
                    num_outputs = 4,
                    kernel_size = [1,1],
                    stride = 1,
                    activation_fn = tf.nn.relu6,
                    scope = level + '_box_coord_conv1x1_' + str(j))

                layer = tf.concat([scores,box_coords],axis = 3)
                layers.append(layer)

            return tf.stack(layers,-1)

        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.contrib.layers.xavier_initializer(
                uniform = False
            )):

            for level in self.levels:
                box_vectors = []
                curr_level = self.feature_pyramid[level]
                box_vectors = prediction_layers(
                    curr_level,
                    self.no_anchors[level])
                self.box_vectors.append(box_vectors)

    def get_loss_metrics(self):
        """
        Loss function and metrics for the RPN.
        L = SmoothL1(true_t,pred_t) + LogLoss(true_iou,pred_iou)
        """

        def binarize(tensor,threshold = self.confidence_threshold):
            return tf.cast(
                tf.greater_equal(tensor,threshold),
                tf.float32)

        def loss_metrics(level,true):
            """
            Calculates the t-values for the predictions and their loss values.
            """

            def huber_loss(labels,predictions,weights,d = 1.):
                err = tf.abs(labels - predictions, name='abs')
                mg = tf.constant(d, name='max_grad')
                lin = mg * (err - 0.5 * mg)
                quad = 0.5 * err * err
                return tf.where(err < mg, quad, lin) * weights

            def log_loss(labels,predictions,eps = 1e-12):
                pos_weight = tf.reduce_sum(1 - labels)
                neg_weight = tf.reduce_sum(labels) + 1

                class_loss = tf.nn.weighted_cross_entropy_with_logits(
                    labels,
                    predictions,
                    (pos_weight / neg_weight)
                )
                return tf.reduce_mean(class_loss)

            pred_iou = level[:,:,:,0,:]
            pred_x = level[:,:,:,1,:]
            pred_y = level[:,:,:,2,:]
            pred_h = level[:,:,:,3,:]
            pred_w = level[:,:,:,4,:]

            n_reg = tf.multiply(*pred_x.get_shape().as_list()[1:3])
            n_reg = tf.cast(n_reg,dtype = tf.float32)
            n_cls = tf.constant(self.batch_size,dtype = tf.float32)

            true_iou = true[:,:,:,0,:]
            binarized_iou = binarize(true_iou)
            no_anchors = true_iou.get_shape().as_list()[-1]
            true_anchor = true[:,:,:,1:5,:]
            true_t = true[:,:,:,5:,:]

            true_mask = tf.cast(
                tf.greater_equal(true_iou,self.confidence_threshold),
                tf.float32
                )

            if len(true_mask.get_shape().as_list()) == 5:
                true_mask_4 = tf.concat([true_mask for i in range(0,4)],
                                        -2)
            elif len(true_mask.get_shape().as_list()) == 4:
                true_mask_4 = tf.stack([true_mask for i in range(0,4)],
                                       -2)

            pred_tx = pred_x - true_anchor[:,:,:,0,:]
            pred_tx = tf.where(tf.abs(true_anchor[:,:,:,2,:]) > 0,
                               pred_tx/true_anchor[:,:,:,2,:],
                               tf.zeros(tf.shape(pred_tx)))
            pred_ty = pred_y - true_anchor[:,:,:,1,:]
            pred_ty = tf.where(tf.abs(true_anchor[:,:,:,3,:]) > 0,
                               pred_ty/true_anchor[:,:,:,3,:],
                               tf.zeros(tf.shape(pred_tx)))
            pred_th = tf.where(
                tf.abs(true_anchor[:,:,:,2,:]) > 0,
                tf.log(pred_h/true_anchor[:,:,:,2,:] + 1e-7),
                tf.zeros(tf.shape(pred_tx)))
            pred_tw = tf.where(
                tf.abs(true_anchor[:,:,:,3,:]) > 0,
                tf.log(pred_w/true_anchor[:,:,:,3,:] + 1e-7),
                tf.zeros(tf.shape(pred_tx)))

            pred_t = tf.stack((pred_tx,
                               pred_ty,
                               pred_th,
                               pred_tw),
                               3)

            huber_loss = huber_loss(true_t,pred_t,true_mask_4)
            class_loss = log_loss(binarized_iou,pred_iou)

            huber_loss = tf.reduce_sum(huber_loss,[1,2,3])
            huber_loss /= tf.reduce_sum(true_mask_4) + 1
            huber_loss = tf.reduce_mean(huber_loss)

            return [binarized_iou,pred_iou,
                    true_t,pred_t,
                    true_mask_4,true_mask,
                    huber_loss,class_loss]

        all_ps = []
        for i in range(len(self.box_vectors)):
            all_ps.append(
                loss_metrics(self.box_vectors[i],
                             self.truth_boxes[i]))

        if self.weight_decay > 0:
            regularization_loss = tf.losses.get_regularization_loss()
            regularization_loss *= self.weight_decay
        else:
            regularization_loss = 0.


        self.class_loss = tf.add_n([p[7] for p in all_ps]) * self.class_factor
        self.huber_loss = tf.add_n([p[6] for p in all_ps]) * self.reg_factor

        self.loss = tf.add_n(
            [self.huber_loss,
             self.class_loss,
             regularization_loss]
        )

        self.mse = tf.metrics.mean_squared_error(
            tf.concat([tf.layers.flatten(x[2]) for x in all_ps],
                      axis = 1),
            tf.concat([tf.layers.flatten(x[3]) for x in all_ps],
                      axis = 1),
            tf.concat([tf.layers.flatten(x[4]) for x in all_ps],
                      axis = 1))

        self.binary_truth = tf.concat(
            [tf.layers.flatten(binarize(x[0])) for x in all_ps],
            axis = 1
            )

        self.binary_prediction = tf.concat(
            [tf.layers.flatten(
                binarize(tf.sigmoid(x[1]),0.5)) for x in all_ps],
            axis = 1
            )
        self.pred_flat_neg = tf.concat(
            [tf.layers.flatten(x[1][:,:,:,0,]) for x in all_ps],
            axis = 1
            )
        self.pred_flat_pos = tf.concat(
            [tf.layers.flatten(x[1][:,:,:,1,]) for x in all_ps],
            axis = 1
            )

        self.auc = tf.metrics.auc(self.binary_truth,
                                  self.binary_prediction)

        self.tp = tf.metrics.true_positives(self.binary_truth,
                                            self.binary_prediction)
        self.tn = tf.metrics.true_negatives(self.binary_truth,
                                            self.binary_prediction)
        self.fp = tf.metrics.false_positives(self.binary_truth,
                                             self.binary_prediction)
        self.fn = tf.metrics.false_negatives(self.binary_truth,
                                             self.binary_prediction)

        self.sensitivity = self.tp[0] / (self.tp[0] + self.fn[0])
        self.specificity = self.tn[0] / (self.tn[0] + self.fp[0])

        self.update_metrics = tf.group(
            self.mse[1],
            self.auc[1],
            self.tp[1],
            self.tn[1],
            self.fp[1],
            self.fn[1]
        )
        self.no_images_batch = tf.shape(self.inputs)[0]

    def pre_flight_operations(self):
        """
        Standard operations to create summaries, set random seeds and assign
        the configuration for the session.
        """
        self.saver = tf.train.Saver()

        #Setting seeds for randomness
        tf.set_random_seed(1)
        np.random.seed(1)

        #Session configuration
        self.config = tf.ConfigProto(**self.config_arguments)

        if self.mode == 'train':
            #Optimizer, minimization and variable initiation
            self.optimizer = tf.contrib.opt.NadamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            #Summaries
            self.summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            for variable in slim.get_model_variables():
                self.summaries.add(
                    tf.summary.histogram(variable.op.name, variable)
                    )
            self.summaries.add(tf.summary.scalar('loss',
                                                 self.loss))
            self.summaries.add(tf.summary.scalar('huber_loss',
                                                 self.huber_loss))
            self.summaries.add(tf.summary.scalar('class_loss',
                                                 self.class_loss))

            self.summaries.add(tf.summary.scalar('mse',
                                                 self.mse[0]))
            self.summaries.add(tf.summary.scalar('auc',
                                                 self.auc[0]))
            self.summaries.add(tf.summary.scalar('sensitivity',
                                                 self.sensitivity))
            self.summaries.add(tf.summary.scalar('specificity',
                                                 self.specificity))
            self.summaries.add(
                tf.summary.image('image',
                                 self.image_with_bb,
                                 max_outputs = 10))

            for i,iwb in enumerate(self.images_with_boxes):
                self.summaries.add(
                    tf.summary.image('image_with_boxes_' + str(i),
                                     iwb,
                                     max_outputs = 10))

            for i,iwbf in enumerate(self.images_with_boxes_fixed):
                self.summaries.add(
                    tf.summary.image('image_with_boxes_fixed_' + str(i),
                                     iwbf,
                                     max_outputs = 10))

            for i,iwpa in enumerate(self.images_with_positive_anchors):
                self.summaries.add(
                    tf.summary.image('image_with_positive_anchors_' + str(i),
                                     iwpa,
                                     max_outputs = 10))

            self.summary_op = tf.summary.merge(list(self.summaries),
                                               name='summary_op')


        elif self.mode == 'test':
            self.config = tf.ConfigProto(**self.config_arguments)
        elif self.mode == 'predict':
            pass

        #Defining variables to be initialized
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.tables_initializer(),
                             tf.local_variables_initializer())

    def train(self):
        """
        Trains the model.
        """

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

            for i in range(1,self.number_of_steps + 1):
                #print(self.sess.run(self.mask))
                a = time.perf_counter()
                _,loss,hl,cl,_,tb = self.sess.run(
                    [self.train_op,
                     self.loss,
                     self.huber_loss,
                     self.class_loss,
                     self.update_metrics,
                     self.truth_boxes[0]]
                    )

                #print(np.unique(tb[:,:,:,1,:]),np.unique(tb[:,:,:,2,:]),
                #      np.unique(tb[:,:,:,3,:]),np.unique(tb[:,:,:,4,:]))
                b = time.perf_counter()
                self.time_list.append(b - a)
                if i % self.log_every_n_steps == 0 or\
                 i % self.number_of_steps == 0 or i == 1:
                    mse,auc,sens,spec,tp,tn,fp,fn = \
                    self.sess.run(
                        [self.mse[0],
                         self.auc[0],
                         self.sensitivity,
                         self.specificity,
                         self.tp[0],self.tn[0],
                         self.fp[0],self.fn[0]]
                        )

                    tp,tn,fp,fn = int(tp),int(tn),int(fp),int(fn)
                    last_time = self.time_list[-1]
                    print(
                        _LOG_TRAIN.format(i,last_time,loss,cl,hl,
                                          mse,auc)
                        )
                    print('\t{0:.0f}/{1:.0f} correctly predicted positives'.format(
                        tp,tp + fn
                    ))
                    print('\tSensitivity = {0:f}'.format(sens))
                    print('\t{0:.0f}/{1:.0f} correctly predicted negatives'.format(
                        tn,tn + fp
                    ))
                    print('\tSpecificity = {0:f}'.format(spec))

                if i % self.save_summary_steps == 0 or i == 1:
                    summary = self.sess.run(self.summary_op)
                    self.writer.add_summary(summary,i)
                    self.writer.flush()
                    print(_SUMMARY_TRAIN.format(
                        i,
                        self.save_summary_folder)
                          )
                    self.sess.run(tf.local_variables_initializer())

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
            self.writer.flush()
            print(_SUMMARY_TRAIN.format(
                i,
                self.save_summary_folder)
                  )

            loss = self.sess.run(self.loss)
            print(
                _LOG_TRAIN_FINAL.format(self.number_of_steps,self.no_images,
                                        np.mean(self.time_list),loss)
            )
            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def test(self):
        """
        Tests the model.
        """

        with tf.Session(config = self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            all_auc_eval = []
            all_sen_eval = []
            all_spe_eval = []
            all_mse_eval = []

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.time_list = []

            total_images = 0
            for i in range(self.number_of_steps):
                self.sess.run(tf.local_variables_initializer())
                a = time.perf_counter()
                n_images,_ = self.sess.run(
                    [self.no_images_batch,self.update_metrics]
                    )
                total_images += n_images
                b = time.perf_counter()
                self.time_list.append(b - a)

                auc,sen,spe,mse = self.sess.run(
                    [self.auc[0],
                     self.sensitivity,
                     self.specificity,
                     self.mse[0]]
                    )

                print(_LOG_TEST.format(
                    self.time_list[-1]/n_images,
                    n_images,
                    auc,
                    sen,
                    spe,
                    mse
                ))

                all_auc_eval.append(auc)
                all_sen_eval.append(sen)
                all_spe_eval.append(spe)
                all_mse_eval.append(mse)

            print(_LOG_TEST_FINAL.format(
                np.sum(self.time_list)/total_images,
                total_images,
                np.mean(all_auc_eval),
                np.mean(all_sen_eval),
                np.mean(all_spe_eval),
                np.mean(all_mse_eval)
            ))

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def predict(self):
        """
        Performs predictions using the model.
        """

        def format_output(boxes,file_names):
            output = ''
            for box,file_name in zip(boxes,file_names):
                positive = np.greater(box[:,:,0],0)
                positive = np.stack((positive for i in range(5)),-1)
                fix_x = np.tile(
                    np.expand_dims(
                        np.arange(self.input_height,step = self.factor),
                        1),
                    [self.input_height / self.factor,1])
                fix_y = np.tile(
                    np.expand_dims(
                        np.arange(self.input_width,step = self.factor),
                        0),
                    [1,self.input_width / self.factor])
                fix_n = np.zeros((self.input_height / self.factor,
                                  self.input_width / self.factor))

                fix_array = np.stack((fix_n,fix_x,fix_y,fix_n,fix_n),-1)
                box[:,:,1:] = box[:,:,1:] * self.factor
                box[:,:,1:3] -= 1/2

                box = box + fix_array
                wh = np.where(box[:,:,0] > 0)
                bb_vectors = box[wh[0],wh[1],:]
                bb_vectors = ';'.join([':'.join(bb) for bb in bb_vectors])
                output += file_name + bb_vectors + '\n'
            return output

        with tf.Session(self.config) as self.sess:
            self.sess.run(self.init)
            self.saver.restore(self.sess,self.checkpoint_path)

            all_mse = []
            all_kl = []

            with open(self.save_predictions_path,'ab') as f_handle:
                for i in range(self.number_of_steps):
                    self.sess.run(tf.local_variables_initializer())
                    a = time.perf_counter()
                    boxes, file_names  = self.sess.run(
                        [self.final_boxes,
                         self.file_names])

                    b = time.perf_counter()
                    self.time_list.append(b - a)

                    pickle.dump(
                        format_output(boxes,file_names),
                        f_handle
                        )

            #Stops the coordinator and the queue (or an exception is raised)
            coord.request_stop()
            coord.join(threads)

    def draw_boxes(self):
        """
        Draws boxes for all relevant images - input images (one box *per
        cell*), images with anchors (n boxes *per* cell) and images with
        anchors (p boxes per cell).
        """

        def draw_original_bb(image,bb):
            bb = tf.string_split(tf.reshape(bb,[1]),
                                 self.bb_sep)
            bb = tf.sparse_to_dense(
                bb.indices,
                bb.dense_shape,
                bb.values,
                '')
            bb = tf.squeeze(bb,0)
            bb = tf.string_split(bb,
                                 delimiter = self.coord_sep)
            bb = tf.sparse_to_dense(bb.indices,
                                    bb.dense_shape,
                                    bb.values,
                                    '')
            bb = tf.string_to_number(bb)
            bb = tf.transpose(bb,[1,0])
            x,y,h,w = tf.split(bb,4)
            min_x = x
            min_y = y
            max_x = min_x + h
            max_y = min_y + w
            bb = tf.concat([min_y,min_x,max_y,max_x],axis = 0)

            bb = tf.expand_dims(bb / self.input_height,0)
            bb = tf.transpose(bb,[0,2,1])
            image = tf.expand_dims(image,0)

            output = tf.image.draw_bounding_boxes(
                image,
                bb
            )
            output = tf.squeeze(output,0)
            return output

        def round_precision(x, decimals = 0):
            multiplier = tf.constant(10**decimals, dtype=x.dtype)
            return tf.round(x * multiplier) / multiplier

        def decode_draw_box(image,box_vectors,nms,fix = False):
            """
            Somewhere I am switching the b.b. coordinates, but I am still
            unaware of *where* exactly this happens. This is a temporary
            solution for a hopefully temporary problem. Hence, the "fix"
            argument.
            """
            f = tf.cast(factor,tf.float32)
            shape = box_vectors.get_shape().as_list()
            c = box_vectors[:,:,0,:]
            if nms == True:
                top_k = tf.nn.top_k(tf.layers.flatten(tf.expand_dims(c,axis = 0)),
                                    k = 100)
                min = tf.reduce_min(
                    top_k.values,
                    axis = None
                    )
            else:
                min = self.confidence_threshold
            if fix == True:
                general_fix = 1.
            else:
                general_fix = 0.
            min = tf.maximum(0.5,min)
            binary_c = tf.greater_equal(c,min)

            fix_x = tf.tile(
                tf.expand_dims(tf.range(shape[0]),1),
                [1,shape[1]]
                )
            fix_x = tf.cast(fix_x,tf.float32)
            fix_x = tf.stack([fix_x for i in range(4)],2) - general_fix

            fix_y = tf.tile(
                tf.expand_dims(tf.range(shape[1]),0),
                [shape[0],1]
                )
            fix_y = tf.cast(fix_y,tf.float32)
            fix_y = tf.stack([fix_y for i in range(4)],2) - general_fix

            idx = tf.where(tf.equal(binary_c,True))
            sparse_x = tf.gather_nd(box_vectors[:,:,1,:] + fix_x,idx) * f
            sparse_y = tf.gather_nd(box_vectors[:,:,2,:] + fix_y,idx) * f
            sparse_h = tf.gather_nd(box_vectors[:,:,3,:],idx) * f
            sparse_w = tf.gather_nd(box_vectors[:,:,4,:],idx) * f

            min_x = sparse_x - sparse_h/2.
            min_y = sparse_y - sparse_w/2.
            max_x = min_x + sparse_h
            max_y = min_y + sparse_w

            if fix == True:
                output = tf.truediv(
                    tf.stack([min_x,min_y,max_x,max_y],axis = 1),
                    shape[0] * f)
            else:
                output = tf.truediv(
                    tf.stack([min_y,min_x,max_y,max_x],axis = 1),
                    shape[0] * f)
            output = tf.expand_dims(output,0)
            image = tf.expand_dims(image,0)
            output_image = tf.image.draw_bounding_boxes(image,output)
            output_image = tf.squeeze(output_image,0)
            return output_image

        def decode_draw_box_pred(image,box_vectors):
            return decode_draw_box(image,box_vectors,True)

        def decode_draw_box_truth(image,box_vectors):
            return decode_draw_box(image,box_vectors,False)

        def decode_draw_box_fixed(image,box_vectors):
            return decode_draw_box(image,box_vectors,True,True)

        self.images_with_boxes = []
        self.images_with_boxes_fixed = []
        self.images_with_positive_anchors = []
        self.pred_box_coords = []
        self.truth_box_coords = []
        for i in range(len(self.box_vectors)):
            factor = self.factors[self.levels[i]]
            iwb = tf.map_fn(
                lambda x: decode_draw_box_pred(x[0],x[1]),
                [self.inputs,self.box_vectors[i]],
                dtype = tf.float32)
            iwbf = tf.map_fn(
                lambda x: decode_draw_box_fixed(x[0],x[1]),
                [self.inputs,self.box_vectors[i]],
                dtype = tf.float32)
            iwpa = tf.map_fn(
                lambda x: decode_draw_box_truth(x[0],x[1]),
                [self.inputs,self.truth_boxes[i]],
                dtype  = tf.float32)
            self.images_with_boxes.append(iwb)
            self.images_with_boxes_fixed.append(iwbf)
            self.images_with_positive_anchors.append(iwpa)
        self.image_with_bb = tf.map_fn(
            lambda x: draw_original_bb(x[0],x[1]),
            [self.orig_inputs,self.bounding_box],
            dtype = tf.float32,
            infer_shape = False
        )

parser = argparse.ArgumentParser(
    description = "Implementation of a single class PFN-based RPN."
    )

#DATA
parser.add_argument('--data_directory',
                    dest = 'data_directory',
                    action = ToDirectory,
                    required = True,
                    help = 'Path to image data.')
parser.add_argument('--extension',
                    dest = 'extension',
                    action = "store",
                    default = 'png',
                    help = 'Image format.')
parser.add_argument('--bounding_box_csv',
                    dest = 'bounding_box_csv',
                    action = ToDirectory,
                    required = True,
                    help = 'CSV file containing data on bounding boxes.')
parser.add_argument('--bb_sep',
                    dest = 'bb_sep',
                    action = "store",
                    default = ';',
                    help = 'Character separating entries in CSV.')
parser.add_argument('--coord_sep',
                    dest = 'coord_sep',
                    action = "store",
                    default = ':',
                    help = 'Character separating coordinates in CSV.')
parser.add_argument('--input_height',
                    dest = 'input_height',
                    action = "store",
                    type = int,
                    default = 1024,
                    help = 'Input height.')
parser.add_argument('--input_width',
                    dest = 'input_width',
                    action = "store",
                    type = int,
                    default = 1024,
                    help = 'Input width.')
parser.add_argument('--batch_size',
                    dest = 'batch_size',
                    action = "store",
                    type = int,
                    default = 1,
                    help = 'Mini-batch size.')
parser.add_argument('--truth_only',
                    dest = 'truth_only',
                    action = "store_true",
                    default = False,
                    help = 'Input width.')
parser.add_argument('--mode',
                    dest = 'mode',
                    action = "store",
                    default = 'train',
                    help = 'Sets the mode - [train,test,predict].')
#NETWORK
parser.add_argument('--confidence_threshold',
                    dest = 'confidence_threshold',
                    action = "store",
                    type = float,
                    default = 0.7,
                    help = 'Minimum confidence level for grid cells.')
parser.add_argument('--checkpoint_path',
                    dest = 'checkpoint_path',
                    action = "store",
                    default = None,
                    help = 'Path to network checkpoint.')

#PREPROCESSING
parser.add_argument('--smart_preprocessing',
                    dest = 'smart_preprocessing',
                    action = "store_true",
                    default = False,
                    help = 'Use smart preprocessing (concatenate HSV\
                     and Sobel edges).')

#TRAIN
parser.add_argument('--learning_rate',
                    dest = 'learning_rate',
                    action = "store",
                    type = float,
                    default = 0.001,
                    help = 'Learning rate.')
parser.add_argument('--weight_decay',
                    dest = 'weight_decay',
                    action = "store",
                    type = float,
                    default = 0.001,
                    help = 'Beta for L2 normalization.')
parser.add_argument('--class_factor',
                    dest = 'class_factor',
                    action = "store",
                    type = float,
                    default = 1,
                    help = 'Factor for classification loss.')
parser.add_argument('--reg_factor',
                    dest = 'reg_factor',
                    action = "store",
                    type = float,
                    default = 1,
                    help = 'Factor for regression loss.')
parser.add_argument('--training_nms',
                    dest = 'training_nms',
                    action = "store_true",
                    default = False,
                    help = 'Trigger NMS in training.')
parser.add_argument('--save_summary_folder',
                    dest = 'save_summary_folder',
                    action = ToDirectory,
                    default = 'summary',
                    help = 'Folder to store summaries.')
parser.add_argument('--save_checkpoint_folder',
                    dest = 'save_checkpoint_folder',
                    action = ToDirectory,
                    default = 'checkpoint',
                    help = 'Folder to store checkpoints.')
parser.add_argument('--save_summary_steps',
                    dest = 'save_summary_steps',
                    action = "store",
                    type = int,
                    default = 500,
                    help = 'How often should summaries be saved.')
parser.add_argument('--save_checkpoint_steps',
                    dest = 'save_checkpoint_steps',
                    action = "store",
                    type = int,
                    default = 500,
                    help = 'How often should checkpoints be saved.')
parser.add_argument('--number_of_steps',
                    dest = 'number_of_steps',
                    action = "store",
                    type = int,
                    default = 1000,
                    help = 'Total number of steps.')
parser.add_argument('--epochs',
                    dest = 'epochs',
                    action = "store",
                    type = int,
                    default = 40,
                    help = 'Number of epochs (overrides number_of_steps).')
parser.add_argument('--log_every_n_steps',
                    dest = 'log_every_n_steps',
                    action = "store",
                    type = int,
                    default = 500,
                    help = 'How often run information should be printed.')

#PREDICTION
parser.add_argument('--save_predictions_path',
                    dest = 'save_predictions_path',
                    action = ToDirectory,
                    default = 'prediction.csv',
                    help = 'File where predictions should be stored.')

args = parser.parse_args()

RPN(#Data
    data_directory = args.data_directory,
    extension = args.extension,
    bounding_box_csv = args.bounding_box_csv,
    bb_sep = args.bb_sep,
    coord_sep = args.coord_sep,
    input_width = args.input_width,
    input_height = args.input_height,
    batch_size = args.batch_size,
    truth_only = args.truth_only,
    mode = args.mode,
    #Network
    confidence_threshold = args.confidence_threshold,
    checkpoint_path = args.checkpoint_path,
    #Preprocessing
    smart_preprocessing = args.smart_preprocessing,
    #Train
    learning_rate = args.learning_rate,
    weight_decay = args.weight_decay,
    class_factor = args.class_factor,
    reg_factor = args.reg_factor,
    training_nms = args.training_nms,
    save_summary_folder = args.save_summary_folder,
    save_checkpoint_folder = args.save_checkpoint_folder,
    save_summary_steps = args.save_summary_steps,
    save_checkpoint_steps = args.save_checkpoint_steps,
    number_of_steps = args.number_of_steps,
    epochs = args.epochs,
    log_every_n_steps = args.log_every_n_steps,
    #Prediction
    save_predictions_path = args.save_predictions_path)
