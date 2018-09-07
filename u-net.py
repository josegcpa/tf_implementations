# -*- coding: utf-8 -*-
"""Multi purpose implementation of a U-Net

This script is an implementation of the U-Net proposed in [1]. It features some
tweaks to improve classification and training/testing/prediction speed, namely:
    * residual links - instead of being regular identity links, links can have
    a residual architecture [2]. This involves branching the link into two
    layers, one carrying the identity and the other performing two convolutions
    in a parallel fashion and summing both of them in the end;
    * convolution factorization - convolutions can be factorized to improve
    speed [3]. This means that instead of performing 9 operations in a 3x3
    convolution, the network only needs to perform 6 operations by factorizing
    the 3x3 convolution into a 1x3 convolution followed by a 3x1 convolution
    (or vice-versa);
    * Iglovikov loss function - instead of using the standard function for
    cross entropy, an extra non-differentiable term used to measure
    segmentation quality, the Jaccard Index (or Intersection Over Union - IOU),
    is added to the loss function [4].

Training example:
    $ python3 u-net.py --dataset_dir split_512/train/input/ \
    --truth_dir split_512/train/truth/ \
    --padding SAME \
    --batch_size 4 \
    --log_every_n_steps 5 \
    --input_height 512 \
    --input_width 512 \
    --n_classes 2 \
    --number_of_steps 3120 \
    --save_checkpoint_folder split_512/checkpoint_residuals \
    --save_summary_folder split_512/summary_residuals \
    --factorization \
    --residuals

Help:
    $ python3 u-net.py -h

[1] https://arxiv.org/abs/1505.04597
[2] https://arxiv.org/abs/1512.03385
[3] https://arxiv.org/abs/1512.00567
[4] https://arxiv.org/pdf/1706.06169"""

import argparse
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Defining functions

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """
    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

def u_net(inputs,
          final_endpoint = 'Final',
          padding = 'VALID',
          factorization = False,
          beta = 0,
          residuals = False,
          n_classes = 2,
          resize = False,
          resize_height = 256,
          resize_width = 256,
          depth_mult = 1):

    """
    Implementation of a standard U-net with some tweaks, namely:
        * Possibility of choosing whether padding should be SAME or VALID (for
        SAME it is necessary that the height % 16 == 0 and width % 16 == 0)
        * Possibility of factorizing the convolutions (a 3x3 convolution
        becomes a 1x3 and a 3x1 convolutions, turning the number of
        operations/filter from 9 to 6)
        * beta is the beta parameter in l2 regularization
        * residuals activates residual blocks in the links

    Arguments [default]:
    * inputs - input tensor;
    * final_endpoint - final layer to return ['Final']
    * padding - whether VALID or SAME padding should be used ['VALID']
    * factorization - whether convolutions should be factorized (3x3conv become
    sequential 1x3 and 3x1 convolutions) [False]
    * beta - L2 regularization factor [0]
    * residuals - whether to use residual linkers in the shortcut links [False]
    * n_classes - number of classes in the output layer (only works with 2) [2]
    * resize - whether the input should be resized before training [False]
    * resize_height - height of the resized input [256]
    * resize_width - width of the resized input [256]
    * depth_mult - factor to increase or decrease the depth in each layer
    """

    endpoints = {}

    def conv2d(x,depth,size,stride,scope,
               factorization = False,padding = padding):
        if factorization == False:
            scope = scope + str(size) + 'x' + str(size)
            x = slim.conv2d(x,depth,[size,size],stride = stride,
                            scope = scope,padding = padding)
        else:
            scope_1 = scope + str(size) + 'x' + '1'
            scope_2 = scope + '1' + 'x' + str(size)
            x = slim.conv2d(x,depth,[size,1],stride = 1,scope = scope_1,
                            padding = padding)
            x = slim.conv2d(x,depth,[1,size],stride = 1,scope = scope_2,
                            padding = padding)
            if stride > 1:
                scope = scope + 'maxpool2d_' + str(stride) + 'x' + str(stride)
                x = slim.max_pool2d(x, [stride,stride], stride = stride,
                                    padding = 'SAME',scope = scope)

        return x

    def block(x,depth,size,stride,padding,factorization = False):
        x = conv2d(x,depth,size,stride,'conv2d_1_',factorization)
        x = conv2d(x,depth,size,stride,'conv2d_2_',factorization)
        return x

    def crop(concat_net,n):
        n = int(n)
        slice_size = concat_net.get_shape().as_list()
        slice_size = [
            - 1,
            int(slice_size[1] - n * 2),
            int(slice_size[2] - n * 2),
            slice_size[3]
        ]

        concat_net = tf.slice(concat_net,
                              [0,n,n,0],
                              slice_size
                              )
        return concat_net

    def residual_block(input_n,depth,size,factorization):
        r = conv2d(input_n,depth,size,1,'res_conv2d_0_',
                   factorization = factorization,padding = 'SAME')
        r = conv2d(r,depth,size,1,'res_conv2d_1_',
                   factorization = factorization,padding = 'SAME')
        return tf.add(input_n,r)

    if beta > 0:
        weights_regularizer = tf.contrib.layers.l2_regularizer(beta)
    else:
        weights_regularizer = None

    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
        activation_fn = tf.nn.relu6,
        padding = padding,
        weights_regularizer = weights_regularizer):

        with tf.variable_scope('U-net', None, [inputs]):

            if resize == True:
                inputs = tf.image.resize_images(
                    inputs,
                    [resize_height,resize_width],
                    method = tf.image.ResizeMethod.BILINEAR)

            with tf.variable_scope('Red_Operations',None,[inputs]):
                with tf.variable_scope('Red_Block_1',None):
                    d_current = int(64 * depth_mult)
                    net = block(inputs,d_current,3,1,factorization,[inputs])
                    if residuals == True:
                        endpoints['Red_Block_1'] = residual_block(
                            net,d_current,3,factorization
                        )
                    else:
                        endpoints['Red_Block_1'] = net
                    if final_endpoint == 'Red_Block_1':
                        return net,endpoints
                    net = slim.max_pool2d(net,[2,2],stride = 2,
                                            scope = 'maxpool2d_2x2')

                with tf.variable_scope('Red_Block_B',None,[net]):
                    d_current = int(128 * depth_mult)
                    net = block(net,d_current,3,1,factorization)
                    if residuals == True:
                        endpoints['Red_Block_2'] = residual_block(
                            net,d_current,3,factorization
                        )
                    else:
                        endpoints['Red_Block_2'] = net
                    if final_endpoint == 'Red_Block_2':
                        return net,endpoints
                    net = slim.max_pool2d(net,[2,2],stride = 2,
                                            scope = 'maxpool2d_2x2')

                with tf.variable_scope('Red_Block_3',None,[net]):
                    d_current = int(256 * depth_mult)
                    net = block(net,d_current,3,1,factorization)
                    if residuals == True:
                        endpoints['Red_Block_3'] = residual_block(
                            net,d_current,3,factorization
                        )
                    else:
                        endpoints['Red_Block_3'] = net
                    if final_endpoint == 'Red_Block_3':
                        return net,endpoints
                    net = slim.max_pool2d(net,[2,2],stride = 2,
                                            scope = 'maxpool2d_2x2')

                with tf.variable_scope('Red_Block_4',None,[net]):
                    d_current = int(512 * depth_mult)
                    net = block(net,d_current,3,1,factorization)
                    if residuals == True:
                        endpoints['Red_Block_4'] = residual_block(
                            net,d_current,3,factorization
                        )
                    else:
                        endpoints['Red_Block_4'] = net
                    if final_endpoint == 'Red_Block_4':
                        return net,endpoints
                    net = slim.max_pool2d(net,[2,2],stride = 2,
                                            scope = 'maxpool2d_2x2')

                with tf.variable_scope('Red_Block_5',None,[net]):
                    d_current = int(1024 * depth_mult)
                    net = block(net,d_current,3,1,factorization)
                    endpoints['Red_Block_5'] = net
                    if final_endpoint == 'Red_Block_5':
                        return net,endpoints

            with tf.variable_scope('Rec_Operations',None,[net]):
                with tf.variable_scope('Rec_Block_1',None,[net]):
                    d_current = int(512 * depth_mult)
                    new_size = net.get_shape().as_list()
                    new_size = [new_size[1] * 2,new_size[2] * 2]
                    net = tf.image.resize_nearest_neighbor(net,new_size)
                    net = slim.conv2d(net,d_current,[2,2],scope = 'conv2d_0_',
                                        padding = 'SAME')
                    if padding == 'VALID':
                        n = endpoints['Red_Block_4'].get_shape().as_list()[1]
                        n = (n - net.get_shape().as_list()[1])/2
                        concat_net = crop(endpoints['Red_Block_4'],n)
                    else:
                        concat_net = endpoints['Red_Block_4']
                    net = tf.concat([net,concat_net],
                                      axis = 3)
                    net = block(net,d_current,3,1,padding,factorization)
                    endpoints['Rec_Block_1'] = net
                    if final_endpoint == 'Rec_Block_1':
                        return net,endpoints

                with tf.variable_scope('Rec_Block_2',None,[net]):
                    d_current = int(256 * depth_mult)
                    new_size = net.get_shape().as_list()
                    new_size = [new_size[1] * 2,new_size[2] * 2]
                    net = tf.image.resize_nearest_neighbor(net,new_size)
                    net = slim.conv2d(net,d_current,[2,2],
                                        scope = 'conv2d_0_',
                                        padding = 'SAME')
                    if padding == 'VALID':
                        n = endpoints['Red_Block_3'].get_shape().as_list()[1]
                        n = (n - net.get_shape().as_list()[1])/2
                        concat_net = crop(endpoints['Red_Block_3'],n)
                    else:
                        concat_net = endpoints['Red_Block_3']
                    net = tf.concat([net,concat_net],
                                      axis = 3)
                    net = block(net,d_current,3,1,padding,factorization)
                    endpoints['Rec_Block_2'] = net
                    if final_endpoint == 'Rec_Block_2':
                        return net,endpoints

                with tf.variable_scope('Rec_Block_3',None,[net]):
                    d_current = int(128 * depth_mult)
                    new_size = net.get_shape().as_list()
                    new_size = [new_size[1] * 2,new_size[2] * 2]
                    net = tf.image.resize_nearest_neighbor(net,new_size)
                    net = slim.conv2d(net,d_current,[2,2],
                                        scope = 'conv2d_0_',
                                        padding = 'SAME')
                    if padding == 'VALID':
                        n = endpoints['Red_Block_2'].get_shape().as_list()[1]
                        n = (n - net.get_shape().as_list()[1])/2
                        concat_net = crop(endpoints['Red_Block_2'],n)
                    else:
                        concat_net = endpoints['Red_Block_2']
                    net = tf.concat([net,concat_net],
                                      axis = 3)
                    net = block(net,d_current,3,1,factorization)
                    endpoints['Rec_Block_3'] = net
                    if final_endpoint == 'Rec_Block_3':
                        return net,endpoints

                with tf.variable_scope('Rec_Block_4',None,[net]):
                    d_current = int(64 * depth_mult)
                    new_size = net.get_shape().as_list()
                    new_size = [new_size[1] * 2,new_size[2] * 2]
                    net = tf.image.resize_nearest_neighbor(net,new_size)
                    net = slim.conv2d(net,d_current,[2,2],
                                      scope = 'conv2d_0_',
                                      padding = 'SAME')
                    if padding == 'VALID':
                        n = endpoints['Red_Block_1'].get_shape().as_list()[1]
                        n = (n - net.get_shape().as_list()[1])/2
                        concat_net = crop(endpoints['Red_Block_1'],n)
                    else:
                        concat_net = endpoints['Red_Block_1']
                    net = tf.concat([net,concat_net],
                                      axis = 3)
                    net = block(net,d_current,3,1,padding,factorization)
                    endpoints['Rec_Block_4'] = net
                    if final_endpoint == 'Rec_Block_4':
                        return net,endpoints

                with tf.variable_scope('Final',None,[net]):
                    net = slim.conv2d(net, n_classes, [1, 1],
                                      normalizer_fn = None,
                                      activation_fn = None,
                                      scope='conv2d_0_sigmoid')
                    endpoints['Final'] = net
                    if final_endpoint == 'Final':
                        return net,endpoints

    return net,endpoints

def image_to_array(image_path):
    """Opens image in image_path and returns it as a numpy array.

    Arguments:
    * image_path - the path to an image
    """

    with Image.open(image_path) as o:
        return np.array(o)

def realtime_image_augmentation(image_path_list,truth_path,noise_chance = 0.05,
                                blur_chance = 0.05,flip_chance = 0.5):
    """A generator that performs a series of image manipulations to an original
    image using salt, pepper, channel dropout, gaussian noise, blur, rotation
    and flipping.
    Additionally, it normalizes the input image through linear stretching and
    standardization.

    Arguments:
    * image_path_list - a list of image paths
    * truth_path - a list of ground truth images
    * noise_chance - the probability of adding salt, pepper (to the image or a
    single channel) or gaussian noise
    * blur_chance - chance of applying median bluring to the input
    * flip_chance - chance of rotating the input by n * 90 degrees and/or by
    [0,89] degrees
    """

    def get_proportions():
        true = np.random.sample() / 10
        false = 1 - true
        return true,false

    def get_sp_mask(image):
        true,false = get_proportions()
        mask = np.random.choice(a = [0,1],size = image.shape[:2],
                                p = [false,true])
        mask = tuple([mask for i in range(image.shape[2])])
        mask = np.stack(mask, axis = 2)

        return mask

    np.random.shuffle(image_path_list)

    for image_path in image_path_list:
        image_name = image_path.split('/')[-1]
        truth_image_path = truth_path + '/' + image_name
        image = image_to_array(image_path)
        truth_image = image_to_array(truth_image_path)
        im_shape = image.shape
        if im_shape[0] == im_shape[1]:
            if np.random.sample() <= blur_chance:
                #Median blur
                cv2.medianBlur(image,np.random.randint(2,4) * 2 - 1)
            if np.random.sample() <= noise_chance:
                #Salt
                mask = get_sp_mask(image)
                image[mask == 1] = 255
            if np.random.sample() <= noise_chance:
                #Pepper
                mask = get_sp_mask(image)
                image[mask == 1] = 0
            if np.random.sample() <= noise_chance:
                #Gaussian noise
                mean = 0
                var = 0.1
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,image.shape)
                gauss = gauss.reshape(*image.shape)
                image = image + gauss
            for i in range(0,3):
                if np.random.sample() <= noise_chance:
                    #Channel dropout
                    mask = get_sp_mask(image)[:,:,0]
                    image[:,:,i][mask == 1] = 0
                if np.random.sample() <= noise_chance:
                    #Channel dropin
                    mask = get_sp_mask(image)[:,:,0]
                    image[:,:,i][mask == 1] = 255
            if np.random.sample() <= flip_chance:
                #Image flipping along one of the axis
                flip = np.random.randint(0,2)
                truth_image = np.flip(truth_image,flip)
                image = np.flip(image,flip)
            if np.random.sample() <= 2:
                #Continuous image rotation (0,89)
                rotation = np.random.sample() * 89
                rot_tup = (im_shape[0]/2, im_shape[1]/2)
                rotation_matrix = cv2.getRotationMatrix2D(rot_tup, rotation, 1)
                image[image == 0] = -1
                image = cv2.warpAffine(image, rotation_matrix,
                                       (im_shape[0],im_shape[1]))
                truth_image = cv2.warpAffine(truth_image, rotation_matrix,
                                             (im_shape[0],im_shape[1]),
                                             flags = cv2.INTER_NEAREST)
                #noise the margins that were left blank from the rotation
                noise = np.random.randint(0,256,
                                          size = len(image[image == 0]))
                image[image == 0] = noise
                image[image == -1] = 0

            if np.random.sample() <= flip_chance:
                #Discrete image rotation (0,90,180,270)
                rotation = np.random.randint(0,3)
                truth_image = np.rot90(truth_image,rotation,(0,1))
                image = np.rot90(image,rotation,(0,1))

            yield image, truth_image

def normal_image_generator(image_path_list,truth_path):
    """
    A standard image generator, to be used for testing purposes only. Goes over
    a list of image paths and outputs an image and its corresponding segmented
    image.

    Arguments:
    * image_path_list - a list of image paths
    * truth_path - a list of ground truth images
    """
    for image_path in image_path_list:
        image_name = image_path.split('/')[-1]
        truth_image_path = truth_path + '/' + image_name
        image = image_to_array(image_path)
        truth_image = image_to_array(truth_image_path)
        yield image,truth_image

def single_image_generator(image_path_list):
    """
    Generates a single image at a time for prediction purposes.

    Arguments:
    * image_path_list - a list of image paths
    """
    for image_path in image_path_list:
        image = image_to_array(image_path)
        yield image,image_path

def generate_tiles(large_image,
                   input_height = 256,input_width = 256,
                   padding = 'VALID'):
    """Uses a large image to generate smaller tiles for prediction.

    Arguments [default]:
    * large_image - a numpy array containing a large image
    * input_height - input height [256]
    * input_width - input width [256]
    * padding - whether VALID or SAME padding should be used ['VALID']
    """

    if padding == 'VALID':
        extra = 92

    stride_height = input_height - extra * 2
    stride_width = input_width - extra * 2

    stride_height,stride_width = input_height,input_width

    height,width = large_image.shape[0:2]
    h_tiles,w_tiles = floor(height/stride_height),floor(width/stride_width)
    for i in range(h_tiles - 1):
        h = i * stride_height
        for j in range(w_tiles - 1):
            w = j * stride_width
            tile = large_image[h:h + input_height,w:w + input_width,:]
            yield tile,h + extra,w + extra
        w = (j + 1) * stride_width
        if w + input_width > width:
            w = width - input_width
        tile = large_image[h:h + input_height,w:w + input_width,:]
        yield tile,h + extra,w + extra
    h = (i + 1) * stride_height
    if h + input_height > height:
        h = stride_height - input_height
    for j in range(w_tiles - 1):
        w = j * stride_width
        tile = large_image[h:h + input_height,w:w + input_width,:]
        yield tile,h + extra,w + extra
    w = (j + 1) * stride_width
    tile = large_image[h:h + input_height,w:w + input_width,:]
    yield tile,h + extra,w + extra

def remap_tiles(mask,division_mask,h_1,w_1,tile):
    """
    Function used to remap the tiles to the original image size. Currently, it
    is working with one class prediction/"channel". The division mask will be
    used in the end to divide the final_prediction and use a sort of a
    "consensus" approach for overlapping regions.

    * mask - mask with the same size of a large image
    * division_mask - mask that sums the number of positive pixels in
    overlapping regions
    * h_1 - height for the tile insertion
    * w_1 - width for the tile insertion
    * tile - numpy array with the output from U-Net
    """

    x,y = tile.shape[0:2]
    mask[h_1:h_1 + x,w_1:w_1 + y,:] += tile
    division_mask[h_1:h_1 + x,w_1:w_1 + y,:] += np.ones(tile.shape)
    return mask,division_mask

def generate_images(image_path_list,truth_path,batch_size,crop,
                    chances = [0,0,0],net_x = None,net_y = None,
                    input_height = 256,input_width = 256,resize = False,
                    resize_height = 256, resize_width = 256,
                    padding = 'VALID',n_classes = 2,
                    truth_only = False,
                    testing = False,prediction = False,
                    large_image_prediction = False):
    """
    Multi-purpose image generator.

    Arguments [default]:
    * image_path_list - a list of image paths
    * truth_path - a list of ground truth image paths
    * batch_size - the size of the batch
    * crop - whether the ground truth image should be cropped or not
    * chances - chances for the realtime_image_augmentation [[0,0,0]]
    * net_x - output height for the network [None]
    * net_y - output width for the network [None]
    * input_height - input height for the network [256]
    * input_width - input width for the network [256]
    * resize - whether the input should be resized or not [False]
    * resize_height - height of the resized input [256]
    * resize_width - width of the resized input [256]
    * padding - whether VALID or SAME padding should be used ['VALID']
    * n_classes - no. of classes [2]
    * truth_only - whether only positive images should be used [False]
    * testing - whether the testing image generator should be used
    (normal_image_generator) [False]
    * prediction - whether the prediction image generator should be used
    (single_image_generator) [False]
    * large_image_prediction - whether the large prediction image generator
    should be used (generate_tiles) [False]
    """

    a = True
    batch = []

    if testing == False and prediction == False and\
     large_image_prediction == False:
        truth_batch = []
        generator = realtime_image_augmentation(image_path_list,
                                                truth_path,
                                                chances[0],
                                                chances[1],
                                                chances[2])

    elif testing == True:
        truth_batch = []
        generator = normal_image_generator(image_path_list,
                                           truth_path)
    elif prediction == True:
        generator = single_image_generator(image_path_list)

    if prediction == False and large_image_prediction == False:
        while a == True:
            for img,truth_img in generator:
                #Normalize data between 0 - 1
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                if len(batch) >= batch_size:
                    batch = []
                    truth_batch = []

                if len(truth_img.shape) == 3:
                    truth_img = np.mean(truth_img,axis = 2)


                if resize == True:
                    truth_img = cv2.resize(truth_img,
                                           dsize = (resize_height,resize_width),
                                           interpolation = cv2.INTER_NEAREST)

                classes = np.unique(truth_img)

                mask = np.zeros((truth_img.shape[0],
                                 truth_img.shape[1],
                                 n_classes))

                for i in range(len(classes)):
                    mask[:,:,i][truth_img == classes[i]] = 1

                truth_img = mask
                if net_x != None and net_y != None:
                    x1,y1 = truth_img.shape[0:2]
                    x2,y2 = (int((x1 - net_x)/2),int((y1 - net_y)/2))
                    truth_img = truth_img[x2:x1 - x2,y2:y1 - y2,:]
                batch.append(img)
                truth_batch.append(truth_img)

                if len(batch) >= batch_size:
                    yield batch,truth_batch
            if len(batch) > 0:
                yield batch,truth_batch
            if testing == True:
                a = False

    elif prediction == True:
        batch_paths = []
        for img,img_path in generator:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            if len(batch) >= batch_size:
                batch = []
                batch_paths = []

            batch.append(img)
            batch_paths.append(img_path)

            if len(batch) >= batch_size:
                yield batch,batch_paths
        if len(batch) > 0:
            yield batch,batch_paths

    elif large_image_prediction == True:
        batch_paths = []
        batch_coord = []
        batch_shape = []
        for large_image_path in image_path_list:
            large_image = np.array(
                Image.open(large_image_path)
                )
            generator = generate_tiles(
                large_image,
                input_height = input_height,
                input_width = input_width,
                padding = padding
                )

            for img,x,y in generator:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                if len(batch) >= batch_size:
                    batch = []
                    batch_paths = []
                    batch_coord = []
                    batch_shape = []

                shape = img.shape
                batch.append(img)
                batch_paths.append(large_image_path)
                batch_coord.append((x,y))
                batch_shape.append(large_image.shape[0:2])

                if len(batch) >= batch_size:
                    yield batch,batch_paths,batch_coord,batch_shape

            if len(batch) > 0:
                yield batch,batch_paths,batch_coord,batch_shape

def iglovikov_loss(truth,network):
    """
    Loss function suggested by Iglovikov to include the Intersection of Union
    metric, a non-differentiable segmentation metric, with softmax cross
    entropy.

    Arguments [default]:
    * truth - tensor with ground truth
    * network - tensor with the output from the U-Net
    """

    network_softmax = tf.nn.softmax(network,axis = -1)

    j = tf.divide(
        tf.multiply(truth,network_softmax),
        tf.subtract(truth + network_softmax,tf.multiply(truth,network_softmax))
        )

    h = tf.losses.softmax_cross_entropy(truth,network)

    loss = tf.subtract(h,tf.reduce_mean(j))

    return loss

def get_weight_map(truth_image,w0 = 10,sigma = 5):
    """
    Has to be one channel only.

    Arguments [default]:
    * truth_image - numpy array with ground truth image
    * w0 - w0 value for the weight map [10]
    * sigma - sigma value for the weight map [5]
    """
    if len(truth_image.shape) > 2:
        truth_image = np.argmax(truth_image,axis = 2)
    sh = truth_image.shape

    truth_image[truth_image > 0] = 255
    truth_image = cv2.erode(truth_image.astype('uint8'),np.ones((3,3)))

    edges = cv2.Laplacian(truth_image,cv2.CV_64F)
    cp, edges = cv2.connectedComponents(edges.astype('uint8'))

    if cp > 1:

        dists = np.zeros((sh[0],sh[1],cp),dtype = 'float32')
        pixel_coords = np.where(truth_image == 0)
        pixel_coords_t = np.transpose(pixel_coords)
        weight_mask = np.zeros((sh[0],sh[1]),dtype = 'float32')
        ratio = len(pixel_coords[0])/(sh[0] * sh[1])
        weight_mask[pixel_coords] = 1 - ratio
        weight_mask[~pixel_coords[0],~pixel_coords[1]] = ratio

        for c in range(1,cp):
            cp_coords = np.transpose(np.where(edges == c))
            distances = distance.cdist(pixel_coords_t,cp_coords)
            mins = np.min(distances,axis = 1)
            temp_coords = (pixel_coords[0],
                           pixel_coords[1],
                           np.ones((len(pixel_coords[0]))).astype('uint') * c)
            dists[temp_coords] = mins
        dists = np.sort(dists,axis = 2,kind = 'mergesort')

        #weight_map = (dists[:,:,1] + dists[:,:,2])**2/(2 * sigma**2)
        weight_map = ((dists[:,:,1] * 2) ** 2)/(2 * sigma ** 2)

        weight_map = w0 * np.exp(-weight_map)
        weight_map[truth_image > 0] = 1
        weight_map = weight_mask + weight_map

    else:
        weight_map = np.ones(truth_image.shape) * 0.5

    weight_map = np.reshape(weight_map,(sh[0],sh[1],1))

    return weight_map

def safe_log(tensor):
    """
    Prevents log(0)

    Arguments:
    * tensor - tensor
    """
    return tf.log(tf.clip_by_value(tensor,1e-32,tf.reduce_max(tensor)))

def variables(vs):
    """
    For logging purposes.

    Arguments:
    * vs - variables from tf.all_variables() or similar functions
    """
    return int(np.sum([np.prod(v.get_shape().as_list()) for v in vs]))

def log_write_print(log_file,to_write):
    """
    Convenience funtion to write something in a log file and also print it.

    Arguments:
    * log_file - path to the file
    * to_write - what should be written/printed
    """
    with open(log_file,'a') as o:
        o.write(to_write + '\n')
    print(to_write)

def weighted_softmax_cross_entropy(prediction,truth,weights):
    """
    Function that calculates softmax cross-entropy and weighs it using a
    pixel-wise map.

    Arguments:
    * prediction - output from the network
    * truth - ground truth
    * weights - weight map
    """
    prediction = tf.nn.softmax(prediction,axis = -1)
    h = tf.add(
        tf.multiply(truth,safe_log(prediction)),
        tf.multiply(1 - truth,safe_log(1 - prediction))
        )

    weighted_cross_entropy = tf.reduce_mean(h * weights)

    return weighted_cross_entropy

def main(log_file,
         log_every_n_steps,
         save_summary_steps,
         save_summary_folder,
         save_checkpoint_steps,
         save_checkpoint_folder,
         iglovikov,
         batch_size,
         number_of_steps,
         epochs,
         beta_l2_regularization,
         learning_rate,
         factorization,
         residuals,
         weighted,
         depth_mult,
         truth_only,
         testing,
         checkpoint_path,
         prediction,
         prediction_output,
         large_image_prediction,
         large_prediction_output,
         convert_hsv,
         noise_chance,
         blur_chance,
         flip_chance,
         resize,
         resize_height,
         resize_width,
         dataset_dir,
         truth_dir,
         padding,
         extension,
         input_height,
         input_width,
         n_classes):

    """
    Wraper for the entire script.

    Arguments:
    * log_file - path to the log file
    * log_every_n_steps - how often should a log be produced
    * save_summary_steps - how often should the summary be updated
    * save_summary_folder - where shold the summary be saved
    * save_checkpoint_steps - how often should checkpoints be saved
    * save_checkpoint_folder - where should checkpoints be saved
    * iglovikov - whether the Iglovikov loss should be used
    * batch_size - number of images *per* training batch
    * number_of_steps - number of iterations
    * epochs - number of epochs (overrides number of steps)
    * beta_l2_regularization - L2 regularization factor for the loss
    * learning_rate - learning rate for the training
    * factorization - whether or not convolutions should be factorized
    * residuals - whether residual linkers in the shortcut connections should
    be used
    * weighted - whether weight maps should be calculated
    * depth_mult - a multiplier for the depth of the layers
    * truth_only - whether only positive images should be used
    * testing - whether the network is going to be used for testing
    * checkpoint_path - path to the checkpoint to be stored (for testing,
    prediction and large_prediction)
    * prediction - whether the network is going to be used for prediction
    * prediction_output - where the predicted output should be stored
    * large_image_prediction - whether the network is going to be used for
    large image prediction
    * large_prediction_output - where the large image predictions should be
    stored
    * convert_hsv - whether the images should be converted to hsv color space
    * noise_chance - probability of corrupting the image with random noise
    * blur_chance - probability of blurring the image
    * flip_chance - probability of rotating the image
    * resize - whether the input image should be resized
    * resize_height - height of the resized image
    * resize_width - width of the resized image
    * dataset_dir - directory containing the image dataset
    * truth_dir - directory containing the truth images
    * padding - whether the images should have 'VALID' or 'SAME' padding
    * extension - extension for the images
    * input_height - height of the input image
    * input_width - width of the input image
    * n_classes - no. of classes (currently only supports 2)
    """

    print("Preparing the network...\n")

    inputs = tf.placeholder(tf.float32, [None,input_height,input_width,3])
    if convert_hsv == True:
        inputs = tf.image.rgb_to_hsv(inputs)

    network, endpoints = u_net(inputs,
                               final_endpoint = 'Final',
                               padding = padding,
                               factorization = factorization,
                               residuals = residuals,
                               beta = beta_l2_regularization,
                               n_classes = n_classes,
                               resize = resize,
                               resize_height = resize_height,
                               resize_width = resize_width,
                               depth_mult = depth_mult)
    network = network + 1e-10

    log_write_print(log_file,
                    'Total parameters: {0:d} (trainable: {1:d})\n'.format(
                        variables(tf.all_variables()),
                        variables(tf.trainable_variables())
                        ))

    if padding == 'VALID':
        net_x,net_y = network.get_shape().as_list()[1:3]
        tf_shape = [None,net_x,net_y,n_classes]
        truth = tf.placeholder(tf.float32, tf_shape)
        crop = True
        weight_x,weight_y = net_x,net_y

    else:
        net_x,net_y = (None, None)
        if resize == True:
            tf_shape = [None,resize_height,resize_width,n_classes]
        else:
            tf_shape = [None,input_height,input_width,n_classes]
        truth = tf.placeholder(tf.float32, tf_shape)
        crop = False
        weight_x, weight_y = input_height,input_width

    saver = tf.train.Saver()

    truth_max = tf.argmax(truth,axis = 3)
    network_max = tf.argmax(network,axis = 3)
    truth_binary = tf.clip_by_value(
        tf.count_nonzero(truth_max,axis = [1,2],
                         dtype = tf.float32),0,1
        )
    network_binary = tf.clip_by_value(
        tf.count_nonzero(network_max,axis = [1,2],
                         dtype = tf.float32),0,1
        )

    if weighted == True:
        weights = tf.placeholder(tf.float32, [None,weight_x,weight_y,1])
    else:
        weights = None

    if iglovikov == True:
        loss = iglovikov_loss(truth,network)

    elif iglovikov == False and weighted == True:
        loss = weighted_softmax_cross_entropy(network,truth,weights)
    else:
        loss = tf.losses.softmax_cross_entropy(truth,network)

    if beta_l2_regularization > 0:
        loss = loss + tf.add_n(slim.losses.get_regularization_losses())

    #Evaluation metrics
    accuracy,accuracy_op = tf.metrics.accuracy(truth_max,network_max)
    mean_iou, mean_iou_op = tf.metrics.mean_iou(truth_max,network_max,
                                                num_classes = n_classes)
    recall,recall_op = tf.metrics.recall(truth_max,network_max)
    precision,precision_op = tf.metrics.precision(truth_max,network_max)

    train_op = tf.train.MomentumOptimizer(learning_rate,0.99).minimize(loss)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    for endpoint in endpoints:
        x = endpoints[endpoint]
        summaries.add(tf.summary.histogram('activations/' + endpoint, x))

    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    summaries.add(tf.summary.scalar('loss', loss))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
        )

    image_path_list = glob(dataset_dir + '/*' + extension)

    tf.set_random_seed(0)
    np.random.seed(0)

    if len(image_path_list) > 0:

        log_write_print(log_file,'INPUT ARGUMENTS:')
        for var in vars(args):
            log_write_print(log_file,'\t{0}={1}'.format(var,vars(args)[var]))
        print('\n')

        if testing == False and prediction == False and\
         large_image_prediction == False:

            tmp = []

            for image_path in image_path_list:
                truth_image_path = os.sep.join([
                    truth_dir,
                    image_path.split(os.sep)[-1]])
                image = np.array(Image.open(truth_image_path))
                if image.shape[0] == input_height and image.shape[1] == input_width:
                    if truth_only == True:
                        if len(np.unique(image)) == n_classes:
                            tmp.append(image_path)
                    else:
                        tmp.append(image_path)

            image_path_list = tmp

            print("Found {0:d} images in {1!s}".format(
                len(image_path_list),dataset_dir
            ))

            print("Training the network...\n")
            LOG = 'Step {0:d}: minibatch loss: {1:f}. '
            LOG += 'Average time/minibatch = {2:f}s. '
            LOG += 'Accuracy: {3:f}; Mean IOU: {4:f}; '
            LOG += 'Recall: {5:f}; Precision: {6:f}'
            SUMMARY = 'Step {0:d}: summary stored in {1!s}'
            CHECKPOINT = 'Step {0:d}: checkpoint stored in {1!s}'
            CHECKPOINT_PATH = os.path.join(save_checkpoint_folder,
                                           'my_u-net.ckpt')

            if epochs != None:
                number_of_steps = epochs * int(len(image_path_list)/batch_size)

            try:
                os.makedirs(save_checkpoint_folder)
            except:
                pass

            try:
                os.makedirs(save_summary_folder)
            except:
                pass

            config = tf.ConfigProto()
            local_dev = device_lib.list_local_devices()
            n_gpu = len([x.name for x in local_dev if x.device_type == 'GPU'])

            if n_gpu > 0:
                config.gpu_options.polling_inactive_delay_msecs = 2000
            else:
                n_phys_cores = psutil.cpu_count(logical = False)
                config.intra_op_parallelism_threads = n_phys_cores
                config.inter_op_parallelism_threads = n_phys_cores

            with tf.Session(config = config) as sess:
                writer = tf.summary.FileWriter(save_summary_folder,sess.graph)

                sess.run(init)

                pre_processing_chances = [noise_chance,blur_chance,flip_chance]
                image_generator = generate_images(
                    image_path_list,
                    truth_dir,
                    chances = pre_processing_chances,
                    batch_size = batch_size,
                    crop = crop,
                    net_x = net_x,
                    net_y = net_y,
                    input_height = input_height,
                    input_width = input_width,
                    resize = resize,
                    resize_height = resize_height,
                    resize_width = resize_width,
                    n_classes = n_classes,
                    truth_only = truth_only
                    )

                time_list = []
                for i in range(1,number_of_steps + 1):
                    batch, truth_batch = next(image_generator,(None,None))
                    if weighted == True:
                        tmp_weights = [get_weight_map(t) for t in truth_batch]

                        batch = np.stack(batch,0)
                        truth_batch = np.stack(truth_batch,0)
                        tmp_weights = np.stack(tmp_weights,0)
                        a = time.perf_counter()
                        _, l, summary,_,_,_,_ = sess.run(
                            [train_op,loss,summary_op,
                             accuracy_op,mean_iou_op,
                             recall_op,precision_op],
                            feed_dict = {truth:truth_batch,
                                         inputs:batch,
                                         weights: tmp_weights})

                        b = time.perf_counter()
                        time_list.append(b - a)

                    else:
                        batch = np.stack(batch,0)
                        truth_batch = np.stack(truth_batch,0)
                        a = time.perf_counter()
                        _, l, summary,_,_,_,_ = sess.run(
                            [train_op,loss,summary_op,
                             accuracy_op,mean_iou_op,
                             recall_op,precision_op],
                            feed_dict = {truth:truth_batch,
                                         inputs:batch})

                        b = time.perf_counter()
                        time_list.append(b - a)

                    if i % log_every_n_steps == 0 or i == 1 or\
                     i % number_of_steps == 0:
                        acc,iou,rec,pre = sess.run([accuracy,mean_iou,
                                                    recall,precision])
                        log_write_print(log_file,
                                        LOG.format(i,l,np.mean(time_list),
                                                   acc,iou,rec,pre))
                        time_list = []

                    if i % save_summary_steps == 0 or\
                     i % number_of_steps == 0:
                        writer.add_summary(summary,i)
                        log_write_print(log_file,
                                        SUMMARY.format(i,save_summary_folder))

                    if i % save_checkpoint_steps == 0 or\
                     i % number_of_steps == 0:
                        saver.save(sess, CHECKPOINT_PATH,global_step = i)
                        log_write_print(log_file,
                                        CHECKPOINT.format(i,CHECKPOINT_PATH))

        elif testing == True and os.path.exists(checkpoint_path + '.index'):

            LOG = 'Time/{0:d} images: {1:f}s (time/1 image: {2:f}s). '
            LOG += 'Accuracy: {3:f}; Mean IOU: {4:f}; '
            LOG += 'Recall: {5:f}; Precision: {6:f}'

            FINAL_LOG = 'Final averages - time/image: {0:f}s; accuracy: {1:f}; '
            FINAL_LOG += 'mean IOU: {2:f}; recall: {3:f}; precision: {4:f}'

            print('Testing...')
            with tf.Session() as sess:
                image_generator = generate_images(
                    image_path_list,
                    truth_dir,
                    batch_size = batch_size,
                    crop = crop,
                    net_x = net_x,
                    net_y = net_y,
                    input_height = input_height,
                    input_width = input_width,
                    resize = resize,
                    resize_height = resize_height,
                    resize_width = resize_width,
                    n_classes = n_classes,
                    testing = True
                    )

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                all_accuracy = []
                all_mean_iou = []
                all_recall = []
                all_precision = []
                time_list = []

                for batch,truth_batch in image_generator:
                    n_images = len(batch)
                    batch = np.stack(batch,0)
                    truth_batch = np.stack(truth_batch,0)

                    a = time.perf_counter()
                    sess.run([network],
                             feed_dict = {truth:truth_batch,inputs:batch})
                    b = time.perf_counter()
                    t_image = (b - a)/n_images
                    time_list.append(t_image)

                    sess.run([accuracy_op,mean_iou_op,recall_op,precision_op],
                             feed_dict = {truth:truth_batch,inputs:batch})

                    acc,iou,rec,pre = sess.run([accuracy,mean_iou,
                                                recall,precision])

                    all_accuracy.append(acc)
                    all_mean_iou.append(iou)
                    all_recall.append(rec)
                    all_precision.append(pre)

                    output = LOG.format(n_images,b - a,t_image,acc,iou,rec,pre)
                    log_write_print(log_file,output)

                avg_time = np.mean(time_list)
                avg_accuracy = np.mean(all_accuracy)
                avg_mean_iou = np.mean(all_mean_iou)
                avg_recall = np.mean(all_recall)
                avg_precision = np.mean(all_precision)

                output = FINAL_LOG.format(avg_time,avg_accuracy,avg_mean_iou,
                                          avg_recall,avg_precision)
                log_write_print(log_file,output)

        elif prediction == True and os.path.exists(checkpoint_path + '.index'):
            print('Predicting...')

            LOG = 'Time/{0:d} images: {1:f}s (time/1 image: {2:f}s).'
            FINAL_LOG = 'Average time/image: {0:f}'

            with tf.Session() as sess:

                try:
                    os.makedirs(prediction_output)
                except:
                    pass

                image_generator = generate_images(
                    image_path_list,
                    truth_dir,
                    batch_size = batch_size,
                    crop = crop,
                    net_x = net_x,
                    net_y = net_y,
                    input_height = input_height,
                    input_width = input_width,
                    resize = resize,
                    resize_height = resize_height,
                    resize_width = resize_width,
                    n_classes = n_classes,
                    testing = False,
                    prediction = True
                    )

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                time_list = []

                for batch,image_names in image_generator:
                    n_images = len(batch)
                    batch = np.stack(batch,0)

                    a = time.perf_counter()
                    prediction = sess.run(network_max,
                                          feed_dict = {inputs:batch})
                    b = time.perf_counter()
                    t_image = (b - a)/n_images
                    time_list.append(t_image)

                    output = LOG.format(n_images,b - a,t_image)
                    log_write_print(log_file,output)

                    for i in range(prediction.shape[0]):
                        image = prediction[i,:,:]
                        image = np.stack((image,image,image),axis = 2)
                        image_name = image_names[i].split(os.sep)[-1]
                        image_output = os.path.join(prediction_output,
                                                    image_name)
                        image_handle = Image.fromarray(image.astype('uint8'))
                        image_handle.save(image_output)

                avg_time = np.mean(time_list)
                output = FINAL_LOG.format(avg_time)
                log_write_print(log_file,output)

        elif large_image_prediction == True and\
         os.path.exists(checkpoint_path + '.index'):
            print('Predicting large image...')

            LOG = 'Time/{0:d} images: {1:f}s (time/1 image: {2:f}s).'
            FINAL_LOG = 'Average time/image: {0:f}.\nTotal stats: {1:f}s '
            FINAL_LOG += 'for {2:d} images.'
            start = time.perf_counter()
            with tf.Session() as sess:

                try:
                    os.makedirs(large_prediction_output)
                except:
                    pass

                if prediction_output != 'no_path':
                    try:
                        os.makedirs(prediction_output)
                    except:
                        pass

                image_generator = generate_images(
                    image_path_list,
                    truth_dir,
                    crop = False,
                    batch_size = batch_size,
                    input_height = input_height,
                    input_width = input_width,
                    padding = padding,
                    large_image_prediction = True
                )

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                time_list = []

                curr_image_name = ''

                for batch,image_names,coords,shapes in image_generator:
                    n_images = len(batch)
                    batch = np.stack(batch,0)

                    a = time.perf_counter()
                    prediction = sess.run(network,
                                          feed_dict = {inputs:batch})
                    b = time.perf_counter()
                    t_image = (b - a)/n_images
                    time_list.append(t_image)

                    output = LOG.format(n_images,b - a,t_image)
                    log_write_print(log_file,output)

                    for i in range(prediction.shape[0]):
                        image_name = image_names[i].split(os.sep)[-1]
                        if image_name != curr_image_name:
                            if curr_image_name != '':
                                division_mask[division_mask == 0] = 1
                                image = np.argmax(mask/division_mask,axis = 2)
                                image = np.stack((image,image,image),axis = 2)
                                image = image.astype('uint8')
                                image = Image.fromarray(image)
                                image.save(large_image_output_name)
                            curr_image_name = image_name
                            final_height,final_width = shapes[i][0:2]
                            if padding == 'VALID':
                                final_height = final_height - 184
                                final_width = final_width - 184
                            mask = np.zeros(
                                (final_height,final_width,n_classes)
                            )
                            division_mask = np.zeros(
                                (final_height,final_width,n_classes)
                            )
                            large_image_output_name = os.path.join(
                                large_prediction_output,
                                curr_image_name
                                )
                        h_1,w_1 = coords[i]
                        tile = prediction[i,:,:]
                        remap_tiles(mask,division_mask,h_1,w_1,tile)

                    division_mask[division_mask == 0] = 1
                    image = np.argmax(mask/division_mask,axis = 2)
                    image[image >= 0.5] = 1
                    image[image < 0.5] = 0
                    image = np.stack((image,image,image),axis = 2)
                    image = image.astype('uint8')
                    image = Image.fromarray(image)
                    image.save(large_image_output_name)

                finish = time.perf_counter()
                avg_time = np.mean(time_list)
                output = FINAL_LOG.format(avg_time,finish - start,
                                          len(image_path_list))
                log_write_print(log_file,output)

#Defining arguments

parser = argparse.ArgumentParser(
    prog = 'u-net.py',
    description = 'Multi-purpose U-Net implementation.'
)

#Logs
parser.add_argument('--log_file',dest = 'log_file',
                    action = ToDirectory,type = str,
                    default = os.getcwd() + '/log.txt',
                    help = 'Directory where training logs are written.')
parser.add_argument('--log_every_n_steps',dest = 'log_every_n_steps',
                    action = 'store',type = int,
                    default = 100,
                    help = 'How often are the loss and global step logged.')

#Summaries
parser.add_argument('--save_summary_steps',dest = 'save_summary_steps',
                    action = 'store',type = int,
                    default = 100,
                    metavar = '',
                    help = 'How often summaries are saved.')
parser.add_argument('--save_summary_folder',dest = 'save_summary_folder',
                    action = ToDirectory,type = str,
                    default = os.getcwd(),
                    help = 'Directory where summaries are saved.')

#Checkpoints
parser.add_argument('--save_checkpoint_steps',dest = 'save_checkpoint_steps',
                    action = 'store',type = int,
                    default = 100,
                    help = 'How often checkpoints are saved.')
parser.add_argument('--save_checkpoint_folder',dest = 'save_checkpoint_folder',
                    action = ToDirectory,type = str,
                    default = os.getcwd(),
                    metavar = '',
                    help = 'Directory where checkpoints are saved.')

#Training
parser.add_argument('--iglovikov',dest = 'iglovikov',
                    action = 'store_true',
                    default = False,
                    help = 'Use Iglovikov loss function.')
parser.add_argument('--batch_size',dest = 'batch_size',
                    action = 'store',type = int,
                    default = 100,
                    help = 'Size of mini batch.')
parser.add_argument('--number_of_steps',dest = 'number_of_steps',
                    action = 'store',type = int,
                    default = 5000,
                    help = 'Number of steps in the training process.')
parser.add_argument('--epochs',dest = 'epochs',
                    action = 'store',type = int,
                    default = None,
                    help = 'Number of epochs (overrides number_of_steps).')
parser.add_argument('--beta_l2_regularization',dest = 'beta_l2_regularization',
                    action = 'store',type = float,
                    default = 0,
                    help = 'Beta parameter for L2 regularization.')
parser.add_argument('--learning_rate',dest = 'learning_rate',
                    action = 'store',type = float,
                    default = 0.001,
                    help = 'Learning rate for the SGD optimizer.')
parser.add_argument('--factorization',dest = 'factorization',
                    action = 'store_true',
                    default = False,
                    help = 'Use convolutional layer factorization.')
parser.add_argument('--residuals',dest = 'residuals',
                    action = 'store_true',
                    default = False,
                    help = 'Use residuals in skip connections.')
parser.add_argument('--weighted',dest = 'weighted',
                    action = 'store_true',
                    default = False,
                    help = 'Calculates weighted cross entropy.')
parser.add_argument('--depth_mult',dest = 'depth_mult',
                    action = 'store',type = float,
                    default = 1.,
                    help = 'Change the number of channels in all layers.')
parser.add_argument('--truth_only',dest = 'truth_only',
                    action = 'store_true',
                    default = False,
                    help = 'Consider only images with all classes.')


#Testing
parser.add_argument('--testing',dest = 'testing',
                    action = 'store_true',
                    default = False,
                    help = 'Used to test the algorithm.')
parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path to checkpoint for testing.')

#Prediction
parser.add_argument('--prediction',dest = 'prediction',
                    action = 'store_true',
                    default = False,
                    help = 'Used to predict new entries.')
parser.add_argument('--prediction_output',dest = 'prediction_output',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path where image predictions are stored.')

#Large image prediction
parser.add_argument('--large_image_prediction',dest = 'large_image_prediction',
                    action = 'store_true',
                    default = False,
                    help = 'Predict large images after tiling.')
parser.add_argument('--large_prediction_output',
                    dest = 'large_prediction_output',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path to store large image predictions.')

#Pre-processing
parser.add_argument('--convert_hsv',dest = 'convert_hsv',
                    action = 'store_true',
                    default = False,
                    help = 'Converts RGB input to HSV.')
parser.add_argument('--noise_chance',dest = 'noise_chance',
                    action = 'store',type = int,
                    default = 0.1,
                    help = 'Probability to add noise.')
parser.add_argument('--blur_chance',dest = 'blur_chance',
                    action = 'store',type = int,
                    default = 0.05,
                    help = 'Probability to blur the input image.')
parser.add_argument('--flip_chance',dest = 'flip_chance',
                    action = 'store',type = int,
                    default = 0.2,
                    help = 'Probability to flip/rotate the image.')
parser.add_argument('--resize',dest = 'resize',
                    action = 'store_true',
                    default = False,
                    help = 'Resize images to input_height and input_width.')
parser.add_argument('--resize_height',dest = 'resize_height',
                    action = 'store',
                    default = 256,
                    help = 'Height for resized images.')
parser.add_argument('--resize_width',dest = 'resize_width',
                    action = 'store',
                    default = 256,
                    help = 'Height for resized images.')

#Dataset
parser.add_argument('--dataset_dir',dest = 'dataset_dir',
                    action = ToDirectory,type = str,
                    required = True,
                    help = 'Directory where the training set is stored.')
parser.add_argument('--truth_dir',dest = 'truth_dir',
                    action = ToDirectory,type = str,
                    help = 'Path to segmented images.')
parser.add_argument('--padding',dest = 'padding',
                    action = 'store',
                    default = 'VALID',
                    help = 'Define padding.',
                    choices = ['VALID','SAME'])
parser.add_argument('--extension',dest = 'extension',
                    action = 'store',type = str,
                    default = '.png',
                    help = 'The file extension for all images.')
parser.add_argument('--input_height',dest = 'input_height',
                    action = 'store',type = int,
                    default = 256,
                    help = 'The file extension for all images.')
parser.add_argument('--input_width',dest = 'input_width',
                    action = 'store',type = int,
                    default = 256,
                    help = 'The file extension for all images.')
parser.add_argument('--n_classes',dest = 'n_classes',
                    action = 'store',type = int,
                    default = 2,
                    help = 'Number of classes in the segmented images.')

args = parser.parse_args()

#Logs
log_file = args.log_file
log_every_n_steps = args.log_every_n_steps

#Summaries
save_summary_steps = args.save_summary_steps
save_summary_folder = args.save_summary_folder

#Checkpoints
save_checkpoint_steps = args.save_checkpoint_steps
save_checkpoint_folder = args.save_checkpoint_folder

#Training
iglovikov = args.iglovikov
batch_size = args.batch_size
number_of_steps = args.number_of_steps
epochs = args.epochs
beta_l2_regularization = args.beta_l2_regularization
learning_rate = args.learning_rate
factorization = args.factorization
residuals = args.residuals
weighted = args.weighted
depth_mult = args.depth_mult
truth_only = args.truth_only

#Testing
testing = args.testing
checkpoint_path = args.checkpoint_path

#Prediction
prediction = args.prediction
prediction_output = args.prediction_output

#Large image prediction
large_image_prediction = args.large_image_prediction
large_prediction_output = args.large_prediction_output

#Pre-processing
convert_hsv = args.convert_hsv
noise_chance = args.noise_chance
blur_chance = args.blur_chance
flip_chance = args.flip_chance
resize = args.resize
resize_height = args.resize_height
resize_width = args.resize_width

#Dataset
dataset_dir = args.dataset_dir
truth_dir = args.truth_dir
padding = args.padding
extension = args.extension
input_height = args.input_height
input_width = args.input_width
n_classes = args.n_classes

if __name__ == '__main__':
    print("Loading dependencies...")

    import sys
    import time
    from glob import glob
    from math import floor
    import numpy as np
    import cv2
    import tensorflow as tf
    import psutil
    from tensorflow.python.client import device_lib
    from PIL import Image
    from scipy.spatial import distance

    tf.logging.set_verbosity(tf.logging.ERROR)
    slim = tf.contrib.slim
    variance_scaling_initializer =\
     tf.contrib.layers.variance_scaling_initializer
    main(log_file = log_file,
         log_every_n_steps = log_every_n_steps,
         save_summary_steps = save_summary_steps,
         save_summary_folder = save_summary_folder,
         save_checkpoint_steps = save_checkpoint_steps,
         save_checkpoint_folder = save_checkpoint_folder,
         iglovikov = iglovikov,
         batch_size = batch_size,
         number_of_steps = number_of_steps,
         epochs = epochs,
         beta_l2_regularization = beta_l2_regularization,
         learning_rate = learning_rate,
         factorization = factorization,
         residuals = residuals,
         weighted = weighted,
         depth_mult = depth_mult,
         truth_only = truth_only,
         testing = testing,
         checkpoint_path = checkpoint_path,
         prediction = prediction,
         prediction_output = prediction_output,
         large_image_prediction = large_image_prediction,
         large_prediction_output = large_prediction_output,
         convert_hsv = convert_hsv,
         noise_chance = noise_chance,
         blur_chance = blur_chance,
         flip_chance = flip_chance,
         resize = resize,
         resize_height = resize_height,
         resize_width = resize_width,
         dataset_dir = dataset_dir,
         truth_dir = truth_dir,
         padding = padding,
         extension = extension,
         input_height = input_height,
         input_width = input_width,
         n_classes = n_classes)
