import os
import numpy as np
from math import inf
import cv2
import tensorflow as tf
from scipy.spatial import distance
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)
slim = tf.contrib.slim

"""
Deep learning/TF-related operations.
"""

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

def u_net(inputs,
          final_endpoint = None,
          padding = 'VALID',
          factorization = False,
          beta = 0,
          residuals = False,
          n_classes = 2,
          resize = False,
          resize_height = 256,
          resize_width = 256,
          depth_mult = 1,
          is_training = True):

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

    def pp_image(image):
        """
        Function to preprocess images with data augmentation - only performs
        operations that alter brightness, saturation, hue and contrast.
        """
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

        color_ordering = tf.random_uniform([],0,6,tf.int32)
        image = distort_colors(image,color_ordering)
        image = tf.clip_by_value(image,0.,1.)
        return image

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
        x = slim.dropout(x,keep_prob=0.8,is_training=is_training)
        return x

    def block(x,depth,size,stride,padding,factorization = False):
        x = conv2d(x,depth,size,stride,'conv2d_1_',factorization)
        x = conv2d(x,depth,size,stride,'conv2d_2_',factorization)
        return x

    def red_block_wrapper(net,depth,factorization,residuals,endpoints,
                          name):
        net = block(net,depth,3,1,factorization)
        if residuals == True:
            endpoints[name] = residual_block(
                net,depth,3,factorization
            )
        else:
            endpoints[name] = net

        net = slim.max_pool2d(net,[2,2],stride = 2,
                              scope = 'maxpool2d_2x2')
        return net,endpoints

    def rec_block_wrapper(net,depth,factorization,
                          residuals,endpoints,padding,
                          name,red_block_name):
        new_size = net.get_shape().as_list()
        new_size = [new_size[1] * 2,new_size[2] * 2]
        net = tf.image.resize_nearest_neighbor(net,new_size)
        net = slim.conv2d(net,depth,[2,2],scope = 'conv2d_0_',
                          padding = 'SAME')
        if padding == 'VALID':
            n = endpoints[red_block_name].get_shape().as_list()[1]
            n = (n - net.get_shape().as_list()[1])/2
            concat_net = crop(endpoints[red_block_name],n)
        else:
            concat_net = endpoints[red_block_name]
        net = tf.concat([net,concat_net],
                        axis = 3)
        net = block(net,d_current,3,1,padding,factorization)
        endpoints[name] = net

        return net,endpoints

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

    def fc_layer_classification(net,depth,n_classes,
                                **kwargs):
        # This node is used to classify the presence/absence of objects
        # in the bottleneck. This forces the network to collect
        # relevant features along the whole network.
        with tf.variable_scope('Aux_Node',None,[net]):
            with slim.arg_scope([slim.fully_connected],
                                **kwargs):
                pre_class = slim.fully_connected(
                    net,
                    num_outputs=depth,
                    activation_fn=None,
                    scope='pre_class'
                )
                classification = slim.fully_connected(
                    pre_class,
                    num_outputs=int(n_classes),
                    activation_fn=None,
                    scope='classification'
                )
        return classification

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

    classifications = []

    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu,
        padding=padding,
        weights_regularizer=weights_regularizer,
        #normalizer_fn=pixel_normalization
        normalizer_fn=slim.batch_norm,
        normalizer_params={"is_training":is_training}
        ):

        with tf.variable_scope('U-net', None, [inputs]):

            if resize == True:
                inputs = tf.image.resize_images(
                    inputs,
                    [resize_height,resize_width],
                    method = tf.image.ResizeMethod.BILINEAR)

            if is_training == True:
                inputs = tf.map_fn(pp_image,
                                   inputs,
                                   dtype=tf.float32)
            else:
                inputs = tf.cast(inputs,tf.float32)

            inputs = slim.batch_norm(inputs,is_training=is_training)

            with tf.variable_scope('Red_Operations',None,[inputs]):
                with tf.variable_scope('Red_Block_1',None):
                    net,endpoints = red_block_wrapper(
                        inputs,
                        depth=int(64 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        name='Red_Block_1')

                with tf.variable_scope('Red_Block_2',None,[net]):
                    net,endpoints = red_block_wrapper(
                        net,
                        depth=int(128 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        name='Red_Block_2')

                with tf.variable_scope('Red_Block_3',None,[net]):
                    net,endpoints = red_block_wrapper(
                        net,
                        depth=int(256 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        name='Red_Block_3')

                with tf.variable_scope('Red_Block_4',None,[net]):
                    net,endpoints = red_block_wrapper(
                        net,
                        depth=int(512 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        name='Red_Block_4')

                with tf.variable_scope('Red_Block_5',None,[net]):
                    d_current = int(1024 * depth_mult)
                    net = block(net,d_current,3,1,factorization)
                    endpoints['Red_Block_5'] = net
                    if final_endpoint == 'Red_Block_5':
                        return net,endpoints

                    flat_bottleneck = tf.reduce_max(net,axis=[1,2])
                    classification = fc_layer_classification(
                        flat_bottleneck,
                        depth=256,
                        n_classes=1,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=weights_regularizer,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={"is_training":is_training}
                    )

                endpoints['Classification'] = classification
                classifications.append(classification)
                if final_endpoint == 'Classification':
                    return net,endpoints

            with tf.variable_scope('Rec_Operations',None,[net]):
                with tf.variable_scope('Rec_Block_1',None,[net]):
                    net,endpoints = rec_block_wrapper(
                        net,
                        depth=int(512 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        padding=padding,
                        name='Rec_Block_1',
                        red_block_name='Red_Block_4')

                with tf.variable_scope('Rec_Block_2',None,[net]):
                    net,endpoints = rec_block_wrapper(
                        net,
                        depth=int(256 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        padding=padding,
                        name='Rec_Block_2',
                        red_block_name='Red_Block_3')

                with tf.variable_scope('Rec_Block_3',None,[net]):
                    net,endpoints = rec_block_wrapper(
                        net,
                        depth=int(128 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        padding=padding,
                        name='Rec_Block_3',
                        red_block_name='Red_Block_2')

                with tf.variable_scope('Rec_Block_4',None,[net]):
                    net,endpoints = rec_block_wrapper(
                        net,
                        depth=int(64 * depth_mult),
                        factorization=factorization,
                        residuals=residuals,
                        endpoints=endpoints,
                        padding=padding,
                        name='Rec_Block_4',
                        red_block_name='Red_Block_1')

                with tf.variable_scope('Final',None,[net]):
                    net = slim.conv2d(net, n_classes, [1, 1],
                                      normalizer_fn = None,
                                      activation_fn = None,
                                      scope='conv2d_0_sigmoid')

                    endpoints['Final'] = net
                    if final_endpoint == 'Final':
                        return net,endpoints
    return net,endpoints,classifications

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

"""
Image and image processing-related operations. Image generators.
"""

def masked_mean(arr,mask):
    masked_arr = arr * mask
    masked_mean = np.sum(masked_arr) / np.sum(mask)
    return masked_mean

def image_to_array(image_path):
    """Opens image in image_path and returns it as a numpy array.

    Arguments:
    * image_path - the path to an image
    """

    with Image.open(image_path) as o:
        return np.array(o)

def realtime_image_augmentation(image_list,truth_list,weight_map_list,
                                classification_list=None,
                                noise_chance = 0.05,blur_chance = 0.05,
                                flip_chance = 0.5):
    """A generator that performs a series of image manipulations to an original
    image using salt, pepper, channel dropout, gaussian noise, blur, rotation
    and flipping.
    Additionally, it normalizes the input image through linear stretching and
    standardization.

    Arguments:
    * image_list - a list of image paths
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

    while True:
        indexes = np.arange(0,len(image_list))
        np.random.shuffle(indexes)

        for index in indexes:
            image = image_list[index]
            if truth_list != None:
                truth_image = truth_list[index]
            else:
                truth_image = None

            if weight_map_list != None:
                weight_map = weight_map_list[index]
            else:
                weight_map = None

            if classification_list != None:
                classification = classification_list[index]
            else:
                classification = None

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
                    image = np.flip(image,flip)
                    if truth_list != None:
                        truth_image = np.flip(truth_image,flip)
                    if weight_map_list != None:
                        weight_map = np.flip(weight_map,flip)
                if np.random.sample() <= flip_chance:
                    #Continuous image rotation (0,89)
                    rotation = np.random.sample() * 89
                    rot_tup = (im_shape[0]/2, im_shape[1]/2)
                    rotation_matrix = cv2.getRotationMatrix2D(rot_tup,
                                                              rotation, 1)
                    image = cv2.warpAffine(image, rotation_matrix,
                                           (im_shape[0],im_shape[1]))
                    if truth_list != None:
                        truth_image = cv2.warpAffine(truth_image,
                                                     rotation_matrix,
                                                     (im_shape[0],im_shape[1]),
                                                     flags = cv2.INTER_NEAREST)
                    if weight_map_list != None:
                        weight_map = cv2.warpAffine(weight_map,
                                                    rotation_matrix,
                                                    (im_shape[0],im_shape[1]),
                                                    flags = cv2.INTER_NEAREST)
                    #noise the margins that were left blank from the rotation

                if np.random.sample() <= flip_chance:
                    #Discrete image rotation (0,90,180,270)
                    rotation = np.random.randint(0,3)
                    image = np.rot90(image,rotation,(0,1))
                    if truth_list != None:
                        truth_image = np.rot90(truth_image,rotation,(0,1))
                    if weight_map_list != None:
                        weight_map = np.rot90(weight_map,rotation,(0,1))
                if weight_map_list != None:
                    wms = weight_map.shape
                    weight_map = np.reshape(weight_map,(wms[0],wms[1]))

                yield image, truth_image, weight_map, classification

def normal_image_generator(image_list,truth_list):
    """
    A standard image generator, to be used for testing purposes only. Goes over
    a list of image paths and outputs an image and its corresponding segmented
    image.

    Arguments:
    * image_path_list - a list of image paths
    * truth_path - a list of ground truth images
    """
    indexes = np.arange(0,len(image_list))
    for i in indexes:
        image = image_list[i]
        truth_image = truth_list[i]
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
                    weight_maps = True,
                    mode = 'train'):
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
    * mode - algorithm mode [train].
    """

    a = True
    batch = []

    if mode == 'train':
        truth_batch = []

        image_list = []
        truth_list = []
        weight_map_list = []

        for i,image_path in enumerate(image_path_list):
            class_array = np.array([-1 for i in range(n_classes)])
            if i % 5 == 0: print(i)
            image_name = image_path.split(os.sep)[-1]
            truth_image_path = truth_path + os.sep + image_name
            truth_img = image_to_array(truth_image_path)

            #assigns a class for edges
            if len(truth_img.shape) == 3:
                truth_img = np.mean(truth_img,axis=2)
            if n_classes == 3:
                edges = cv2.Canny(truth_img.astype('uint8'),100,200)
                edges = cv2.dilate(edges,
                                   kernel=np.ones([3,3]),
                                   iterations=1)
                #assigns one channel per class
                classes = np.unique(truth_img)
                for index,label in enumerate(classes):
                    class_array[index] = label
                mask = np.zeros((truth_img.shape[0],
                                 truth_img.shape[1],
                                 n_classes))
                for j in range(len(classes)):
                    mask[:,:,j][truth_img == classes[j]] = 1
                mask[:,:,len(classes) - 1][[edges > 0]] = 0
                mask[:,:,len(classes)][edges > 0] = 1

            if n_classes == 2:
                #assigns one channel per class
                classes = np.unique(truth_img)
                for index,label in enumerate(classes):
                    class_array[index] = label
                classes = class_array
                mask = np.zeros((truth_img.shape[0],
                                 truth_img.shape[1],
                                 n_classes))
                for j in range(n_classes):
                    mask[:,:,j][truth_img == classes[j]] = 1
            truth_img = mask

            if resize == True:
                    truth_img = cv2.resize(
                        truth_img,
                        dsize = (resize_height,resize_width),
                        interpolation = cv2.INTER_NEAREST)
            weight_map = get_weight_map(truth_img)
            dist_weight_map = get_near_weight_map(truth_img,w0=5,sigma=20)
            weight_map = weight_map + dist_weight_map
            image_list.append(image_to_array(image_path))
            truth_list.append(truth_img)
            weight_map_list.append(weight_map)

        generator = realtime_image_augmentation(
            image_list=image_list,
            truth_list=truth_list,
            weight_map_list=weight_map_list,
            noise_chance=chances[0],
            blur_chance=chances[1],
            flip_chance=chances[2])

    elif mode == 'test':
        truth_batch = []

        image_list = []
        truth_list = []

        for i,image_path in enumerate(image_path_list):
            if i % 5 == 0: print(i)
            image_name = image_path.split(os.sep)[-1]
            truth_image_path = truth_path + os.sep + image_name
            truth_img = image_to_array(truth_image_path)

            #assigns a class for edges
            if len(truth_img.shape) == 3:
                truth_img = np.mean(truth_img,axis=2)
            if n_classes == 3:
                edges = cv2.Canny(truth_img.astype('uint8'),100,200)
                edges = cv2.dilate(edges,
                                   kernel=np.ones([3,3]),
                                   iterations=1)
                #assigns one channel per class
                classes = np.unique(truth_img)
                mask = np.zeros((truth_img.shape[0],
                                 truth_img.shape[1],
                                 n_classes))
                for j in range(len(classes)):
                    mask[:,:,j][truth_img == classes[j]] = 1
                mask[:,:,len(classes) - 1][[edges > 0]] = 0
                mask[:,:,len(classes)][edges > 0] = 1

            if n_classes == 2:
                #assigns one channel per class
                classes = np.unique(truth_img)
                mask = np.zeros((truth_img.shape[0],
                                 truth_img.shape[1],
                                 n_classes))
                for j in range(len(classes)):
                    mask[:,:,j][truth_img == classes[j]] = 1
            truth_img = mask

            if resize == True:
                    truth_img = cv2.resize(
                        truth_img,
                        dsize = (resize_height,resize_width),
                        interpolation = cv2.INTER_NEAREST)
            image_list.append(image_to_array(image_path))
            truth_list.append(truth_img)

        generator = normal_image_generator(image_list,
                                           truth_list)
    elif mode == 'predict':
        generator = single_image_generator(image_path_list)

    if mode == 'train' or mode == 'test':
        weight_batch = []
        while a == True:
            for element in generator:
                if mode == 'train':
                    img,truth_img,weight_map,_ = element
                else:
                    img,truth_img = element
                #Normalize data between 0 - 1
                img = img / 255.
                img = img.astype(np.float32)
                #for i in range(3):
                    #tmp = img[:,:,i]
                    #z_tmp = (tmp - np.mean(tmp)) / np.std(tmp)
                    #img[:,:,i] = z_tmp
                    #num = (z_tmp - np.min(z_tmp))
                    #den = (np.max(z_tmp) - np.min(z_tmp))
                    #img[:,:,i] = num / den
                if len(batch) >= batch_size:
                    batch = []
                    truth_batch = []
                    weight_batch = []
                if net_x != None and net_y != None:
                    x1,y1 = truth_img.shape[0:2]
                    x2,y2 = (int((x1 - net_x)/2),int((y1 - net_y)/2))
                    truth_img = truth_img[x2:x1 - x2,y2:y1 - y2,:]
                    if mode == 'train':
                        weight_map = weight_map[x2:x1 - x2,y2:y1 - y2]
                batch.append(img)
                truth_batch.append(truth_img)
                if mode == 'train':
                    weight_batch.append(weight_map)

                if len(batch) >= batch_size:
                    if mode == 'train':
                        yield batch,truth_batch,weight_batch
                    else:
                        yield batch,truth_batch
            if len(batch) > 0:
                if mode == 'train':
                    yield batch,truth_batch,weight_batch
                else:
                    yield batch,truth_batch
            if mode == 'test':
                a = False

    elif mode == 'predict':
        batch_paths = []
        for img,img_path in generator:
            if len(batch) >= batch_size:
                batch = []
                batch_paths = []
            img = img / 255.
            img = img.astype(np.float32)
            #for i in range(3):
                #tmp = img[:,:,i]
                #z_tmp = (tmp - np.mean(tmp)) / np.std(tmp)
                #img[:,:,i] = z_tmp
                #num = (z_tmp - np.min(z_tmp))
                #den = (np.max(z_tmp) - np.min(z_tmp))
                #img[:,:,i] = num / den
            batch.append(img)
            batch_paths.append(img_path)

            if len(batch) >= batch_size:
                yield batch,batch_paths
        if len(batch) > 0:
            yield batch,batch_paths

    elif mode == 'large_predict':
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

    elif mode == 'tumble_predict':
        batch_paths = []
        for img,img_path in generator:
            for rotation in range(1,4):
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

def classification_generator(image_path_list,classification_list,
                             chances = [0,0,0],
                             input_height = 256,input_width = 256,
                             padding = 'VALID',n_classes = 2,
                             mode = 'train',batch_size = 4):
    """
    Realtime image generator for encoder pre-training.
    """

    batch = []
    classification_batch = []
    a = True

    image_list = [image_to_array(im_path) for im_path in image_path_list]
    tmp_images = []
    tmp_classifications = []
    for image,classification in zip(image_list,classification_list):
        if image.shape[0] == input_height and image.shape[1] == input_width:
            tmp_images.append(image)
            tmp_classifications.append(classification)

            if len(tmp_images) % 5 == 0:
                print(len(tmp_images))
    image_list = tmp_images
    classification_list = tmp_classifications

    if mode == 'train':
        generator = realtime_image_augmentation(
            image_list=image_list,
            truth_list=None,
            weight_map_list=None,
            classification_list=classification_list,
            noise_chance=chances[0],
            blur_chance=chances[1],
            flip_chance=chances[2])

    else:
        generator = normal_image_generator(image_list,classification_list)

    while a == True:
        for element in generator:
            if mode == 'train':
                img,_,_,classification = element
            else:
                img,classification = element
            #Normalize data between 0 - 1
            img = img.astype(np.float32)
            for i in range(3):
                tmp = img[:,:,i]
                z_tmp = (tmp - np.mean(tmp)) / np.std(tmp)
                num = (z_tmp - np.min(z_tmp))
                den = (np.max(z_tmp) - np.min(z_tmp))
                img[:,:,i] = num / den
            if len(batch) >= batch_size:
                batch = []
                classification_batch = []

            batch.append(img)
            classification_batch.append(classification)

            if len(batch) >= batch_size:
                yield batch,classification_batch
        print(a)
        if len(batch) > 0:
            yield batch,classification_batch
        if mode == 'test':
            a = False

"""
Weight map calculation from segmentation maps operations.
"""

def get_poormans_weight_map(truth_image,w0=0.5,convolution_size=9):
    """
    Instead of defining a weight map using distances to nearest positive pixels
    (computationally heavy), use the product of a large convolution to weigh
    the pixels (pixels closer to a lot of white pixels will have higher weight
    values, while those further away will be lower).

    EDIT: turns out this is a not very successful idea unless it is done with
    very large convolutions and, ultimately, it does not lead to better results
    than the distance weight map. Keeping this here for some possible
    usefulness it might have.

    Arguments [default]:
    * truth_image - numpy array with ground truth image
    * w0 - lowest weight value possible
    * convolution_size - size of the convolutional filter to be applied
    """
    truth_image = truth_image.copy()
    truth_image = truth_image[:,:,1]
    if truth_image.max() > 0:
        w = cv2.GaussianBlur(
            truth_image,
            (convolution_size,convolution_size),
            0)
        w = (w - w.min())/(w.max() - w.min())
        w = w * (1 - w0) + w0
        w[truth_image > 0] = 1
    else:
        w = np.ones(truth_image.shape) * w0
    return w

def get_near_weight_map(truth_image,w0=10,sigma=5):
    """
    Calculates the weight map for pixels between nearby cells.

    Arguments [default]:
    * truth_image - numpy array with ground truth image
    * w0 - w0 value for the weight map [10]
    * sigma - sigma value for the weight map [5]
    """
    truth_image = truth_image.copy()
    truth_image = truth_image[:,:,1]
    sh = truth_image.shape
    truth_image[truth_image > 0] = 255
    kernel = np.ones((3,3))
    truth_image = cv2.morphologyEx(truth_image.astype('uint8'),
                                   cv2.MORPH_OPEN, kernel)
    if truth_image.max() > 0:
        n_cc,cc = cv2.connectedComponents(truth_image)
        zero_pixels = np.where(truth_image == 0)
        zero_pixels = np.array(zero_pixels).T
        edges = cv2.Canny(truth_image,100,200)
        edges = cv2.dilate(edges,kernel)
        edges[edges > 0] = 1
        accumulator = np.zeros((zero_pixels.shape[0],n_cc))
        for i in range(1,n_cc):
            arr = np.zeros(truth_image.shape)
            arr[cc == i] = 1
            arr = arr * edges
            one_pixels = np.where(arr > 0)
            one_pixels = np.array(one_pixels).T
            d = distance.cdist(zero_pixels,one_pixels)
            d = np.amin(d,axis=1)
            accumulator[:,i] = d
        accumulator = np.sort(accumulator,axis=1,kind='mergesort')
        if accumulator.shape[1] > 2:
            weight_values = np.exp(
                - np.square(
                    accumulator[:,1] + accumulator[:,2]
                ) / (2 * (sigma ** 2))
            )
            output = np.zeros(sh)
            output[zero_pixels[:,0],zero_pixels[:,1]] = w0 * weight_values
        else:
            output = np.zeros(sh)
    else:
        output = np.zeros(sh)
    return output

def get_weight_map(truth_image,w0=0.5,sigma=10):
    """
    Has to be one channel only.

    Arguments [default]:
    * truth_image - numpy array with ground truth image
    * w0 - w0 value for the weight map [10]
    * sigma - sigma value for the weight map [5]
    """
    stride = 128
    size = 256
    sh = truth_image.shape
    max_x = sh[0] // stride
    max_y = sh[1] // stride
    if max_x % stride != 0: max_x += 1
    if max_y % stride != 0: max_y += 1
    kernel = np.ones((3,3))
    truth_image = truth_image.copy()
    truth_image = truth_image[:,:,1]
    truth_image[truth_image > 0] = 255
    truth_image = cv2.morphologyEx(truth_image.astype('uint8'),
                                   cv2.MORPH_OPEN, kernel)
    truth_image_ = truth_image.copy()
    edges = cv2.Canny(truth_image,100,200)

    if 255 in truth_image:
        final_weight_mask = np.zeros((sh[0],sh[1]),dtype = 'float32')
        for i in range(max_x):
            for j in range(max_y):
                sub_image = truth_image[i*stride:(i+1)*size,
                                        j*stride:(j+1)*size]
                sub_edges = edges[i*stride:(i+1)*size,
                                  j*stride:(j+1)*size]
                tmp_weight_mask = final_weight_mask[i*stride:(i+1)*size,
                                                    j*stride:(j+1)*size]
                ssh = sub_image.shape
                pixel_coords = np.where(sub_image == 0)
                pixel_coords_t = np.transpose(pixel_coords)
                weight_mask = np.zeros((ssh[0],ssh[1]),dtype = 'float32')

                cp_coords = np.transpose(np.where(sub_edges > 0))
                distances = distance.cdist(pixel_coords_t,cp_coords)
                distances[distances == 0] = inf
                if distances.any():
                    mins = np.array(np.min(distances,axis = 1))
                    weight_map = ((mins * 2) ** 2)/(2 * sigma ** 2)
                    weight_map = w0 + ((1 - w0) * np.exp(-weight_map))

                    weight_mask[pixel_coords[0],
                                pixel_coords[1]] = weight_map

                else:
                    weight_mask = np.ones((ssh[0],ssh[1]),
                                          dtype = 'float32') * w0
                tmp_weight_mask = np.where(
                    weight_mask > tmp_weight_mask,
                    weight_mask,
                    tmp_weight_mask
                )
                final_weight_mask[i*stride:(i+1)*size,
                                  j*stride:(j+1)*size] = tmp_weight_mask

    else:
        final_weight_mask = np.ones(truth_image.shape) * w0

    final_weight_mask[truth_image > 0] = 1
    return final_weight_mask

"""
Miscellaneous operations.
"""

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
