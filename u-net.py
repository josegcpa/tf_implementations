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
from unet_utilities import *

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

def main(mode,
         log_file,
         log_every_n_steps,
         save_summary_steps,
         save_summary_folder,
         save_checkpoint_steps,
         save_checkpoint_folder,
         squeeze_and_excite,
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
         checkpoint_path,
         prediction_output,
         large_prediction_output,
         data_augmentation_params,
         resize,
         resize_height,
         resize_width,
         dataset_dir,
         path_csv,
         truth_dir,
         padding,
         extension,
         input_height,
         input_width,
         n_classes,
         trial,
         aux_node):

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
    * mode - algorithm mode.
    * checkpoint_path - path to the checkpoint to restore
    * prediction_output - where the predicted output should be stored
    * large_prediction_output - where the large image predictions should be
    stored
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
    * n_classes - no. of classes (currently only supports 2/3)
    """

    log_write_print(log_file,'INPUT ARGUMENTS:')
    for var in vars(args):
        log_write_print(log_file,'\t{0}={1}'.format(var,vars(args)[var]))
    print('\n')

    print("Preparing the network...\n")

    if dataset_dir != None:
        image_path_list = glob(dataset_dir + '/*' + extension)

    if path_csv != None:
        with open(path_csv) as o:
            lines = o.readlines()
        image_path_list = []
        for line in lines[1:]:
            tmp = line.strip().split(',')
            if len(tmp) == 3:
                if tmp[2] == '1':
                    image_path_list.append(tmp[1])
        image_path_list = list(set(image_path_list))

    if trial:
        image_path_list = image_path_list[:50]

    if mode == 'train':
        is_training = True
        output_types = (tf.uint8,tf.float32,tf.float32)
        output_shapes = (
            [input_height,input_width,3],
            [input_height,input_width,n_classes],
            [input_height,input_width,1]
            )
    elif 'test' in mode:
        is_training = False
        output_types = (tf.uint8,tf.float64)
        output_shapes = (
            [input_height,input_width,3],
            [input_height,input_width,n_classes]
            )
    elif 'predict' in mode:
        is_training = False
        output_types = tf.uint8,tf.string
        output_shapes = ([input_height,input_width,3],[])
    elif mode == 'large_predict':
        is_training = False
        output_types = (tf.uint8,tf.string,tf.int32,tf.int32)
        output_shapes = ([input_height,input_width,3],[],[2],[])

    if np.all([extension == 'tfrecord',
               dataset_dir != None,
               mode in ['train','test']]):
        def parse_example(serialized_example):
            feature = {
                'image': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string),
                'weight_mask': tf.FixedLenFeature([], tf.string),
                'image_name': tf.FixedLenFeature([], tf.string),
                'classification': tf.FixedLenFeature([], tf.int64)
            }
            features = tf.parse_single_example(serialized_example,
                                               features=feature)
            image = tf.decode_raw(
                features['image'],tf.uint8)
            mask = tf.decode_raw(
                features['mask'],tf.uint8)
            weights = tf.decode_raw(
                features['weight_mask'],tf.float64)

            image = tf.reshape(image,[input_height, input_width, 3])
            mask = tf.reshape(mask,[input_height, input_width, n_classes])
            weights = tf.reshape(weights,[input_height, input_width, 1])
            weights = tf.cast(weights,tf.float32)
            return image,mask,weights

        files = tf.data.Dataset.list_files(
            '{}/*tfrecord*'.format(dataset_dir))
        dataset = files.interleave(
            tf.data.TFRecordDataset,
            np.maximum(np.minimum(len(image_path_list)//10,50),1)
        )
        if mode == 'train':
            dataset = dataset.repeat()
            dataset = dataset.shuffle(len(image_path_list))
        dataset = dataset.map(parse_example)
        dataset = dataset.batch(batch_size)
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=500)
        iterator = dataset.make_one_shot_iterator()

        next_element = iterator.get_next()
        if 'test' in mode:
            next_element = [next_element[0],next_element[1]]

    else:
        if 'tumble' in mode:
            gen_mode = mode.replace('tumble_','')
        else:
            gen_mode = mode

        next_element = tf_dataset_from_generator(
            generator=generate_images,
            generator_params={
                'image_path_list':image_path_list,
                'truth_path':truth_dir,
                'input_height':input_height,
                'input_width':input_width,
                'n_classes':n_classes,
                'truth_only':truth_only,
                'mode':gen_mode
                },
            output_types=output_types,
            output_shapes=output_shapes,
            is_training=is_training,
            buffer_size=500,
            batch_size=batch_size)

    if epochs != None:
        number_of_steps = epochs * int(len(image_path_list)/batch_size)

    if mode == 'train':
        inputs,truth,weights = next_element

        IA = tf_da.ImageAugmenter(**data_augmentation_params)
        inputs_original = inputs
        inputs,truth,weights = tf.map_fn(
            lambda x: IA.augment(x[0],x[1],x[2]),
            [inputs,truth,weights],
            (tf.float32,tf.float32,tf.float32)
            )

    elif 'test' in mode:
        inputs,truth = next_element
        truth = tf.cast(truth,tf.float32)
        weights = tf.placeholder(tf.float32,
                                 [batch_size,input_height,input_width,1])

    elif 'predict' in mode:
        inputs,image_names = next_element
        truth = tf.placeholder(tf.float32,
                               [batch_size,input_height,input_width,n_classes])
        weights = tf.placeholder(tf.float32,
                                 [batch_size,input_height,input_width,1])

    elif mode == 'large_predict':
        inputs,large_image_path,large_image_coords,batch_shape = next_element
        truth = tf.placeholder(tf.float32,
                               [batch_size,input_height,input_width,n_classes])
        weights = tf.placeholder(tf.float32,
                                 [batch_size,input_height,input_width,1])

    if 'tumble' in mode:
        flipped_inputs = tf.image.flip_left_right(inputs)
        inputs = tf.concat(
            [inputs,
             tf.image.rot90(inputs,1),
             tf.image.rot90(inputs,2),
             tf.image.rot90(inputs,3),
             flipped_inputs,
             tf.image.rot90(flipped_inputs,1),
             tf.image.rot90(flipped_inputs,2),
             tf.image.rot90(flipped_inputs,3)],
             axis=0
             )

    inputs = tf.image.convert_image_dtype(inputs,tf.float32)

    if padding == 'VALID':
        net_x,net_y = input_height - 184,input_width - 184
        tf_shape = [None,net_x,net_y,n_classes]
        if training == True:
            truth = truth[:,92:(input_height - 92),92:(input_width - 92),:]
            weights = weights[:,92:(input_height - 92),92:(input_width - 92),:]
        crop = True

    else:
        if resize == True:
            inputs = tf.image.resize_bilinear(images,
                                              [resize_height,resize_width])
            if training == True:
                truth = tf.image.resize_bilinear(
                    truth,
                    [resize_height,resize_width])
                weights = tf.image.resize_bilinear(
                    weights,
                    [resize_height,resize_width])
        net_x,net_y = (None, None)
        crop = False

    weights = tf.squeeze(weights,axis=-1)

    network,endpoints,classifications = u_net(
        inputs,
        final_endpoint=None,
        padding=padding,
        factorization=factorization,
        residuals=residuals,
        beta=beta_l2_regularization,
        n_classes=n_classes,
        resize=resize,
        resize_height=resize_height,
        resize_width=resize_width,
        depth_mult=depth_mult,
        aux_node=aux_node,
        squeeze_and_excite=squeeze_and_excite
        )

    log_write_print(log_file,
                    'Total parameters: {0:d} (trainable: {1:d})\n'.format(
                        variables(tf.all_variables()),
                        variables(tf.trainable_variables())
                        ))

    saver = tf.train.Saver()
    loading_saver = tf.train.Saver()


    class_balancing = tf.stack(
        [tf.ones_like(truth[:,:,:,0])/tf.reduce_sum(truth[:,:,:,0]),
         tf.ones_like(truth[:,:,:,1])/tf.reduce_sum(truth[:,:,:,1])],
        axis=3
        )

    if iglovikov == True:
        loss = iglovikov_loss(truth,network)

    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=truth,
            logits=network)# * class_balancing
        loss = tf.reduce_sum(loss,axis=3) * weights
        loss = tf.reduce_mean(loss)

    if beta_l2_regularization > 0:
        reg_losses = slim.losses.get_regularization_losses()
        loss = loss + tf.add_n(reg_losses) / len(reg_losses)

    binarized_truth = tf.argmax(truth,axis = 3)
    binarized_network = tf.argmax(network,axis = 3)
    prediction_network = tf.expand_dims(
        tf.nn.softmax(network,axis=3)[:,:,:,1],-1)

    if 'tumble' in mode:
        flipped_prediction = prediction_network[4:,:,:,:]
        prediction_network = prediction_network[:4,:,:,:]
        flipped_prediction = tf.image.flip_left_right(flipped_prediction)
        print(prediction_network,flipped_prediction)

        prediction_network = tf.stack([
            prediction_network[0,:,:,:],
            tf.image.rot90(prediction_network[1,:,:,:],-1),
            tf.image.rot90(prediction_network[2,:,:,:],-2),
            tf.image.rot90(prediction_network[3,:,:,:],-3),
            flipped_prediction[0,:,:,:],
            tf.image.rot90(flipped_prediction[1,:,:,:],-1),
            tf.image.rot90(flipped_prediction[2,:,:,:],-2),
            tf.image.rot90(flipped_prediction[3,:,:,:],-3)],axis=0)
        prediction_network = tf.reduce_mean(prediction_network,
                                            axis=0,
                                            keepdims=True)

        binarized_network = tf.where(prediction_network > 0.5,
                                     tf.ones_like(prediction_network),
                                     tf.zeros_like(prediction_network))

        binarized_truth = tf.expand_dims(binarized_truth,axis=-1)

    auc, auc_op = tf.metrics.auc(
        binarized_truth,
        binarized_network)
    f1score,f1score_op = tf.contrib.metrics.f1_score(
        binarized_truth,
        binarized_network)
    m_iou,m_iou_op = tf.metrics.mean_iou(
        labels=binarized_truth,
        predictions=binarized_network,
        num_classes=2)
    auc_batch, auc_batch_op = tf.metrics.auc(
        binarized_truth,
        binarized_network,
        name='auc_batch')
    f1score_batch,f1score_batch_op = tf.contrib.metrics.f1_score(
        binarized_truth,
        binarized_network,
        name='f1_batch')
    m_iou_batch,m_iou_batch_op = tf.metrics.mean_iou(
        labels=binarized_truth,
        predictions=binarized_network,
        num_classes=2,
        name='m_iou_batch')

    batch_vars = [v for v in tf.local_variables()]
    batch_vars = [v for v in batch_vars if 'batch' in v.name]
    #train_op = tf.train.MomentumOptimizer(learning_rate,0.99).minimize(loss)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=int(number_of_steps * 0.8)
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,global_step=global_step)

    if aux_node:
        presence = tf.reduce_sum(binarized_truth,axis=[1,2]) > 0
        presence = tf.expand_dims(
            tf.cast(presence,tf.float32),
            axis=1)
        class_loss = tf.reduce_mean(
            tf.add_n(
                [
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=presence,
                        logits=c) for c in classifications
                ]
            )
        )
        trainable_variables = tf.trainable_variables()
        aux_vars = []
        for var in trainable_variables:
            if 'Aux_Node' in var.name:
                aux_vars.append(var)
        aux_optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            class_train_op = aux_optimizer.minimize(
                    class_loss,
                    var_list=aux_vars,
                    global_step=global_step
                )
        train_op = tf.group(train_op,class_train_op)
        loss = [loss,class_loss]

    if mode == 'train':
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        for endpoint in endpoints:
            x = endpoints[endpoint]
            summaries.add(tf.summary.histogram('activations/' + endpoint, x))

        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        if aux_node:
            summaries.add(tf.summary.scalar('loss', loss[0]))
            summaries.add(tf.summary.scalar('class_loss', class_loss))
        else:
            summaries.add(tf.summary.scalar('loss', loss))
        summaries.add(tf.summary.scalar('f1score', f1score))
        summaries.add(tf.summary.scalar('auc', auc))

        summaries.add(
            tf.summary.image('image',inputs_original,max_outputs = 4))
        summaries.add(
            tf.summary.image('image',inputs,max_outputs = 4))
        summaries.add(
            tf.summary.image('truth_image',
                             tf.cast(
                                 tf.expand_dims(binarized_truth,-1),tf.float32),
                             max_outputs = 4))
        summaries.add(
            tf.summary.image('weight_map',
                             tf.expand_dims(weights,-1),
                             max_outputs = 4))
        summaries.add(
            tf.summary.image(
                'prediction',
                tf.expand_dims(tf.nn.softmax(network,axis=3)[:,:,:,1],-1),
                max_outputs = 4))
        summaries.add(
            tf.summary.image('prediction_binary',
                             tf.cast(
                                 tf.expand_dims(binarized_network,-1),tf.float32),
                             max_outputs = 4))
        summaries.add(
            tf.summary.image(
                'compare_binary',
                tf.stack(
                    [tf.cast(binarized_network,tf.float32),
                     tf.cast(binarized_truth,tf.float32),
                     tf.cast(tf.zeros_like(binarized_truth),tf.float32)],axis=-1),
                max_outputs = 4)
            )

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.tables_initializer()
        )

    tf.set_random_seed(0)
    np.random.seed(42)

    ckpt_exists = os.path.exists(checkpoint_path + '.index')
    if len(image_path_list) > 0:

        if mode == 'train':

            print("Training the network...\n")
            LOG = 'Step {0:d}: minibatch loss: {1:f}. '
            LOG += 'Average time/minibatch = {2:f}s. '
            LOG += 'F1-Score: {3:f}; AUC: {4:f}; '
            SUMMARY = 'Step {0:d}: summary stored in {1!s}'
            CHECKPOINT = 'Step {0:d}: checkpoint stored in {1!s}'
            CHECKPOINT_PATH = os.path.join(save_checkpoint_folder,
                                           'my_u-net.ckpt')

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
                config.gpu_options.allow_growth = True
            else:
                n_phys_cores = psutil.cpu_count(logical = False)
                config.intra_op_parallelism_threads = n_phys_cores
                config.inter_op_parallelism_threads = n_phys_cores

            with tf.Session(config = config) as sess:
                writer = tf.summary.FileWriter(save_summary_folder,sess.graph)

                sess.run(init)

                if ckpt_exists:
                    loading_saver.restore(sess,checkpoint_path)
                    print('Restored')

                time_list = []

                all_class_losses = []
                for i in range(number_of_steps):
                    a = time.perf_counter()
                    _,l,_,_ = sess.run(
                        [train_op,loss,f1score_op,auc_op])

                    if aux_node:
                        class_l = l[1]
                        l = l[0]
                        all_class_losses.append(class_l)
                    b = time.perf_counter()
                    time_list.append(b - a)
                    if i % log_every_n_steps == 0 or i == 1:
                        l,_,_ = sess.run([loss,auc_op,f1score_op])
                        f1,auc_ = sess.run([f1score,auc])
                        log_write_print(log_file,
                                        LOG.format(i,l,np.mean(time_list),
                                                   f1,auc_))
                        time_list = []
                        if aux_node:
                            class_l = np.mean(all_class_losses)
                            all_class_losses = []
                            print('\tAux_Node loss = {}'.format(class_l))

                    if i % save_summary_steps == 0 or i == 1:
                        summary = sess.run(summary_op)
                        writer.add_summary(summary,i)
                        log_write_print(
                            log_file,SUMMARY.format(i,save_summary_folder))

                        if i % save_checkpoint_steps == 0 or i == 1:
                            saver.save(sess, CHECKPOINT_PATH,global_step=i)
                            log_write_print(log_file,
                                            CHECKPOINT.format(i,
                                                              CHECKPOINT_PATH))
                        sess.run(tf.local_variables_initializer())

                summary = sess.run(summary_op)
                writer.add_summary(summary,i)
                log_write_print(
                    log_file,SUMMARY.format(i,save_summary_folder))
                saver.save(sess,CHECKPOINT_PATH,global_step=i)
                log_write_print(log_file,
                                CHECKPOINT.format(i,CHECKPOINT_PATH))

        elif 'test' in mode and ckpt_exists:
            LOG = 'Time/{0:d} images: {1:f}s (time/1 image: {2:f}s). '
            LOG += 'F1-Score: {3:f}; AUC: {4:f}; MeanIOU: {5:f}'

            FINAL_LOG = 'Final averages - time/image: {0}s; F1-score: {1}; '
            FINAL_LOG += 'AUC: {2}; MeanIOU: {3}'

            print('Testing...')
            with tf.Session() as sess:

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                all_f1score = []
                all_auc = []
                all_m_iou = []
                time_list = []

                keep_going = True

                while keep_going == True:

                    try:
                        a = time.perf_counter()
                        img,_,(f1,auc_,iou) = sess.run(
                            [network,
                             (auc_op,f1score_op,m_iou_op,
                              auc_batch_op,
                              f1score_batch_op,
                              m_iou_batch_op),
                             (f1score_batch,
                              auc_batch,
                              m_iou_batch)])

                        n_images = img.shape[0]

                        b = time.perf_counter()
                        t_image = (b - a)/n_images
                        time_list.append(t_image)

                        all_f1score.append(f1)
                        all_auc.append(auc_)
                        all_m_iou.append(iou)

                        tf.initializers.variables(var_list=batch_vars)

                    except:
                        keep_going = False

                f1score_,auc_,iou = sess.run([f1score,auc,m_iou])
                averages = [np.mean(time_list),
                            np.mean(all_f1score),
                            np.mean(all_auc),
                            np.mean(all_m_iou)]

                stds = [np.std(time_list),
                        np.std(all_f1score),
                        np.std(all_auc),
                        np.std(all_m_iou)]

                min_ci = [np.percentile(time_list,2.5),
                          np.percentile(all_f1score,2.5),
                          np.percentile(all_auc,2.5),
                          np.percentile(all_m_iou,2.5)]
                max_ci = [np.percentile(time_list,97.5),
                          np.percentile(all_f1score,97.5),
                          np.percentile(all_auc,97.5),
                          np.percentile(all_m_iou,97.5)]

                tmp = '{0:.5f} (Mean:{1:.5f}; CI:{2:.5f}-{3:.5f};std:{4:.5f})'
                output = FINAL_LOG.format(
                    tmp.format(averages[0],averages[0],min_ci[0],
                               max_ci[0],stds[0]),
                    tmp.format(f1score_,averages[1],min_ci[1],
                               max_ci[1],stds[1]),
                    tmp.format(auc_,averages[2],min_ci[2],
                               max_ci[2],stds[2]),
                    tmp.format(iou,averages[3],min_ci[3],
                               max_ci[3],stds[3]))
                log_write_print(log_file,output)

        elif 'predict' in mode and ckpt_exists:
            print('Predicting...')

            LOG = 'Time/{0:d} images: {1:f}s (time/1 image: {2:f}s).'
            FINAL_LOG = 'Average time/image: {0:f}'

            prob_network = tf.nn.softmax(network)[:,:,:,1]

            with tf.Session() as sess:
                try:
                    os.makedirs(prediction_output)
                except:
                    pass

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                time_list = []

                keep_going = True

                while keep_going == True:

                    try:
                        a = time.perf_counter()
                        prediction,im_names = sess.run([prob_network,
                                                        image_names])
                        n_images = prediction.shape[0]
                        b = time.perf_counter()
                        t_image = (b - a)/n_images
                        time_list.append(t_image)

                        output = LOG.format(n_images,b - a,t_image)
                        log_write_print(log_file,output)

                        for i in range(prediction.shape[0]):
                            image = prediction[i,:,:]
                            image_name = im_names[i].decode().split(os.sep)[-1]
                            image_name = image_name.split('.')[0]
                            image_name = image_name + '.tif'
                            image_output = os.path.join(prediction_output,
                                                        image_name)
                            tiff.imsave(image_output,image)

                    except:
                        keep_going = False

                avg_time = np.mean(time_list)
                output = FINAL_LOG.format(avg_time)
                log_write_print(log_file,output)

        elif mode == 'large_predict' and ckpt_exists:
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

                sess.run(init)
                trained_network = saver.restore(sess,checkpoint_path)

                time_list = []

                curr_image_name = ''

                for batch,image_names,coords,shapes in image_generator:
                    n_images = len(batch)
                    batch = np.stack(batch,0)

                    a = time.perf_counter()
                    prediction = sess.run(network)
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

parser.add_argument('--mode',dest = 'mode',
                    action = 'store',
                    default = 'train',
                    help = 'Algorithm mode.')

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
parser.add_argument('--squeeze_and_excite',dest='squeeze_and_excite',
                    action='store_true',
                    default=False,
                    help='Adds SC SqAndEx layers to the enc/dec.')
parser.add_argument('--iglovikov',dest='iglovikov',
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
parser.add_argument('--aux_node',dest = 'aux_node',
                    action = 'store_true',
                    default = False,
                    help = 'Aux node for classification task in bottleneck.')

parser.add_argument('--checkpoint_path',dest = 'checkpoint_path',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path to checkpoint to restore.')

#Prediction
parser.add_argument('--prediction_output',dest = 'prediction_output',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path where image predictions are stored.')

#Large image prediction
parser.add_argument('--large_prediction_output',
                    dest = 'large_prediction_output',
                    action = ToDirectory,
                    default = 'no_path',
                    help = 'Path to store large image predictions.')

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
    ['max_jpeg_quality',70,int]
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
#Pre-processing
parser.add_argument('--noise_chance',dest = 'noise_chance',
                    action = 'store',type = float,
                    default = 0.1,
                    help = 'Probability to add noise.')
parser.add_argument('--blur_chance',dest = 'blur_chance',
                    action = 'store',type = float,
                    default = 0.05,
                    help = 'Probability to blur the input image.')
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
                    action = ToDirectory,
                    default = None,
                    type = str,
                    help = 'Directory where the training set is stored.')
parser.add_argument('--path_csv',dest = 'path_csv',
                    action = ToDirectory,
                    default = None,
                    type = str,
                    help = 'CSV with QCd paths.')
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
parser.add_argument('--trial',dest = 'trial',
                    action = 'store_true',
                    default = False,
                    help = 'Subsamples the dataset for a quick run.')

args = parser.parse_args()

mode = args.mode

#Logs
log_file = args.log_file
log_every_n_steps = args.log_every_n_steps

#Summaries
save_summary_steps = args.save_summary_steps
save_summary_folder = args.save_summary_folder

#Checkpoints
save_checkpoint_steps = args.save_checkpoint_steps
save_checkpoint_folder = args.save_checkpoint_folder
checkpoint_path = args.checkpoint_path

#Training
squeeze_and_excite = args.squeeze_and_excite
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
aux_node = args.aux_node

#Prediction
prediction_output = args.prediction_output

#Large image prediction
large_prediction_output = args.large_prediction_output

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
    'max_jpeg_quality':args.max_jpeg_quality
}

#Pre-processing
resize = args.resize
resize_height = args.resize_height
resize_width = args.resize_width

#Dataset
dataset_dir = args.dataset_dir
path_csv = args.path_csv
truth_dir = args.truth_dir
padding = args.padding
extension = args.extension
input_height = args.input_height
input_width = args.input_width
n_classes = args.n_classes
trial = args.trial

if __name__ == '__main__':
    print("Loading dependencies...")

    import sys
    import time
    from glob import glob
    from math import floor,inf
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

    main(log_file=log_file,
         log_every_n_steps=log_every_n_steps,
         save_summary_steps=save_summary_steps,
         save_summary_folder=save_summary_folder,
         save_checkpoint_steps=save_checkpoint_steps,
         save_checkpoint_folder=save_checkpoint_folder,
         squeeze_and_excite=squeeze_and_excite,
         iglovikov=iglovikov,
         batch_size=batch_size,
         number_of_steps=number_of_steps,
         epochs=epochs,
         beta_l2_regularization=beta_l2_regularization,
         learning_rate=learning_rate,
         factorization=factorization,
         residuals=residuals,
         weighted=weighted,
         depth_mult=depth_mult,
         truth_only=truth_only,
         mode=mode,
         checkpoint_path=checkpoint_path,
         prediction_output=prediction_output,
         large_prediction_output=large_prediction_output,
         data_augmentation_params=data_augmentation_params,
         resize=resize,
         resize_height=resize_height,
         resize_width=resize_width,
         dataset_dir=dataset_dir,
         path_csv=path_csv,
         truth_dir=truth_dir,
         padding=padding,
         extension=extension,
         input_height=input_height,
         input_width=input_width,
         n_classes=n_classes,
         trial=trial,
         aux_node=aux_node)
