"""
Implementation of a DNN using the attention modules described in [1]. It is
similar to achtung!.py but while achtung!.py was implemented for a specific
dataset, this one is generalized for a folder containing images and a csv
containing its classes.

[1] http://openaccess.thecvf.com/content_ECCV_2018/papers/Pau_Rodriguez_Lopez_Attend_and_Rectify_ECCV_2018_paper.pdf
"""

import sys
import os
import time
from glob import glob
from math import ceil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from multi_gpu_neural_net import get_available_gpus
from multi_gpu_neural_net import assign_to_device
from multi_gpu_neural_net import average_gradients
from resnet_v2 import resnet_v2_50,resnet_v2_101,resnet_v2_152
from resnet_v2 import resnet_arg_scope
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

slim = tf.contrib.slim

height = 512
width = 512
n_channels = 3
num_classes = 2
is_training = True
save_checkpoint_steps = 500
save_summary_steps = 100
beta_l2_regularization = 0.005

checkpoint_path = sys.argv[1]
image_folder = sys.argv[2]
csv_path = sys.argv[3]
learning_rate = float(sys.argv[4])
iterations = int(sys.argv[5])
batch_size = int(sys.argv[6])
mode = str(sys.argv[7])
checkpoint_save_path = str(sys.argv[8])
attention = str(sys.argv[9])
multi_gpu = str(sys.argv[10])
if attention == 'True':
    attention = True
if multi_gpu == 'True':
    multi_gpu = True

def sess_debugger(wtv,times = 1):
    """
    Convenience function used for debugging. It creates a self contained
    session and runs whatever its input is.
    """
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        if times > 1:
            for _ in range(times):
                yield sess.run(wtv)
        else:
            return sess.run(wtv)
        coord.request_stop()
        coord.join(threads)

def attention_head(inputs, depth, kernel):
    with tf.name_scope('AttentionHead') and tf.variable_scope('AttentionHead'):
        convolution = slim.conv2d(
            inputs,
            depth,
            kernel,
            activation_fn=tf.nn.leaky_relu,
            normalizer_params={'is_training':is_training},
            scope='conv2d_{}'.format(depth))
        convolution = slim.dropout(convolution,keep_prob=keep_prob)
        output = tf.nn.softmax(convolution, axis=-1,
                               name='output_vector')
    return output

def attention_output(inputs, attention_head, depth, kernel):
    with tf.name_scope('AttentionOutput') and\
     tf.variable_scope('AttentionOutput'):
        convolution = slim.conv2d(
            inputs,
            depth,
            kernel,
            activation_fn=tf.nn.leaky_relu,
            normalizer_params={'is_training':is_training},
            scope='conv2d_{}'.format(depth))
        convolution = slim.dropout(convolution,keep_prob=keep_prob)
        element_wise_mult = tf.multiply(convolution,
                                        attention_head,
                                        name='element_wise_mult')
        output = tf.reduce_mean(element_wise_mult,
                                axis=[1, 2],
                                name='output_vector')
    return output

def attention_gates(inputs, attention_head, depth, kernel):
    with tf.name_scope('AttentionGates') and\
     tf.variable_scope('AttentionGates'):
        convolution = slim.conv2d(
            inputs,
            depth,
            kernel,
            activation_fn=tf.nn.leaky_relu,
            normalizer_params={'is_training':is_training},
            scope='conv2d_{}'.format(depth))
        convolution = slim.dropout(convolution,keep_prob=keep_prob)
        element_wise_mult = tf.multiply(convolution,
                                        attention_head,
                                        name='element_wise_mult')
        avg_pool = tf.reduce_mean(element_wise_mult,
                                  axis=[1, 2],
                                  name='avg_pool')
        output = tf.nn.softmax(tf.nn.tanh(avg_pool), axis=1,
                               name='output_vector')
    return output

def attention_module(inputs, depth, kernel, n):
    with tf.name_scope('AttentionModule' + str(n)) and\
     tf.variable_scope('AttentionModule' + str(n)):
        head = attention_head(inputs, depth, kernel)
        output = attention_output(inputs, head, depth, kernel)
        gates = attention_gates(inputs, head, depth, kernel)
        output = tf.multiply(output,
                             gates,
                             name='output')
    return output

def final_attention(inputs, predictions, attention_modules, depth, kernel):
    with tf.name_scope('FinalAttentionGate') and\
     tf.variable_scope('FinalAttentionGate'):
        head = attention_head(inputs, depth, kernel)
        final_gate = attention_gates(inputs, head, depth, kernel)
        aggregated_gates = tf.add_n(attention_modules)
        output_vector = final_gate * aggregated_gates * predictions
    return output_vector

def csv_parser(csv_path,num_classes):
    """
    Uses a csv file to create a lookup table in tensorflow.
    """

    file_names = []
    classes = []
    counter = [0 for i in range(num_classes)]
    with open(csv_path) as o:
        lines = o.readlines()
    for line in lines:
        file_name, cl = line.split(',')
        file_names.append(file_name.encode('ascii'))
        cl = cl.strip()
        classes.append(cl)
        counter[int(cl)] += 1
    file_names = tf.convert_to_tensor(file_names)
    classes = tf.convert_to_tensor(np.array(classes))
    counter = np.array(counter)
    weight_vector = np.sum(counter) / counter
    weight_vector = weight_vector / np.linalg.norm(weight_vector,1)
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            file_names,
            classes,
            key_dtype = tf.string,
            value_dtype = tf.string),
        ''
    )

    return table,weight_vector

def image_generator(image_path_list, batch_size, width, height, n_channels):
    """
    Creates a tf native input pipeline. Makes everything quite a bit faster
    than using the feed_dict method.
    """

    train_channels = []
    image_path_tensor = ops.convert_to_tensor(image_path_list,
                                              dtype=dtypes.string)
    train_input_queue = tf.train.slice_input_producer(
        [image_path_tensor],
        shuffle=False)[0]
    file_content = tf.read_file(train_input_queue)
    train_image = tf.image.decode_image(file_content, channels=n_channels)
    train_image.set_shape([width, height, n_channels])
    train_image = tf.image.convert_image_dtype(train_image,
                                               dtype=tf.float32)

    images,file_names = tf.train.shuffle_batch(
        [train_image,train_input_queue],
        batch_size=batch_size,
        capacity=1000,
        min_after_dequeue=500,
        allow_smaller_final_batch=True
    )

    return images, file_names

def string_to_one_hot(string,one_hot_size):
    string = tf.reshape(string,[1])
    cl = tf.string_split(string)
    cl = tf.sparse_to_dense(sparse_indices = cl.indices,
                            output_shape = cl.dense_shape,
                            sparse_values = cl.values,
                            default_value = '')
    cl = tf.string_to_number(cl,out_type=tf.int32)
    cl = tf.one_hot(cl,one_hot_size)
    cl = tf.reduce_sum(cl,axis = 1)
    return cl

def loss_function(cl,predictions):
    weights = tf.reduce_sum(cl,0)
    weights = (weights * weight_vector) / (weights + 1)
    pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.squeeze(cl,1),
        logits = predictions)
    weighted_loss = pre_loss * weight_vector
    return weighted_loss

def check_size(image_path_list):
    new_path_list = []
    for i,image_path in enumerate(image_path_list):
        if i % 100 == 0: print(i)
        image = cv2.imread(image_path)
        if image.shape[0] == height and image.shape[1] == width:
            new_path_list.append(image_path)
    return new_path_list

image_path_list = glob(
    os.path.join(image_folder,'*')
    )

image_path_list = check_size(image_path_list)

if is_training == True: keep_prob = 0.8
else: keep_prob = 1.

if multi_gpu == False:
    inputs, file_names = image_generator(image_path_list,
                                         batch_size,
                                         height,
                                         width,
                                         n_channels
                                         )

    fixed_file_names = tf.string_split(file_names,delimiter=os.sep)
    fixed_file_names = tf.sparse_to_dense(
        sparse_indices=fixed_file_names.indices,
        output_shape=fixed_file_names.dense_shape,
        sparse_values=fixed_file_names.values,
        default_value='')[:,-1]

    if mode == 'train' or mode == 'test':
        table,weight_vector = csv_parser(csv_path,num_classes)
        truth = table.lookup(fixed_file_names)

        cl = tf.map_fn(
            lambda x: string_to_one_hot(x,num_classes),
            truth,
            dtype=tf.float32
        )
    with slim.arg_scope(resnet_arg_scope()):
        resnet, end_points = resnet_v2_152(inputs,
                                           num_classes=num_classes,
                                           is_training=is_training)


    if attention:
        attention_modules = {
            1: attention_module(end_points['resnet_v2_152/block1'],
                                num_classes,
                                [3, 3],
                                1),
            2: attention_module(end_points['resnet_v2_152/block2'],
                                num_classes,
                                [3, 3],
                                2),
            3: attention_module(end_points['resnet_v2_152/block3'],
                                num_classes,
                                [3, 3],
                                3),
            4: attention_module(end_points['resnet_v2_152/block4'],
                                num_classes,
                                [3, 3],
                                4)
        }

        predictions = final_attention(inputs,
                                      end_points['predictions'],
                                      list(attention_modules.values()),
                                      num_classes,
                                      [3, 3])

        """
        print_tensors_in_checkpoint_file(file_name=checkpoint_path,
                                         tensor_name='',
                                         all_tensors=False,
                                         all_tensor_names=True)
        """
    else:
        predictions = end_points['predictions']

else:
    gpus = get_available_gpus()
    reuse = None
    all_predictions = []
    all_classes = []
    for i,gpu in enumerate(gpus):
        with tf.device('/cpu:0'):
            with tf.name_scope('InputPipeline_{}'.format(i)):
                inputs, file_names = image_generator(image_path_list,
                                                     batch_size//len(gpus),
                                                     height,
                                                     width,
                                                     n_channels
                                                     )
                if mode == 'train':
                    inputs = tf.image.random_flip_left_right(inputs)
                    inputs = tf.image.random_flip_up_down(inputs)
                elif mode == 'test':
                    inputs = inputs
                fixed_file_names = tf.string_split(file_names,delimiter=os.sep)
                fixed_file_names = tf.sparse_to_dense(
                    sparse_indices=fixed_file_names.indices,
                    output_shape=fixed_file_names.dense_shape,
                    sparse_values=fixed_file_names.values,
                    default_value='')[:,-1]

                if mode == 'train' or mode == 'test':
                    table,weight_vector = csv_parser(csv_path,num_classes)
                    truth = table.lookup(fixed_file_names)
                    cl = tf.map_fn(
                        lambda x: string_to_one_hot(x,num_classes),
                        truth,
                        dtype=tf.float32
                    )

        with tf.device(assign_to_device(device='/gpu:{}'.format(i),
                                        ps_device='/cpu:0')):
            with slim.arg_scope(resnet_arg_scope()):
                resnet, end_points = resnet_v2_152(inputs,
                                                   num_classes=num_classes,
                                                   is_training=is_training,
                                                   reuse=reuse)
            with tf.variable_scope('AttentionModules'):
                with slim.arg_scope([slim.conv2d],reuse=reuse):
                    if attention:
                        attention_modules = {
                            1: attention_module(
                                end_points['resnet_v2_152/block1'],
                                num_classes,
                                [3, 3],
                                1),
                            2: attention_module(
                                end_points['resnet_v2_152/block2'],
                                num_classes,
                                [3, 3],
                                2),
                            3: attention_module(
                                end_points['resnet_v2_152/block3'],
                                num_classes,
                                [3, 3],
                                3),
                            4: attention_module(
                                end_points['resnet_v2_152/block4'],
                                num_classes,
                                [3, 3],
                                4)
                        }
                        predictions = final_attention(
                            inputs,
                            end_points['predictions'],
                            list(attention_modules.values()),
                            num_classes,
                            [3, 3])
                    else:
                        predictions = end_points['predictions'
                    all_predictions.append(predictions)
                    if mode == 'train' or mode == 'test':
                        all_classes.append(cl)
                    reuse = True

trainable_var = tf.trainable_variables()
resnet_v2_variables = []

for var in trainable_var:
    if 'resnet_v2' in var.name and\
     'logits' not in var.name and\
      'resnet_v2_101/conv1' not in var.name:
        resnet_v2_variables.append(var)

saver = tf.train.Saver(var_list=resnet_v2_variables)
other_saver = tf.train.Saver()

if mode == 'train':
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if multi_gpu == False:
        weighted_loss = loss_function(cl,predictions)
        loss = tf.reduce_mean(weighted_loss) + tf.reduce_mean(
            tf.losses.get_regularization_losses()) * beta_l2_regularization

        auc,auc_op = tf.metrics.auc(
            labels = tf.squeeze(cl,1),
            predictions = tf.nn.sigmoid(predictions)
        )
        train_op = optimizer.minimize(loss)

    else:
        tower_grads = []
        all_losses = []
        all_auc = []
        all_auc_op = []
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for i,gpu in enumerate(gpus):
                with tf.device(assign_to_device(
                    device='/gpu:{}'.format(i),
                    ps_device='/cpu:0')),tf.name_scope('tower_{}'.format(i)):
                    predictions = all_predictions[i]
                    truth = all_classes[i]
                    weighted_loss = loss_function(truth,predictions)
                    loss = tf.add(
                        tf.reduce_mean(weighted_loss),
                        tf.reduce_mean(
                            tf.losses.get_regularization_losses()
                            ) * beta_l2_regularization)
                    with tf.name_scope("compute_gradients"):
                        grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)
                all_losses.append(loss)
                outer_scope.reuse_variables()
                auc,auc_op = tf.metrics.auc(
                    labels = tf.squeeze(truth,1),
                    predictions = tf.nn.sigmoid(predictions)
                )
                all_auc.append(auc)
                all_auc_op.append(auc_op)

        with tf.name_scope("apply_gradients"), tf.device('/cpu:0'):
            gradients = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(gradients,
                                                 global_step)
            loss = tf.reduce_mean(all_losses)

        auc = tf.reduce_mean(all_auc)
        auc_op = tf.group(all_auc_op)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    for variable in slim.get_model_variables():
        summaries.add(
            tf.summary.histogram(variable.op.name, variable)
        )
    summaries.add(tf.summary.scalar('loss',loss))
    summaries.add(tf.summary.scalar('auc',auc))
    summary_op = tf.summary.merge(list(summaries),name='summary_op')

    with tf.Session() as sess:
        tf.set_random_seed(42)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('Loading checkpoint...')

        saver.restore(sess, checkpoint_path)
        print('Checkpoint loaded.')
        writer = tf.summary.FileWriter('summary',
                                       sess.graph)
        print('Training...')
        for i in range(iterations):
            a = time.perf_counter()
            _,l,_ = sess.run([train_op,loss,auc_op])
            b = time.perf_counter()
            print(b-a)
            if i % 5 == 0 or i == 1:
                print('Batch: {}; loss: {}; AUC: {}'.format(i,l,sess.run(auc)))
            if i % save_checkpoint_steps == 0:
                other_saver.save(sess,os.path.join(os.getcwd(),
                                                   checkpoint_save_path))
            if i % save_summary_steps == 0:
                summary = sess.run(summary_op)
                writer.add_summary(summary,i)
                writer.flush()
                sess.run(tf.local_variables_initializer())

        other_saver.save(sess,os.path.join(os.getcwd(),
                                           checkpoint_save_path))
        sess.run(auc_op)
        summary = sess.run(summary_op)
        writer.add_summary(summary,i)
        writer.flush()
        coord.request_stop()
        coord.join(threads)

elif mode == 'test':
    iterations = ceil(len(image_path_list) / batch_size)
    auc,auc_op = tf.metrics.auc(
        labels = tf.squeeze(cl,1),
        predictions = tf.nn.sigmoid(predictions)
    )

    with tf.Session() as sess:
        tf.set_random_seed(42)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('Loading checkpoint...')

        other_saver.restore(sess, checkpoint_path)
        print('Checkpoint loaded.')
        print('Testing...')
        for i in range(1,iterations+1):
            _ = sess.run([auc_op])
            if i % 5 == 0 or i == 1:
                print('\t',('{}/{}'.format(i,iterations)),sess.run(auc))
        print('final_auc:',sess.run(auc))
        coord.request_stop()
        coord.join(threads)

elif mode == 'predict':
    predictions = tf.argmax(predictions,axis=1)
    iterations = ceil(len(image_path_list) / batch_size)

    with tf.Session() as sess:
        tf.set_random_seed(42)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('Loading checkpoint...')

        other_saver.restore(sess, checkpoint_path)
        print('Checkpoint loaded.')
        print('Testing...')
        for i in range(1,iterations+1):
            fn,pred = sess.run([file_names,predictions])
            for f,p in zip(fn,pred):
                print('OUT,{},{}'.format(os.path.abspath(str(f,'utf-8')),p))
        coord.request_stop()
        coord.join(threads)
