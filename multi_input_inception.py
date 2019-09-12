import argparse
import os
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from inception_v4 import *
slim = tf.contrib.slim

LR = 0.0001
N_CLASSES = 56
save_checkpoint_steps = 1000
save_summary_steps = 250

def sess_debugger(wtv):
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

def image_generator(path,mode='train',batch_size=4):
    """
    Creates a tf native input pipeline. Makes everything quite a bit faster
    than using the feed_dict method.
    """

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
                                 [512, 512, 3])
        train_image = tf.image.convert_image_dtype(
            train_image,
            tf.float32
        )
        train_input_queue = features['image/filename']
        class_input = features['image/class/label']
        purity_input = features['image/tp']

        return train_image,train_input_queue,class_input,purity_input

    all_files = glob('{}/*tfrecord*'.format(path))
    files = tf.data.Dataset.list_files('{}/*tfrecord*'.format(path))
    dataset = files.interleave(tf.data.TFRecordDataset,
                               np.minimum(len(all_files)/10,50))
    if mode == 'train':
        dataset = dataset.repeat()
    dataset = dataset.shuffle(len(all_files))
    dataset = dataset.map(parse_example)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=2500)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs,file_names,classes,purities = iterator.get_next()
    return inputs,file_names,classes,purities

def inception_v4_from_stem(net, final_endpoint='Mixed_7d', scope=None,
                           reuse=False,num_classes=42,
                           is_training=True,dropout_keep_prob=0.8,
                           weight_decay=0.0):
    """Creates the Inception V4 network up to the given final endpoint.

    Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d']
    scope: Optional variable_scope.

    Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.

    Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
    """
    end_points = {}

    def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint

    with tf.variable_scope(scope, 'InceptionV4', [inputs]):
        with slim.arg_scope([slim.conv2d],
                            stride=1, padding='SAME',reuse=reuse,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training':is_training}):
            # 35 x 35 x 384
            # 4 x Inception-A blocks
            for idx in range(4):
                block_scope = 'Mixed_5' + chr(ord('b') + idx)
                net = block_inception_a(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points

            # 35 x 35 x 384
            # Reduction-A block
            net = block_reduction_a(net, 'Mixed_6a')
            if add_and_check_final('Mixed_6a', net): return net, end_points

            # 17 x 17 x 1024
            # 7 x Inception-B blocks
            for idx in range(7):
                block_scope = 'Mixed_6' + chr(ord('b') + idx)
                net = block_inception_b(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points

            # 17 x 17 x 1024
            # Reduction-B block
            net = block_reduction_b(net, 'Mixed_7a')
            if add_and_check_final('Mixed_7a', net): return net, end_points

            # 8 x 8 x 1536
            # 3 x Inception-C blocks
            for idx in range(3):
                block_scope = 'Mixed_7' + chr(ord('b') + idx)
                net = block_inception_c(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points

            # Final pooling and prediction
            # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
            # can be set to False to disable pooling here (as in resnet_*()).
            with tf.variable_scope('Logits'):
                # 8 x 8 x 1536
                kernel_size = net.get_shape()[1:3]
                if kernel_size.is_fully_defined():
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                          scope='AvgPool_1a')
                else:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                         name='global_pool')
                    end_points['global_pool'] = net
            if not num_classes:
                return net, end_points
            # 1 x 1 x 1536
            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
            net = slim.flatten(net, scope='PreLogitsFlatten')
            end_points['PreLogitsFlatten'] = net
            # 1536
            logits = slim.fully_connected(
                net, num_classes,
                activation_fn=None,
                reuse=reuse,
                scope='Logits',
                weights_regularizer=slim.l2_regularizer(0.05))
            end_points['Logits'] = logits
            end_points['Predictions'] = tf.nn.softmax(logits,
                                                      name='Predictions')
    return logits, end_points

def calculate_metrics(labels,predictions):

    auc,auc_op = tf.metrics.auc(labels=labels,
                                predictions=predictions)

    tp,tp_op = tf.metrics.true_positives(labels=labels,
                                         predictions=predictions)
    tn,tn_op = tf.metrics.true_negatives(labels=labels,
                                         predictions=predictions)
    fp,fp_op = tf.metrics.false_positives(labels=labels,
                                          predictions=predictions)
    fn,fn_op = tf.metrics.false_negatives(labels=labels,
                                          predictions=predictions)
    return (auc,auc_op),(tp,tp_op),(tn,tn_op),(fp,fp_op),(fn,fn_op)

def process_metrics(metric_dict):
    def get_metrics(tp,tn,fp,fn):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1score = ((precision * sens) / (precision + sens)) * 2

        return [sens,spec,precision,accuracy,f1score]

    general_keys = [
        'general_tp',
        'general_tn',
        'general_fp',
        'general_fn',
        'general_auc'
        ]
    pc_keys = [
        'per_class_tp',
        'per_class_tn',
        'per_class_fp',
        'per_class_fn',
        'per_class_auc'
    ]

    general_metrics = get_metrics(metric_dict['general_tp'],
                                  metric_dict['general_tn'],
                                  metric_dict['general_fp'],
                                  metric_dict['general_fn'])
    general_metrics.append(metric_dict['general_auc'])

    pc_counts = [[],[],[],[]]
    pc_auc = []
    pc_keys = metric_dict['per_class_auc'].keys()
    for key in pc_keys:
        tp = metric_dict['per_class_tp'][key]
        tn = metric_dict['per_class_tn'][key]
        fp = metric_dict['per_class_fp'][key]
        fn = metric_dict['per_class_fn'][key]
        pc_counts[0].append(tp)
        pc_counts[1].append(tn)
        pc_counts[2].append(fp)
        pc_counts[3].append(fn)
        pc_auc.append(metric_dict['per_class_auc'][key])
    pc_counts = np.array(pc_counts)
    pc_metrics = get_metrics(pc_counts[0,:],
                             pc_counts[1,:],
                             pc_counts[2,:],
                             pc_counts[3,:])
    pc_metrics.append(np.array(pc_auc))
    pc_metrics = np.stack(pc_metrics,axis=1)

    return general_metrics,pc_metrics

parser = argparse.ArgumentParser(
    prog = 'multi_input_inception.py',
    description = 'Multiple input InceptionV4.'
)

parser.add_argument('--paths',dest='paths',
                    type=str,
                    nargs='+',
                    required=True,
                    help = 'Paths to folders containing tfrecords.')
parser.add_argument('--cw_paths',dest='cw_paths',
                    type=str,
                    nargs='+',
                    help = 'Paths paths to class weights.')
parser.add_argument('--mode',dest='mode',
                    type=str,
                    default='train',
                    help = 'Network running mode.')
parser.add_argument('--checkpoint_path',dest='checkpoint_path',
                    type=str,
                    default=None,
                    help = 'Path to checkpoint.')

parser.add_argument('--iterations',dest='iterations',
                    type=int,
                    default=100000,
                    help = 'No. of iterations.')
parser.add_argument('--batch_size',dest='batch_size',
                    type=int,
                    default=8,
                    help = 'Batch size.')
parser.add_argument('--routine',dest='routine',
                    type=str,
                    default='separate',
                    help='Training routine.')

args = parser.parse_args()

batch_size = args.batch_size

if args.mode == 'train':
    is_training = True
else:
    is_training = False

inputs = []
truths = []
N_INPUTS = 0

if args.mode == 'train':
    try:
        len(args.cw_paths)
        cw_arg = 'cw'
    except:
        cw_arg = 'no_cw'

    try:
        os.makedirs(os.path.join(os.getcwd(),
                                 'checkpoint_{}'.format(args.routine)))
    except:
        pass

    for path in args.paths:
        image_path_list = glob('{}/*tfrecord'.format(path))
        tmp_ig,tmp_fn,tmp_cl,tmp_pu = image_generator(path,
                                                      'train',
                                                      args.batch_size)
        tmp_pu = tf.divide(
            tf.cast(tmp_pu,tf.float32),
            tf.constant(100.))
        inputs.append(tmp_ig)
        is_cancer = tf.mod(tmp_cl,2)

        purity_cancer_mask = tf.cast(
            tf.multiply(
                tf.expand_dims(is_cancer,axis=1),
                tf.cast(tf.one_hot(tmp_cl,depth=N_CLASSES),tf.int64),
                ),
            tf.float32
            ) * tf.expand_dims(tmp_pu,axis=1)
        purity_cancer_normal_mask = tf.cast(
            tf.multiply(
                tf.expand_dims(is_cancer,axis=1),
                tf.cast(tf.one_hot(tf.clip_by_value(tmp_cl - 1,0,N_CLASSES),
                                   depth=N_CLASSES),tf.int64)
                ),
            tf.float32
            ) * tf.expand_dims(1 - tmp_pu,axis=1)
        purity_normal_mask = tf.cast(
            tf.multiply(
                tf.expand_dims(1 - is_cancer,axis=1),
                tf.cast(tf.one_hot(tmp_cl,depth=N_CLASSES),tf.int64)),
            tf.float32)

        truth_one_hot = tf.cast(tf.add_n(
            [purity_cancer_mask,purity_cancer_normal_mask,purity_normal_mask]
        ),tf.float32)

        truths.append(truth_one_hot)
        N_INPUTS += 1

    stems = []
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.05),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training':is_training}):
        for i in range(N_INPUTS):
            stem,_ = inception_v4_base(
                inputs[i],
                final_endpoint='Mixed_5a',
                scope='Stem_{}'.format(i))
            stems.append(stem)

    full_batch = tf.concat(stems,axis=0)
    full_batch_truths = tf.concat(truths,axis=0)

    inception_full_branches,_ = inception_v4_from_stem(
        full_batch,
        scope='Inception_v4',
        final_endpoint='Logits',
        num_classes=N_CLASSES,reuse=False)

    full_branch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=full_batch_truths,
        logits=inception_full_branches)

    stem_branches = []
    stem_branches_loss = []
    stem_branches_auc = []
    stem_branches_auc_ops = []
    weighted_full_loss = []
    for i in range(N_INPUTS):
        tmp_inception,_ = inception_v4_from_stem(
            stems[i],
            scope='Inception_v4',
            final_endpoint='Logits',
            num_classes=N_CLASSES,reuse=True)

        stem_branches.append(tmp_inception)
        tmp_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=truths[i],
            logits=tmp_inception)

        try:
            curr_cw_path = args.cw_paths[i]
        except:
            curr_cw_path = None
            total_samples = args.iterations * args.batch_size // 2

        if curr_cw_path != None:
            with open(curr_cw_path) as o:
                lines = o.readlines()
            lines = [l.strip().split() for l in lines]
            counts = {int(l[0]):int(l[1]) for l in lines}
            count_array = np.zeros([N_CLASSES])
            for c in counts:
                count_array[c] = counts[c]
            count_max = np.max(count_array[np.nonzero(count_array)])
            weight_array = 1 - ((count_array / count_max) * 0.9)

            weight_array = weight_array * np.where(count_array > 0,1.,0.)
            total_samples = np.sum(count_array)
            weight_tensor = tf.convert_to_tensor(weight_array,dtype=tf.float32)
            weight_tensor = tf.expand_dims(weight_tensor,axis=0)

        else:
            weight_tensor = tf.ones([1,N_CLASSES],dtype=tf.float32)

        corrected_weight_tensor = truths[i] * weight_tensor
        sample_weight = tf.reduce_max(corrected_weight_tensor,
                                      axis=1,keepdims=True)

        tmp_loss = tf.reduce_mean(tmp_loss * sample_weight)
        tfl = full_branch_loss[batch_size*i:batch_size*(i+1)] * sample_weight
        weighted_full_loss.append(tfl)
        stem_branches_loss.append(tmp_loss)

        tmp_auc,tmp_auc_op = tf.metrics.auc(
              labels = tf.where(truths[i] > 0.5,
                                tf.ones_like(truths[i]),
                                tf.zeros_like(truths[i])),
              predictions = tf.nn.softmax(tmp_inception,axis=-1)
            )
        stem_branches_auc.append(tmp_auc)
        stem_branches_auc_ops.append(tmp_auc_op)

    full_branch_loss = tf.concat(weighted_full_loss,axis=0)
    full_branch_loss = tf.reduce_mean(full_branch_loss)

    stem_var_list = [[] for _ in range(N_INPUTS)]
    full_branches_var_list = []
    for var in tf.trainable_variables():
        if 'Stem' in var.name:
            stem_idx = int(var.name.split('/')[0].split('_')[1])
            stem_var_list[stem_idx].append(var)
            if args.routine == 'one':
                full_branches_var_list.append(var)
        else:
            full_branches_var_list.append(var)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        LR,
        global_step=global_step,
        decay_steps=total_samples * 2 / args.batch_size,
        decay_rate=0.94,
        staircase=True,
        name='exponential_decay_learning_rate'
    )
    full_branch_optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        full_branch_train_op = full_branch_optimizer.minimize(
            full_branch_loss,global_step=global_step,
            var_list=full_branches_var_list)

    stem_train_ops = []
    for i in range(N_INPUTS):
        opt = tf.train.AdamOptimizer(learning_rate)
        if args.routine == 'joint' or args.routine == 'joint_and_separate':
            tmp_var_list = list(set(full_branches_var_list + stem_var_list[i]))
        else:
            tmp_var_list = stem_var_list[i]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            tmp_train_op = opt.minimize(
                stem_branches_loss[i],global_step=global_step,
                var_list=tmp_var_list)
        stem_train_ops.append(tmp_train_op)

    if args.routine == 'joint':
        all_train_ops = stem_train_ops

    elif args.routine == 'one':
        all_train_ops = [full_branch_train_op]
    else:
        all_train_ops = stem_train_ops + [full_branch_train_op]

    all_train_ops = tf.group(all_train_ops)

    saver = tf.train.Saver()

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    for variable in slim.get_model_variables():
        print(variable)
        summaries.add(
            tf.summary.histogram(variable.op.name, variable)
            )

    for i,loss in enumerate(stem_branches_loss):
        summaries.add(tf.summary.scalar('loss_{}'.format(i),loss))
    for i,auc in enumerate(stem_branches_auc):
        summaries.add(tf.summary.scalar('auc_{}'.format(i),auc))

    summary_op = tf.summary.merge(list(summaries),name='summary_op')

    with tf.Session() as sess:
        tf.set_random_seed(42)

        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter(
            'summary_{}_{}'.format(args.routine,cw_arg),
            sess.graph)
        print('Training...')
        for i in range(args.iterations):
            sess.run(all_train_ops)
            l = sess.run([stem_branches_loss,full_branch_loss])
            sess.run(stem_branches_auc_ops)
            print(sess.run(tf.get_collection(tf.GraphKeys.VARIABLES,'Inception_v4/Mixed_7d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean:0')[0]))
            if i % save_checkpoint_steps == 0:
                print('saving checkpoint')
                saver.save(sess,
                           os.path.join(
                               'checkpoint_{}_{}'.format(args.routine,cw_arg),
                               'multi_input_inception.ckpt'),
                           global_step=i)
            if i % save_summary_steps == 0:
                print('\t',i,l)
                print('\t',sess.run(stem_branches_auc))
                summary = sess.run(summary_op)
                writer.add_summary(summary,i)
                sess.run(tf.local_variables_initializer())

        sess.run(summary_op)
        writer.add_summary(summary,i)
        saver.save(sess,
                   os.path.join(
                       'checkpoint_{}_{}'.format(args.routine,cw_arg),
                       'multi_input_inception.ckpt'),
                   global_step=i)

if args.mode == 'test':
    metric_list = []
    update_ops_list = []
    predictions = []
    truths = []

    for i in range(len(args.paths)):
        metrics = {
            'general_tp':0,
            'general_tn':0,
            'general_fp':0,
            'general_fn':0,
            'general_auc':0,
            'per_class_tp':{},
            'per_class_tn':{},
            'per_class_fp':{},
            'per_class_fn':{},
            'per_class_auc':{}
        }
        update_ops = {
            'general_tp':0,
            'general_tn':0,
            'general_fp':0,
            'general_fn':0,
            'general_auc':0,
            'per_class_tp':{},
            'per_class_tn':{},
            'per_class_fp':{},
            'per_class_fn':{},
            'per_class_auc':{}
        }
        path = args.paths[i]
        if i == 0: reuse = False
        else: reuse = True

        N_INPUTS += 1
        image_path_list = glob('{}/*tfrecord'.format(path))
        tmp_ig,tmp_fn,tmp_cl,tmp_pu = image_generator(path,
                                                      'test',
                                                      args.batch_size)

        truth_one_hot = tf.one_hot(tf.cast(tmp_cl,tf.int64),
                                   depth=N_CLASSES)

        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training':False}):

            stem,_ = inception_v4_base(
                tmp_ig,
                final_endpoint='Mixed_5a',
                scope='Stem_{}'.format(i))

        tmp_inception,_ = inception_v4_from_stem(
            stem,
            scope='Inception_v4',
            final_endpoint='Logits',
            is_training=False,
            dropout_keep_prob=1.0,
            num_classes=N_CLASSES,reuse=reuse)

        binary_pred = tf.one_hot(tf.argmax(tmp_inception, dimension=-1),
                                 depth = N_CLASSES)

        # General metrics
        tmp = calculate_metrics(labels=truth_one_hot,
                                predictions=binary_pred)
        (auc,auc_op),(tp,tp_op),(tn,tn_op),(fp,fp_op),(fn,fn_op) = tmp

        metrics['general_auc'] = auc
        update_ops['general_auc'] = auc_op
        metrics['general_tp'] = tp
        update_ops['general_tp'] = tp_op
        metrics['general_tn'] = tn
        update_ops['general_tn'] = tn_op
        metrics['general_fp'] = fp
        update_ops['general_fp'] = fp_op
        metrics['general_fn'] = fn
        update_ops['general_fn'] = fn_op

        # Per class metrics
        split_prediction = tf.split(binary_pred,N_CLASSES,axis=-1)
        split_truth = tf.split(truth_one_hot,N_CLASSES,axis=-1)

        for pred,truth in zip(split_prediction,split_truth):
            tmp = calculate_metrics(labels=truth,
                                    predictions=pred)
            (auc,auc_op),(tp,tp_op),(tn,tn_op),(fp,fp_op),(fn,fn_op) = tmp
            metrics['per_class_auc'][pred.name] = auc
            update_ops['per_class_auc'][pred.name] = auc_op
            metrics['per_class_tp'][pred.name] = tp
            update_ops['per_class_tp'][pred.name] = tp_op
            metrics['per_class_tn'][pred.name] = tn
            update_ops['per_class_tn'][pred.name] = tn_op
            metrics['per_class_fp'][pred.name] = fp
            update_ops['per_class_fp'][pred.name] = fp_op
            metrics['per_class_fn'][pred.name] = fn
            update_ops['per_class_fn'][pred.name] = fn_op
        metric_list.append(metrics)
        update_ops_list.append(update_ops)
        predictions.append(tf.argmax(tmp_inception, dimension=-1))
        truths.append(tmp_cl)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        tf.set_random_seed(42)
        sess.run(tf.tables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess,args.checkpoint_path)

        print('Testing...')
        for metrics,update_ops in zip(metric_list,update_ops_list):
            try:
                i = 0
                while True:
                    i += 1
                    sess.run(update_ops)
                    if i % 1000 == 0:
                        print(i)
                    print(sess.run([tmp_inception,binary_pred]))
            except tf.errors.OutOfRangeError:
                print('Finished one stem...')

            metric_info = sess.run(metrics)
            m = process_metrics(metric_info)
            print(','.join([str(x) for x in m[0]]))
            print('\n'.join([','.join([str(y) for y in x]) for x in m[1]]))
