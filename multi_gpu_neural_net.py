"""
Set of utility functions for multi-GPU implemmentations in tensorflow.
"""

import os
import sys
import time
import argparse
import pickle
from math import floor
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import PIL

PS_OPS = [
      'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
      'MutableHashTableOfTensors', 'MutableDenseHashTable',
      'QueueEnqueueManyV2','FIFOQueueV2'
]

def get_available_gpus():
    """
    Function to get all available GPUs.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def assign_to_device(device,ps_device=None):
    """
    Function to place variables in ps_device and everything else in device.
    This has shown to improve the training of several models in a
    multi-GPU context.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign

def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Towers are synchronously updated with this.
    Args:
    tower_grads: Output from optimizer.compute_gradients(loss)
    Returns:
            List of pairs of (gradient, variable) where the gradient has been
            averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
