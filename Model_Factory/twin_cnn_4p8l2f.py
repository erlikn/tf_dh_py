# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

import Model_Factory.model_base as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]

    modelShape = kwargs.get('modelShape')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)

    ############# CONV1_TWIN 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv1', images, kwargs.get('imageChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  wd, **kwargs)
    # BATCH_NORM -> CONV1_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV2_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv2', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  wd, **kwargs)
    # BATCH_NORM -> CONV2_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # POOLING1 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool1')
    ############# CONV3_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv3', pool, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  wd, **kwargs)
    # BATCH_NORM -> CONV3_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV4_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_module('conv4', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  wd, **kwargs)
   # BATCH_NORM -> CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # POOLING2 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    ############# CONV5 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv5', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[4]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV5
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV6 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv6', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[5]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV6
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # POOLING3 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV7 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv7', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[6]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV7
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV8 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv8', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[7]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV8
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # POOLING4 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV9 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv9', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[8]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV9
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV10 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv10', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[9]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV10
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # POOLING5 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV11 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv11', pool, prevExpandDim,
                                                         {'cnn3x3': modelShape[10]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV11
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV12 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_module('conv12', fireOut, prevExpandDim,
                                                         {'cnn3x3': modelShape[11]},
                                                         wd, **kwargs)
    # BATCH_NORM -> CONV12
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### DROPOUT after CONV12
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    ###### Prepare for fully connected layers
    # Reshape firout - flatten
    prevExpandDim = (kwargs.get('imageHeight')//(2*2*2*2*2))*(kwargs.get('imageWidth')//(2*2*2*2*2))*prevExpandDim
    fireOutFlat = tf.reshape(fireOut, [batchSize, -1])

    ############# FC1 layer with 1024 outputs
    fireOut, prevExpandDim = model_base.fc_fire_module('fc1', fireOutFlat, prevExpandDim,
                                                       {'fc': modelShape[12]},
                                                       wd, **kwargs)
    # BATCH_NORM -> FC1
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# FC2 layer with 8 outputs
    fireOut, prevExpandDim = model_base.fc_regression_module('fc2', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('outputSize')},
                                                             wd, **kwargs)

    return fireOut

def loss(pHAB, tHAB, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pHAB, tHAB, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)