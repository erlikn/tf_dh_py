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

import re
import glob

import tensorflow as tf
import numpy as np


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.histogram_summary(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP_16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd, trainable=True):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    
    Returns:
      Variable Tensor
    """     
    dtype = tf.float16 if USE_FP_16 else tf.float32
    # tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    if isinstance(initializer, np.ndarray):
        var = tf.get_variable(name, initializer=initializer, dtype=dtype, trainable=trainable)
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    #if wd is not None:
    #    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #    tf.add_to_collection('losses', weight_decay)
        
    return var

def _batch_norm(tensorConv):
    # Calc batch mean for parallel module
    batchMean, batchVar = tf.nn.moments(tensorConv, axes=[0]) # moments along x,y
    scale = tf.Variable(tf.ones(tensorConv.get_shape()[-1]))
    beta = tf.Variable(tf.zeros(tensorConv.get_shape()[-1]))
    epsilon = 1e-3
    if USE_FP_16:
        scale = tf.cast(scale, tf.float16)
        beta = tf.cast(beta, tf.float16)
        epsilon = tf.cast(epsilon, tf.float16)
    batchNorm = tf.nn.batch_normalization(tensorConv, batchMean, batchVar, beta, scale, epsilon)
    return batchNorm
    


def conv_fire_parallel_module(name, prevLayerOut, prevLayerDims, fireDimsSingleModule, wd=None, **kwargs):
    """ 
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDims:      size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')

    numParallelModules = kwargs.get('numParallelModules') # 2
    # Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(3, numParallelModules, prevLayerOut)
    prevLayerIndivDims = prevLayerDims / numParallelModules

    with tf.variable_scope(name):
        with tf.variable_scope('cnn3x3') as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[3, 3, prevLayerIndivDims, fireDimsSingleModule['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut[0].get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, prevLayerIndivDims, fireDimsSingleModule['cnn3x3']],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.random_normal_initializer(stddev=stddev) if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            for prl in range(numParallelModules):
                conv = tf.nn.conv2d(prevLayerOut[prl], kernel, [1, 1, 1, 1], padding='SAME')
                
                if kwargs.get('weightNorm'):
                    # calc weight norm
                    conv = _batch_norm(conv)
                
                #if existingParams is not None and layerName in existingParams:
                #    biases = tf.get_variable('biases',
                #                             initializer=existingParams[layerName]['biases'], dtype=dtype)
                #else:
                #    biases = tf.get_variable('biases', [fireDimsSingleModule['cnn3x3']],
                #                             initializer=tf.constant_initializer(0.0),
                #                             dtype=dtype)

                #bias = tf.nn.bias_add(conv, biases)
                convReluPrl = tf.nn.relu(conv, name=scope.name)
                # Concatinate results along last dimension to get one output tensor
                if prl is 0:
                    convRelu = convReluPrl
                else:    
                    convRelu = tf.concat(3, [convRelu, convReluPrl])

            _activation_summary(convRelu)

        return convRelu, numParallelModules*fireDimsSingleModule['cnn3x3']

def conv_fire_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    with tf.variable_scope(name):
        with tf.variable_scope('cnn3x3') as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[3, 3, prevLayerDim, fireDims['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, prevLayerDim, fireDims['cnn3x3']],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.random_normal_initializer(stddev=stddev) if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            conv = tf.nn.conv2d(prevLayerOut, kernel, [1, 1, 1, 1], padding='SAME')

            if kwargs.get('weightNorm'):
                # calc weight norm
                conv = _batch_norm(conv)

            #if existingParams is not None and layerName in existingParams:
            #    biases = tf.get_variable('biases',
            #                             initializer=existingParams[layerName]['biases'], dtype=dtype)
            #else:
            #    biases = tf.get_variable('biases', [fireDims['cnn3x3']],
            #                             initializer=tf.constant_initializer(0.0),
            #                             dtype=dtype)

            #bias = tf.nn.bias_add(conv, biases)
            convRelu = tf.nn.relu(conv, name=scope.name)
            _activation_summary(convRelu)

        return convRelu, fireDims['cnn3x3']

def fc_fire_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.random_normal_initializer(stddev=stddev) if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = _batch_norm(fc)

            #biases = tf.get_variable('biases', fireDims['fc'],
            #                         initializer=tf.constant_initializer(0.0), dtype=dtype)
            #bias = tf.nn.bias_add(fc, biases)
            fcRelu = tf.nn.relu(fc, name=scope.name)
            _activation_summary(fcRelu)
        
        return fcRelu, fireDims['fc']

def fc_regression_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.random_normal_initializer(stddev=stddev) if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = _batch_norm(fc)

            #biases = tf.get_variable('biases', fireDims['fc'],
            #                         initializer=tf.constant_initializer(0.0), dtype=dtype)
            #bias = tf.nn.bias_add(fc, biases)
            _activation_summary(fc)
        
        return fc, fireDims['fc']

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    
    modelShape = kwargs.get('modelShape') 
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    batchSize = kwargs.get('activeBatchSize', None)
    
    ############# CONV1_TWIN 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_parallel_module('conv1', images, kwargs.get('imageChannels'),
                                              {'cnn3x3': modelShape[0]},
                                              wd, **kwargs)
    # calc batch norm CONV1_TWIN
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ############# CONV2_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_parallel_module('conv2', fireOut, prevExpandDim,
                                              {'cnn3x3': modelShape[1]},
                                              wd, **kwargs)
    # calc batch norm CONV2_TWIN
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ###### Pooling1 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool1')
    ############# CONV3_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_parallel_module('conv3', pool, prevExpandDim,
                                              {'cnn3x3': modelShape[2]},
                                              wd, **kwargs)
    # calc batch norm CONV3_TWIN
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ############# CONV4_TWIN 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_parallel_module('conv4', fireOut, prevExpandDim,
                                              {'cnn3x3': modelShape[3]},
                                              wd, **kwargs)
   # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ###### Pooling2 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    ############# CONV5 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_module('conv5', pool, prevExpandDim,
                                              {'cnn3x3': modelShape[4]},
                                              wd, **kwargs)
    # calc batch norm CONV5
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ############# CONV6 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_module('conv6', fireOut, prevExpandDim,
                                              {'cnn3x3': modelShape[5]},
                                              wd, **kwargs)
    # calc batch norm CONV6
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ###### Pooling2 2x2 wit stride 2
    pool = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV7 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_module('conv7', pool, prevExpandDim,
                                              {'cnn3x3': modelShape[6]},
                                              wd, **kwargs)
    # calc batch norm CONV7
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ############# CONV8 3x3 conv, 64 input dims, 64 output dims (filters)
    fireOut, prevExpandDim = conv_fire_module('conv8', fireOut, prevExpandDim,
                                              {'cnn3x3': modelShape[7]},
                                              wd, **kwargs)
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ###### DROPOUT after CONV8
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase')=='train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    ###### Prepare for fully connected layers
    # Reshape firout - flatten 
    prevExpandDim = (kwargs.get('imageSize')//(2*2*2))*(kwargs.get('imageSize')//(2*2*2))*prevExpandDim
    fireOutFlat = tf.reshape(fireOut, [batchSize, -1])
    
    ############# FC1 layer with 1024 outputs
    fireOut, prevExpandDim = fc_fire_module('fc1', fireOutFlat, prevExpandDim,
                                            {'fc': modelShape[8]},
                                            wd, **kwargs)
    # calc batch norm FC1
    if kwargs.get('batchNorm'):
        fireOut = _batch_norm(fireOut)
    ############# FC2 layer with 8 outputs
    fireOut, prevExpandDim = fc_regression_module('fc2', fireOut, prevExpandDim,
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
    #if not batch_size:
    #    batch_size = kwargs.get('train_batch_size')
    
    #l1_loss = tf.abs(tf.subtract(logits, HAB), name="abs_loss")
    #l1_loss_mean = tf.reduce_mean(l1_loss, name='abs_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)
    
    l2_loss = tf.nn.l2_loss(tf.subtract(pHAB, tHAB), name="l2_loss")
    tf.add_to_collection('losses', l2_loss)

    #l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    #mse = tf.reduce_mean(tf.square(logits - HAB), name="mse")
    #tf.add_to_collection('losses', mse)
    
    # Calculate the average cross entropy loss across the batch.
#     labels = tf.cast(labels, tf.int64)
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits, labels, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
    
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss, batchSize):
    """Add summaries for losses in calusa_heatmap model.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Individual average loss
    lossPixelIndividual = tf.sqrt(tf.multiply(total_loss, 2/(batchSize*8)))
    tf.summary.scalar('Average_Pixel_Error', lossPixelIndividual)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(loss, globalStep, **kwargs):
    ##############################
    # Variables that affect learning rate.
    numExamplesPerEpochForTrain = kwargs.get('numExamplesPerEpoch') #56016 #56658   # <--------------?????
    #numBatchesPerEpoch = numExamplesPerEpochForTrain / kwargs.get('activeBatchSize')
    print("Using %d example for phase %s" % (numExamplesPerEpochForTrain, kwargs.get('phase'))) # <--------------?????

    # 0.005 for [0,30000] -> 0.0005 for [30001,60000], 0.00005 for [60001, 90000]
    # [30000, 60000]
    boundaries = [kwargs.get('numEpochsPerDecay'),
                  2*kwargs.get('numEpochsPerDecay')]
    #[0.005, 0.0005, 0.00005]
    values = [kwargs.get('initialLearningRate'), 
              kwargs.get('initialLearningRate')*kwargs.get('learningRateDecayFactor'),
              kwargs.get('initialLearningRate')*kwargs.get('learningRateDecayFactor')*kwargs.get('learningRateDecayFactor')]
    learningRate = tf.train.piecewise_constant(globalStep, boundaries, values)
    tf.summary.scalar('learning_rate', learningRate)
    momentum = kwargs.get('momentum')
        
    # Generate moving averages of all losses and associated summaries.
    lossAveragesOp = _add_loss_summaries(loss, kwargs.get('activeBatchSize', None))
    
    # Compute gradients.
    tvars = tf.trainable_variables()
    with tf.control_dependencies([lossAveragesOp]):
        #optim = tf.train.AdamOptimizer(learning_rate=learningRate, epsilon=0.1)
        optim = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum)
        #optim = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        #gradsNvars = optim.compute_gradients(loss, tvars)
        #gradsNvars, norm = tf.clip_by_global_norm(gradsNvars, kwargs.get('clipNorm')
        grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), kwargs.get('clipNorm'))

    # Apply gradients.
    #applyGradientOp = opt.apply_gradients(grads, global_step=globalStep)
    #train_op = opt.apply_gradients(gradsNvars, global_step=globalStep)
    opApplyGradients = optim.apply_gradients(zip(grads, tvars), global_step=globalStep)

        
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():    
        tf.summary.histogram(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in zip(grads, tvars):
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    with tf.control_dependencies([opApplyGradients]):
        opTrain = tf.no_op(name='train')

    return opTrain

def test(loss, globalStep, **kwargs):
    ##############################
    # Variables that affect learning rate.
    numExamplesPerEpochForTest = kwargs.get('numExamplesPerEpoch') #56016 #56658   # <--------------?????
    print("Using %d example for phase %s" % (numExamplesPerEpochForTest, kwargs.get('phase'))) # <--------------?????
    
    # Generate moving averages of all losses and associated summaries.
    lossAveragesOp = _add_loss_summaries(loss, kwargs.get('activeBatchSize', None))
    
    with tf.control_dependencies([]):
        opTest = tf.no_op(name='test')

    return opTest