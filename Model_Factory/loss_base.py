from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def add_loss_summaries(total_loss, batchSize):
    """Add summaries for losses in calusa_heatmap model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='Average')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Individual average loss
#    lossPixelIndividual = tf.sqrt(tf.multiply(total_loss, 2/(batchSize*4))) # dvidied by (8/2) = 4 which is equal to sum of 2 of them then sqrt will result in euclidean pixel error
#    tf.summary.scalar('Average_Pixel_Error_Real', lossPixelIndividual)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _l2_loss(pred, tval): # batchSize=Sne
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

    l2_loss = tf.nn.l2_loss(tf.subtract(pred, tval), name="l2_loss")
    tf.add_to_collection('losses', l2_loss)

    #l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    #mse = tf.reduce_mean(tf.square(logits - HAB), name="mse")
    #tf.add_to_collection('losses', mse)

    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, labels, name='cross_entropy_per_example')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss(pred, tval, lossFunction):
    """
    Choose the proper loss function and call it.
    """
    if lossFunction == 'L2':
        return _l2_loss(pred, tval)
