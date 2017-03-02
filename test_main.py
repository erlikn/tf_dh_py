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
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug
import Data_IO.data_input as data_input


with open('Model_Settings/170126_SIN_W.json') as data_file:
    modelParams = json.load(data_file)

#### Override Model Parameters for Batch Normalization and Weight Normalization
modelParams['batchNorm'] = False
modelParams['weightNorm'] = False
####

# import corresponding model name as model_cnn
model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

PHASE = 'test'
####################################################
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('printOutStep', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 250,
                            """Number of batches to run.""")
####################################################
def _get_control_params():
    modelParams['phase'] = PHASE

    modelParams['existingParams'] = None

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']

    modelParams['batchNorm'] = False
    modelParams['weightNorm'] = False


def test():
    _get_control_params()

    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])

    lossValueSum = 0

    _setupLogging(os.path.join(modelParams['testLogDir'], "genlog"))

    with tf.Graph().as_default():

        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images and transformation for model_cnn.
        #batchImageOrig, batchImage, batchPOrig, batchHAB, tfrecFileID
        batchImageOrig, images, batchPOrig, tHAB, tfrecFileID = data_input.inputs(**modelParams)

        # Build a Graph that computes the HAB predictions from the
        # inference model.
        pHAB = model_cnn.inference(images, **modelParams)

        # Calculate loss.
        loss = model_cnn.loss(pHAB, tHAB, **modelParams)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        opTest = model_cnn.test(loss, globalStep, **modelParams)
        ##############################

        # Build the summary operation based on the TF collection of Summaries.
        summaryOp = tf.summary.merge_all()

        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        opCheck = tf.add_check_numerics_ops()
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config = config)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)

        # restore a saver.
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, modelParams['trainLogDir']+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summaryWriter = tf.summary.FileWriter(modelParams['testLogDir'], sess.graph)

        for step in xrange(modelParams['maxSteps']):
            startTime = time.time()
            _, lossValue = sess.run([opTest, loss])
            duration = time.time() - startTime

            assert not np.isnan(lossValue), 'Model diverged with loss = NaN'

            lossValueSum += np.sqrt(lossValue*(2/(modelParams['activeBatchSize']*8)))
            lossValueAvgPixel = lossValueSum/(step+1)
            
            if step % FLAGS.printOutStep == 0:
                numExamplesPerStep = modelParams['activeBatchSize']
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch), pixelErr = %.3f')
                logging.info(format_str % (datetime.now(), step, lossValue,
                                     examplesPerSec, secPerBatch, lossValueAvgPixel))
            
            if step % FLAGS.summaryWriteStep == 0:
                summaryStr = sess.run(summaryOp)
                summaryWriter.add_summary(summaryStr, step)

            # Save the model checkpoint periodically.
            #if step % FLAGS.modelCheckpointStep == 0 or (step + 1) == FLAGS.maxSteps:
            #    checkpoint_path = os.path.join(FLAGS.testLogDir, 'model.ckpt')
            #    saver.save(sess, checkpoint_path, global_step=step)
            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or (step+1 == modelParams['maxSteps']):
                print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
                        (step/modelParams['maxSteps'], lossValueAvgPixel, duration/60, ((duration*modelParams['maxSteps'])/(step+1))/60))


def _setupLogging(logPath):
    # cleanup
    if (os.path.isfile(logPath)):
        os.remove(logPath)
        
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logPath,
                        filemode='w')
    
    # also write out to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    
    logging.info("Logging setup complete to %s" % logPath)

def main(argv=None):  # pylint: disable=unused-argumDt
    #if argv[1]:
    #    jsonFileName = argv[1]
    #else:
    #    print('please choose a json file')
    #    return
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print(modelParams['trainLogDir'])
    print(modelParams['testLogDir'])
    if input("(Overwrite WARNING) Are you sure about this logs directory? ") != "yes":
        print("Please consider changing logs directory in order to avoid overwrite!")
        return
    if tf.gfile.Exists(modelParams['testLogDir']):
        tf.gfile.DeleteRecursively(modelParams['testLogDir'])
    tf.gfile.MakeDirs(modelParams['testLogDir'])
    test()
    print('Model Shape = %s' % modelParams['modelShape'])
    print('BNORM = %s , WNORM = %s' % (modelParams['batchNorm'], modelParams['weightNorm']))

if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
