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

PHASE = 'test'
# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()
jsonToRead = '170126_SIN_B.json'
print("Reading %s" % jsonToRead)
with open('Model_Settings/'+jsonToRead) as data_file:
    modelParams = json.load(data_file)

# import input & output modules 
import Data_IO.data_input as data_input
import Data_IO.data_output as data_output

# import corresponding model name as model_cnn, specifed at json file
model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 50,
                            """Number of batches to run.""")
####################################################
def _get_control_params():
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None
    if modelParams['initExistingWeights'] is not None and modelParams['initExistingWeights'] != "":
        modelParams['existingParams'] = np.load(modelParams['initExistingWeights']).item()

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']


def test():
    _get_control_params()

    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])    

    #meanImgFi1000le = os.path.join(FLAGS.dataDir, "meta")
    #if not os.path.isfile(meanImgFile):
    #    raise ValueError("Warning, no meta file found at %s" % meanImgFile)
    #else:
    #    with open(meanImgFile, "r") as inMeanFile:
    #        meanInfo = json.load(inMeanFile)
    #
    #    meanImg = meanInfo['mean']
    #
    #    # also load the target output sizes
    #    params['targSz'] = meanInfo["targSz"]

    _setupLogging(os.path.join(modelParams['testLogDir'], "genlog"))

    with tf.Graph().as_default():
        # BGR to RGB
        #params['meanImg'] = tf.constant(meanImg, dtype=tf.float32)

        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images and transformation for model_cnn.
        imagesOrig, images, pOrig, tHAB, tfrecFileIDs = data_input.inputs(**modelParams)

        # Build a Graph that computes the HAB predictions from the
        # inference model.
        pHAB = model_cnn.inference(images, **modelParams)

        # Calculate loss.
        loss = model_cnn.loss(pHAB, tHAB, **modelParams)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

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
        
        lossValueSum = 0
        durationSum = 0
        HABperPixelsum = 0
        maxErrbatchsum=0

        print('Warping images with batch size %d in %d steps' % (modelParams['activeBatchSize'], modelParams['maxSteps']))

        for step in xrange(modelParams['maxSteps']):
            startTime = time.time()
            evtHAB, evpHAB, lossValue = sess.run([tHAB, pHAB, loss])
            duration = time.time() - startTime
            durationSum += duration
            HABRES = evtHAB-evpHAB
            HABperPixel = 0
            maxErrbatch = 0
            for i in xrange(modelParams['activeBatchSize']):
                H = np.asarray([[HABRES[i][0],HABRES[i][1],HABRES[i][2],HABRES[i][3]],
                                [HABRES[i][4],HABRES[i][5],HABRES[i][6],HABRES[i][7]]], np.float32)
                HABperPixel += np.sqrt((H*H).sum(axis=0)).mean()
                maxErr = np.asarray([[evtHAB[i][0],evtHAB[i][1],evtHAB[i][2],evtHAB[i][3]],
                                     [evtHAB[i][4],evtHAB[i][5],evtHAB[i][6],evtHAB[i][7]]], np.float32)
                maxErrbatch += np.sqrt((maxErr*maxErr).sum(axis=0)).mean()
            HABperPixel = HABperPixel/modelParams['activeBatchSize']
            maxErrbatch = maxErrbatch/modelParams['activeBatchSize']
            HABperPixelsum += HABperPixel
            maxErrbatchsum += maxErrbatch
            assert not np.isnan(lossValue), 'Model diverged with loss = NaN'

            # Calculate pixel error for current batch
            lossValuePixel = np.sqrt(lossValue*(2/(modelParams['activeBatchSize']*4)))
            lossValueSum += lossValuePixel

            if step % FLAGS.printOutStep == 0:
                numExamplesPerStep = modelParams['activeBatchSize']
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch) pixel_err_avg = %.2f max_err_avg = %.2f')
                logging.info(format_str % (datetime.now(), step, HABperPixel,
                                           examplesPerSec, secPerBatch, HABperPixelsum/(step+1), maxErrbatchsum/(step+1)))

            if step % FLAGS.summaryWriteStep == 0:
                summaryStr = sess.run(summaryOp)
                summaryWriter.add_summary(summaryStr, step)
            
            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == modelParams['maxSteps']):
                print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins, max_err_avg = %.2f' %
                        ((100*step)/modelParams['maxSteps'], HABperPixelsum/(step+1), durationSum/60, 
                         (((durationSum*modelParams['maxSteps'])/(step+1))/60)-(durationSum/60), maxErrbatchsum/(step+1)))

        print('Average test pixel error = %.2f - Average max pixel error = %.2f - Average time per sample= %.2f s, Steps = %d' %
                        (HABperPixelsum/(step), maxErrbatchsum/(step), duration/(step*modelParams['activeBatchSize']), step))


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
    print(modelParams['modelName'])
    print('Rounds on datase = %.1f' % float((modelParams['testBatchSize']*modelParams['testMaxSteps'])/modelParams['numTestDatasetExamples']))
    #print('Train Input: %s' % modelParams['trainDataDir'])
    print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Input: %s' % modelParams['trainLogDir'])
    print('Test  Logs Output: %s' % modelParams['testLogDir'])
    #print('Train Warp Output: %s' % modelParams['warpedTrainDataDir'])
    print('Test  Warp Output: %s' % modelParams['warpedTestDataDir'])
    if input("(Overwrite WARNING) Did you change logs directory? ") != "yes":
        print("Please consider changing logs directory in order to avoid overwrite!")
        return
    if tf.gfile.Exists(modelParams['testLogDir']):
        tf.gfile.DeleteRecursively(modelParams['testLogDir'])
    tf.gfile.MakeDirs(modelParams['testLogDir'])
    test()


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
