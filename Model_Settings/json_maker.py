import json
import collections
import numpy as np

def write_json_file(filename, datafile):
        datafile = collections.OrderedDict(sorted(datafile.items()))
        with open(filename, 'w') as outFile:
                json.dump(datafile, outFile, indent = 0)


# Twin Common Parameters
trainLogDirBase = '../Data/128_logs/tfdh_twin_py_logs/train_logs/'
testLogDirBase = '../Data/128_logs/tfdh_twin_py_logs/test_logs/'

# Shared Descriptions
modelName_desc = "Name of the model file to be loaded from Model_Factory"

usefp16_desc = "Use 16 bit floating point precision"
pretrainedModelCheckpointPath_desc = "If specified, restore this pretrained model before beginning any training"
trainDataDir_desc = "Directory to read training samples"
testDataDir_desc = "Directory to read test samples"
numTrainDatasetExamples_desc = "Number of images to process in train dataset"
numTestDatasetExamples_desc = "Number of images to process in test dataset"
trainLogDir_desc = "Directory where to write train event logs and checkpoints"
testLogDir_desc = "Directory where to write test event logs and checkpoints"

imageSize_desc = "Image square size"
imageChannels_desc = "Image channels"
outputSize_desc = "Final output size"
modelShape_desc = "Network model with 8 convolutional layers with 2 fully connected layers"
numParallelModules_desc = "Number of parallel modules of the network"
trainBatchSize_desc = "Batch size of input data for train"
testBatchSize_desc = "Batch size of input data for test"
batchNorm_desc = "Should we use batch normalization"
weightNorm_desc = "Should we use weight normalization"
optimizer_desc = "Type of optimizer to be used [AdamOptimizer, MomentumOptimizer, GradientDescentOptimizer]"
initialLearningRate_desc = "Initial learning rate."
learningRateDecayFactor_desc = "Learning rate decay factor"
numEpochsPerDecay_desc = "Epochs after which learning rate decays"
momentum_desc = "Momentum Optimizer: momentum"
epsilon_desc = "epsilon value used in AdamOptimizer"
dropOutKeepRate_desc = "Keep rate for drop out"
clipNorm_desc = "Gradient global normalization clip value"
trainMaxSteps_desc = "Number of batches to run"
testMaxSteps_desc = "Number of batches to run during test. numTestDatasetExamples = testMaxSteps x testBatchSize" 
lossFunction_desc = "Indicates type of the loss function to be used [L2, CrossEntropy, ..]"

initExistingWeights_desc = "Should we load existing weights for training? Leave blank if not"
tuneExistingWeights_desc = "Should any reloaded weights be tuned?"
fineTune_desc = "If set, randomly initialize the final layer of weights in order to train the network on a new task"
logDevicePlacement_desc = "Whether to log device placement"

# Network Parameters
modelName = 'twin_cnn_4p4l2f'

usefp16 = False
pretrainedModelCheckpointPath = ''
trainDataDir = '../Data/128_train_tfrecords'
testDataDir = '../Data/128_test_tfrecords'
numTrainDatasetExamples = 500000
numTestDatasetExamples = 25000
trainLogDir = trainLogDirBase+'170127_TWN_MOM_W'
testLogDir = testLogDirBase+'170127_TWN_MOM_W'

imageSize = 128
imageChannels = 2
outputSize = 8
modelShape = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
numParallelModules = 2
trainBatchSize = 20
testBatchSize = 20
batchNorm = False
weightNorm = True
optimizer = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
initialLearningRate = 0.005
learningRateDecayFactor = 0.1
numEpochsPerDecay = 30000.0
momentum = 0.9
epsilon = 0.1
dropOutKeepRate = 0.5
clipNorm = 1.0
trainMaxSteps = 90000
testMaxSteps = int(np.ceil(numTestDatasetExamples/testBatchSize))
lossFunction = 'L2'

initExistingWeights = ''
tuneExistingWeights = True
fineTune = False
logDevicePlacement = False

data = {'modelName' : modelName,

        'usefp16' : usefp16,
        'pretrainedModelCheckpointPath' : pretrainedModelCheckpointPath,
        'trainDataDir' : trainDataDir,
        'testDataDir' : testDataDir,
        'numTrainDatasetExamples' : numTrainDatasetExamples,
        'numTestDatasetExamples' : numTestDatasetExamples,
        'trainLogDir' : trainLogDir,
        'testLogDir' : testLogDir,

        'imageSize' : imageSize,
        'imageChannels' : imageChannels,
        'outputSize' : outputSize,
        'modelShape' : modelShape,
        'numParallelModules' : numParallelModules,
        'trainBatchSize' : trainBatchSize,
        'testBatchSize' : testBatchSize,
        'batchNorm' : batchNorm,
        'weightNorm' : weightNorm,
        'optimizer' : optimizer,
        'initialLearningRate' : initialLearningRate,
        'learningRateDecayFactor' : learningRateDecayFactor,
        'numEpochsPerDecay' : numEpochsPerDecay,
        'momentum' : momentum,
        'epsilon' : epsilon,
        'dropOutKeepRate' : dropOutKeepRate,
        'clipNorm' : clipNorm,
        'trainMaxSteps' : trainMaxSteps,
        'testMaxSteps' : testMaxSteps,
        'lossFunction' : lossFunction,

        'initExistingWeights' : initExistingWeights,
        'tuneExistingWeights' : tuneExistingWeights,
        'fineTune' : fineTune,
        'logDevicePlacement' : logDevicePlacement,


        'modelName_desc' : modelName_desc,

        'usefp16_desc' : usefp16_desc,
        'pretrainedModelCheckpointPath_desc' : pretrainedModelCheckpointPath_desc,
        'trainDataDir_desc' : trainDataDir_desc,
        'testDataDir_desc' : testDataDir_desc,
        'numTrainDatasetExamples_desc' : numTrainDatasetExamples_desc,
        'numTestDatasetExamples_desc' : numTestDatasetExamples_desc,
        'trainLogDir_desc' : trainLogDir_desc,
        'testLogDir_desc' : testLogDir_desc,
        
        'imageSize_desc' : imageSize_desc,
        'imageChannels_desc' : imageChannels_desc,
        'outputSize_desc' : outputSize_desc,
        'modelShape_desc' : modelShape_desc,
        'numParallelModules_desc' : numParallelModules_desc,
        'trainBatchSize_desc' : trainBatchSize_desc,
        'testBatchSize_desc' : testBatchSize_desc,
        'batchNorm_desc' : batchNorm_desc,
        'weightNorm_desc' : weightNorm_desc,
        'optimizer_desc' : optimizer_desc,
        'initialLearningRate_desc' : initialLearningRate_desc,
        'learningRateDecayFactor_desc' : learningRateDecayFactor_desc,
        'numEpochsPerDecay_desc' : numEpochsPerDecay_desc,
        'momentum_desc' : momentum_desc,
        'epsilon_desc' : epsilon_desc,
        'dropOutKeepRate_desc' : dropOutKeepRate_desc,
        'clipNorm_desc' : clipNorm_desc,
        'trainMaxSteps_desc' : trainMaxSteps_desc,
        'testMaxSteps_desc' : testMaxSteps_desc,
        'lossFunction_desc' : lossFunction_desc,

        'initExistingWeights_desc' : initExistingWeights_desc,
        'tuneExistingWeights_desc' : tuneExistingWeights_desc,
        'fineTune_desc' : fineTune_desc,
        'logDevicePlacement_desc' : logDevicePlacement_desc
        }

##############
reCompileJSON=True

############## TWIN
if reCompileJSON:
    write_json_file('170127_TWN_MOM_W.json', data)

if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170127_TWN_MOM_B'
    data['testLogDir'] = testLogDirBase+'170127_TWN_MOM_B'
    data['trainMaxSteps'] = 120000
    data['numEpochsPerDecay'] = 40000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170127_TWN_MOM_B.json', data)

if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170128_TWN_MOM_B'
    data['testLogDir'] = testLogDirBase+'170128_TWN_MOM_B'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 2048]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170128_TWN_MOM_B.json', data)
    
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170127_TWN_MOM_BW'
    data['testLogDir'] = testLogDirBase+'170127_TWN_MOM_BW'
    data['trainMaxSteps'] = 120000
    data['numEpochsPerDecay'] = 40000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = True
    write_json_file('170127_TWN_MOM_BW.json', data)
##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170118_AdamOpt_B16_256'
    data['testLogDir'] = testLogDirBase+'170118_AdamOpt_B16_256'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 16
    data['testBatchSize'] = 16
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 256, 256, 1024]
    data['batchNorm'] = False
    data['weightNorm'] = True
    write_json_file('170118_AdamOpt_B16_256.json', data)
###############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170118_MomentumOpt_B20_256'
    data['testLogDir'] = testLogDirBase+'170118_MomentumOpt_B20_256'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 256, 256, 1024]
    data['batchNorm'] = False
    data['weightNorm'] = True
    write_json_file('170118_MomentumOpt_B20_256.json', data)
##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170120_MomentumOpt_256_256'
    data['testLogDir'] = testLogDirBase+'170120_MomentumOpt_256_256'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
    data['batchNorm'] = False
    data['weightNorm'] = True
    write_json_file('170120_MomentumOpt_256_256.json', data)

if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170120_MomentumOpt_256_256_150k'
    data['testLogDir'] = testLogDirBase+'170120_MomentumOpt_256_256_150k'
    data['trainMaxSteps'] = 150000
    data['numEpochsPerDecay'] = 50000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
    data['batchNorm'] = False
    data['weightNorm'] = True
    write_json_file('170120_MomentumOpt_256_256_150k.json', data)
    
############## 
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170125_MomentumOpt_256_256_BNorm'
    data['testLogDir'] = testLogDirBase+'170125_MomentumOpt_256_256_BNorm'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170125_MomentumOpt_256_256_BNorm.json', data)
##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170125_MomentumOpt_256_256_WBNorm'
    data['testLogDir'] = testLogDirBase+'170125_MomentumOpt_256_256_WBNorm'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 10
    data['testBatchSize'] = 10
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = True
    write_json_file('170125_MomentumOpt_256_256_WBNorm.json', data)

##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170129_TWN_MOM_B_64'
    data['testLogDir'] = testLogDirBase+'170129_TWN_MOM_B_64'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 64, 64, 64, 64, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170129_TWN_MOM_B_64.json', data)


##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170129_TWN_MOM_B_32'
    data['testLogDir'] = testLogDirBase+'170129_TWN_MOM_B_32'
    data['trainMaxSteps'] = 30000#90000
    data['numEpochsPerDecay'] = 10000.0#30000.0
    data['trainBatchSize'] = 60#20
    data['testBatchSize'] = 60#20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [32, 32, 32, 32, 32, 32, 32, 32, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170129_TWN_MOM_B_32.json', data)
    
##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170130_TWN_MOM_B_16'
    data['testLogDir'] = testLogDirBase+'170130_TWN_MOM_B_16'
    data['trainMaxSteps'] = 15000#90000
    data['numEpochsPerDecay'] = 5000.0#30000.0
    data['trainBatchSize'] = 120#20
    data['testBatchSize'] = 120#20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [16, 16, 16, 16, 16, 16, 16, 16, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170130_TWN_MOM_B_16.json', data)
    

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

# Single Common Parameters
trainLogDirBase = '../Data/128_logs/tfdh_py_logs/train_logs/'
testLogDirBase = '../Data/128_logs/tfdh_py_logs/test_logs/'

data['modelName'] = 'cnn_8l2f'

##############
if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170126_SIN_B'
    data['testLogDir'] = testLogDirBase+'170126_SIN_B'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = False
    write_json_file('170126_SIN_B.json', data)

if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170126_SIN_W'
    data['testLogDir'] = testLogDirBase+'170126_SIN_W'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
    data['batchNorm'] = False
    data['weightNorm'] = True
    write_json_file('170126_SIN_W.json', data)

if reCompileJSON:
    data['trainLogDir'] = trainLogDirBase+'170126_SIN_BW'
    data['testLogDir'] = testLogDirBase+'170126_SIN_BW'
    data['trainMaxSteps'] = 90000
    data['numEpochsPerDecay'] = 30000.0
    data['trainBatchSize'] = 20
    data['testBatchSize'] = 20
    data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
    data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
    data['batchNorm'] = True
    data['weightNorm'] = True
    write_json_file('170126_SIN_BW.json', data)
##############