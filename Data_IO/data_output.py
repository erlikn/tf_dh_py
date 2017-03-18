# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
import os
import json
import collections
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

import Data_IO.tfrecord_io as tfrecord_io

# Global Variables
NUM_OF_TEST_PERTURBED_IMAGES = 5
NUM_OF_TRAIN_PERTURBED_IMAGES = 7

def _warp_wOut_orig_newTarget(imageDuo, pHAB):
    """
    FIX BEFORE USE
    """
    # update the 
    # split for depth dimension
    orig, pert = np.asarray(np.split(imageDuo, 2, axis=2))
    orig = orig.reshape([orig.shape[0], orig.shape[1]])
    pert = pert.reshape([pert.shape[0], pert.shape[1]])
    
    # p & 0 is top left    - 1 is top right
    # 2     is bottom left - 3 is bottom right
    pRow = 0
    pCol = 0
    squareSize = orig.shape[0]

    pOrig = np.array([[pRow, pRow, pRow+squareSize, pRow+squareSize],
                      [pCol, pCol+squareSize, pCol, pCol+squareSize]], np.float32)
    pPert = np.asarray(pOrig+pHAB)
    # get transformation matrix and transform the image to new space
    Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
    pert = cv2.warpPerspective(orig, Hmatrix, (orig.shape[0], orig.shape[1]))
    return orig, pert

def _warp_w_orig_newTarget(imageOrig, imageDuo, pOrig, cHAB, **kwargs):
    # update the Perturbed image given the original image

    # split for depth dimension
    orig, pert = np.asarray(np.split(imageDuo, 2, axis=2))
    orig = orig.reshape([orig.shape[0], orig.shape[1]])

    # Get the correct box information
    # p & 0 is top left    - 1 is top right
    # 2     is bottom left - 3 is bottom right
    pRow = int(pOrig[0][0])
    pCol = int(pOrig[1][0])
    squareSize = orig.shape[0]

    # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
    if kwargs.get('phase') == 'test':
        cHAB = 2*cHAB
        squareSize = 2*squareSize

    pPert = np.asarray(pOrig+cHAB)
    # get transformation matrix and transform the image to new space
    Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
    pert = cv2.warpPerspective(imageOrig, Hmatrix, (imageOrig.shape[1], imageOrig.shape[0]))

    # crop the image at original location
    pert = pert[pRow:pRow+squareSize, pCol:pCol+squareSize]

    # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
    if kwargs.get('phase') == 'test':
        pert = cv2.resize(pert, (orig.shape[1], orig.shape[0]))

    return orig, pert

def output(batchImageOrig, batchImage, batchPOrig, batchTHAB, batchPHAB, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER 

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    ## for each value call the record writer function
    corrupt_patchOrig=0
    corrupt_patchPert=0
    corrupt_imageOrig=0
    for i in range(kwargs.get('activeBatchSize')):
        # Get the difference of tHAB and pHAB, and make new perturbed image based on that
        cHAB = batchTHAB[i]-batchPHAB[i]
        # put them in correct form
        HAB = np.asarray([[cHAB[0], cHAB[1], cHAB[2], cHAB[3]],
                          [cHAB[4], cHAB[5], cHAB[6], cHAB[7]]], np.float32)
        pOrig = np.asarray([[batchPOrig[i][0], batchPOrig[i][1], batchPOrig[i][2], batchPOrig[i][3]],
                            [batchPOrig[i][4], batchPOrig[i][5], batchPOrig[i][6], batchPOrig[i][7]]])
        if kwargs.get('warpOriginalImage'):
            patchOrig, patchPert = _warp_w_orig_newTarget(batchImageOrig[i], batchImage[i], pOrig, HAB, **kwargs)
            # NOT DEVELOPED YET
            #imageOrig, imagePert = _warp_w_orig_newOrig(batchImageOrig[i], batchImage[i], pOrig, batchPHAB[i], **kwargs)
        else:
            patchOrig, patchPert = _warp_wOut_orig_newTarget(batchImage[i], batchPHAB[i])

        # Write each Tensorflow record
        fileIDs = str(batchTFrecFileIDs[i][0]) + '_' + str(batchTFrecFileIDs[i][1])
        tfrecord_io.tfrecord_writer(batchImageOrig[i], patchOrig, patchPert, pOrig, HAB,
                                    kwargs.get('warpedOutputFolder')+'/',
                                    fileIDs, batchTFrecFileIDs[i])
        if batchImageOrig[i].shape[0] != 240:
            corrupt_imageOrig+=1
        if patchOrig.shape[0]!=128:
            corrupt_patchOrig+=1
        if patchPert.shape[0]!=128:
            corrupt_patchPert+=1

    return corrupt_imageOrig, corrupt_patchOrig, corrupt_patchPert

def write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def output_with_test_image_files(batchImageOrig, batchImage, batchPOrig, batchTHAB, batchPHAB, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER 

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    imagesOutputFolder = kwargs.get('testLogDir')+'/images/'
    _set_folders(imagesOutputFolder)
    ## for each value call the record writer function
    corrupt_patchOrig = 0
    corrupt_patchPert = 0
    corrupt_imageOrig = 0
    dataJson = {'pOrig':[], 'tHAB':[], 'pHAB':[]}
    for i in range(kwargs.get('activeBatchSize')):
        dataJson['pOrig'] = batchPOrig[i].tolist()
        dataJson['tHAB'] = batchTHAB[i].tolist()
        dataJson['pHAB'] = batchPHAB[i].tolist()
        write_json_file(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1]), dataJson)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+ str(batchTFrecFileIDs[i][1])+'_fullOrig.jpg', batchImageOrig[i]*255)

        orig, pert = np.asarray(np.split(batchImage[i], 2, axis=2))
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_ob.jpg', orig*255)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_pb.jpg', pert*255)

        # Get the difference of tHAB and pHAB, and make new perturbed image based on that
        cHAB = batchTHAB[i]-batchPHAB[i]
        # put them in correct form
        HAB = np.asarray([[cHAB[0], cHAB[1], cHAB[2], cHAB[3]],
                          [cHAB[4], cHAB[5], cHAB[6], cHAB[7]]], np.float32)
        pOrig = np.asarray([[batchPOrig[i][0], batchPOrig[i][1], batchPOrig[i][2], batchPOrig[i][3]],
                            [batchPOrig[i][4], batchPOrig[i][5], batchPOrig[i][6], batchPOrig[i][7]]])
        if kwargs.get('warpOriginalImage'):
            patchOrig, patchPert = _warp_w_orig_newTarget(batchImageOrig[i], batchImage[i], pOrig, HAB, **kwargs)
            # NOT DEVELOPED YET
            #imageOrig, imagePert = _warp_w_orig_newOrig(batchImageOrig[i], batchImage[i], pOrig, batchPHAB[i], **kwargs)
        else:
            patchOrig, patchPert = _warp_wOut_orig_newTarget(batchImage[i], batchPHAB[i])

        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_op.jpg', patchOrig*255)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_pp.jpg', patchPert*255)
        # Write each Tensorflow record
        fileIDs = str(batchTFrecFileIDs[i][0]) + '_' + str(batchTFrecFileIDs[i][1])
        tfrecord_io.tfrecord_writer(batchImageOrig[i], patchOrig, patchPert, pOrig, HAB,
                                    kwargs.get('warpedOutputFolder')+'/',
                                    fileIDs, batchTFrecFileIDs[i])
        if batchImageOrig[i].shape[0] != 240:
            corrupt_imageOrig+=1
        if patchOrig.shape[0]!=128:
            corrupt_patchOrig+=1
        if patchPert.shape[0]!=128:
            corrupt_patchPert+=1

    return corrupt_imageOrig, corrupt_patchOrig, corrupt_patchPert