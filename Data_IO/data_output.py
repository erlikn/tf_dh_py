# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
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


def _result_writer(imgPatchOrig, imgPatchPert, HAB, filePath):
    # Tensorflow record
    tfrecord_io.tfrecord_writer(imgPatchOrig, imgPatchPert, HAB, filePath)
    return

def _warp_without_orig_image(imageDuo, pHAB):
    # update the 
    # split for depth dimension
    orig, pert = np.asarray(np.split(imageDuo[0], 2, axis=2))
    orig = orig.reshape([orig.shape[0], orig.shape[1]])
    pert = pert.reshape([pert.shape[0], pert.shape[1]])
    
    # p & 0 is top left    - 1 is top right
    # 2     is bottom left - 3 is bottom right
    pRow = 0
    pCol = 0
    squareSize = orig.shape[0]

    HAB = np.asarray([[pHAB[0], pHAB[1], pHAB[2], pHAB[3]],
                      [pHAB[4], pHAB[5], pHAB[6], pHAB[7]]], np.float32)
    

    pOrig = np.array([[pRow, pRow, pRow+squareSize, pRow+squareSize],
                      [pCol, pCol+squareSize, pCol, pCol+squareSize]], np.float32)
    pPert = np.asarray(pOrig+HAB)
    # get transformation matrix and transform the image to new space
    Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
    pert = cv2.warpPerspective(orig, Hmatrix, (orig.shape[0], orig.shape[1]))
    return orig, pert

def _warp_with_orig_image(imageOrig, imageDuo, pOrig, pHAB):
    # update the Perturbed image given the original image

    # split for depth dimension
    orig, pert = np.asarray(np.split(imageDuo[0], 2, axis=2))
    orig = orig.reshape([orig.shape[0], orig.shape[1]])
    # p & 0 is top left    - 1 is top right
    # 2     is bottom left - 3 is bottom right
    pRow = pOrig[0]
    pCol = pOrig[5]
    squareSize = orig.shape[0]

    # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
    if kwargs.get('phase') == 'test':
        pHAB = 2*pHAB
        squareSize = 2*squareSize

    HAB = np.asarray([[pHAB[0], pHAB[1], pHAB[2], pHAB[3]],
                      [pHAB[4], pHAB[5], pHAB[6], pHAB[7]]], np.float32)
    
    pPert = np.asarray(pOrig+HAB)
    # get transformation matrix and transform the image to new space
    Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
    pert = cv2.warpPerspective(imageOrig, Hmatrix, (imageOrig.shape[0], imageOrig.shape[1]))
    # crop the image at original location
    pert = pert[pRow:pRow+squareSize, pCol:pCol+squareSize]
    
    # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
    if kwargs.get('phase') == 'test':
        pert = cv2.resize(pert, (128,128))
    
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
    for i in range(kwargs.get('activeBatchSize')):
        if kwargs.get('warpOriginalImage'):
            imageOrig, imagePert = _warp_with_orig_image(batchImageOrig[i], batchImage[i], batchPOrig[i], batchPHAB[i], **kwargs)
        else:
            imageOrig, imagePert = _warp_without_orig_image(batchImage[i], batchPHAB[i])
        
        tHAB = batchTHAB[i]-batchPHAB[i]
        _result_writer(imageOrig, imagePert, tHAB, kwargs.get('wrapedImageFolderName')+'/', batchTFrecFileIDs[i], batchTFrecFileIDs[i])

    return
