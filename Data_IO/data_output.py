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

def _warp(image, pHAB):
    HAB = np.asarray([[pHAB[0], pHAB[1], pHAB[2], pHAB[3]],
                      [pHAB[4], pHAB[5], pHAB[6], pHAB[7]]], np.float32)
    
    # get transformation matrix and transform the image to new space
    Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
    dst = cv2.warpPerspective(img, Hmatrix, imageSize)
    return

def output(batchImage, batchTHAB, batchPHAB, batchTFrecFilenames, **kwargs):
    """
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    ## for each value call the record writer function
    for i in range():
        batchImage = _warp(batchImage, batchPHAB)
        batchTHAB = batchTHAB-batchPHAB
        _result_writer(batchImage[i][0], batchImage[i][1], batchTHAB[i], kwargs.get('wrapedImageFolderName')+'/'+batchTFrecFilenames[i])

    return
