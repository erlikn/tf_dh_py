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

# Global Variables
NUM_OF_TEST_PERTURBED_IMAGES = 5
NUM_OF_TRAIN_PERTURBED_IMAGES = 7



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# to get HAB and pOrig
def _float_nparray(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tfrecord_writer(imgPatchOrig, imgPatchPert, HAB, pOrig, tfRecordFolder, tfFileName):
    """Converts a dataset to tfrecords."""
    #images = data_set.images
    #labels = data_set.labels
    #num_examples = data_set.num_examples
    tfRecordPath = tfRecordFolder + tfFileName;
    filename = np.asarray(tfFileName).tostring()

    rows = imgPatchOrig.shape[0]
    cols = imgPatchOrig.shape[1]
    depth = 2
    #OLD
    stackedImage = np.stack((imgPatchOrig, imgPatchPert), axis=2) #3D array (hieght, width, channels)
    flatImage = stackedImage.reshape(rows*cols*depth)
    #Stack Images
    #stackedImage = np.array([imgPatchOrig.reshape(imgPatchOrig.shape[0]*imgPatchOrig.shape[1]),
    #                imgPatchPert.reshape(imgPatchPert.shape[0]*imgPatchPert.shape[1])])
    #flatImage = stackedImage.reshape(stackedImage.shape[0]*stackedImage.shape[1])
    
    flatImageList = flatImage.tostring()

    pOrigRow = np.array([pOrig[0][0], pOrig[0][1], pOrig[0][2], pOrig[0][3],
                         pOrig[1][0], pOrig[1][1], pOrig[1][2], pOrig[1][3]], np.float32)
    pOrigList = pOrigRow.tolist()
    HABRow = np.asarray([HAB[0][0], HAB[0][1], HAB[0][2], HAB[0][3],
                         HAB[1][0], HAB[1][1], HAB[1][2], HAB[1][3]], np.float32)
    HABList = HABRow.tolist()
    
    #print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(filename),
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'pOrig': _float_nparray(pOrigList),
        'HAB': _float_nparray(HABList), # 2D np array
        'image': _bytes_feature(flatImageList)
        }))
    writer.write(example.SerializeToString())
    writer.close()

def image_process_subMean_divStd(img):
    out = img - np.mean(img)
    out = out / img.std()
    return out

def image_process_subMean_divStd_n1p1(img):
    out = img - np.mean(img)
    out = out / img.std()
    out = (2*((out-out.min())/(out.max()-out.min())))-1
    return out

def perturb_writer( filenameWrite, idx,
                    imgPatchOrig, imgPatchPert, HAB, pOrig,
                    tfRecFolder):
    ##### original patch
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_orig.jpg")
    #cv2.imwrite(patchFolder+filename, imgPatchOrig)
    ##### perturbed patch
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_pert.jpg")
    #cv2.imwrite(patchFolder+filename, imgPatchPert)
    ##### HAB
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +"_HAB.csv")
    #with open(HABFolder+filename, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    if (HAB.ndim > 1):
    #        writer.writerows(HAB)
    #    else:
    #        writer.writerow(HAB)
    ##### Original square
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +".csv")
    #with open(squareFolder+filename, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    if (pOrig.ndim > 1):
    #        writer.writerows(pOrig)
    #    else:
    #        writer.writerow(pOrig)
    # Tensorflow record
    filename = filenameWrite.replace(".jpg", "_"+ str(idx) +".tfrecords")
    tfrecord_writer(imgPatchOrig, imgPatchPert, HAB, pOrig, tfRecFolder, filename)
    
    #imgOp = image_process_subMean_divStd(imgPatchOrig)
    #imgPp = image_process_subMean_divStd(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolder+filename)
    # Tensorflow record in range -1 and 1
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx) +".tfrecords")
    #imgOp = image_process_subMean_divStd_n1p1(imgPatchOrig)
    #imgPp = image_process_subMean_divStd_n1p1(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolderN1P1+filename)

    return

def generate_random_perturbations(datasetType, img, filenameWrite, num, tfRecFolder):
    if "train" in datasetType:
        # if 320x240 => 128x128 w thrPerturbation=32
        squareSize=128
        thrPerturbation=32
        imageSize = (320,240)
    if "test" in datasetType:
        # if 640x480 => 256x256 w thrPerturbation=64
        squareSize=256
        thrPerturbation=64
        imageSize = (640,480)
    rndListRowOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    rndListColOrig = random.sample(range(thrPerturbation,img.shape[0]-thrPerturbation-squareSize), num)
    for i in range(0, len(rndListRowOrig)):
        pRow = rndListRowOrig[i]
        pCol = rndListColOrig[i]
        imgTempOrig = img[pRow:pRow+squareSize, pCol:pCol+squareSize]
        # p & 0 is top left    - 1 is top right
        # 2     is bottom left - 3 is bottom right
        pOrig = np.array([[pRow, pRow, pRow+squareSize, pRow+squareSize], 
                          [pCol, pCol+squareSize, pCol, pCol+squareSize]], np.float32)
        # generate random perturbations (H^AB)
        rndListRowPert = np.asarray(random.sample(range(-32,32), 4))
        rndListColPert = np.asarray(random.sample(range(-32,32), 4))   
        H_AB = np.asarray([rndListRowPert, rndListColPert], np.float32)
        # 
        pPert = np.asarray(pOrig+H_AB)
        # get transformation matrix and transform the image to new space
        Hmatrix = cv2.getPerspectiveTransform(np.transpose(pOrig), np.transpose(pPert))
        dst = cv2.warpPerspective(img, Hmatrix, imageSize)
        # crop the image at original location
        imgTempPert = dst[pRow:pRow+squareSize, pCol:pCol+squareSize]
        if dst.max() > 256:
            print("NORMALIZATION to uint8 NEEDED!!!!!!!!!!")
        # Write down outputs
        # for TEST samples divide H_AB by 2 (64->32) and reduce divide image size by 4 (256x256->128x128)
        if "test" in datasetType:
            imgTempOrig = cv2.resize(imgTempOrig, (128,128))
            imgTempPert = cv2.resize(imgTempPert, (128,128))
            H_AB = H_AB/2
        perturb_writer(filenameWrite, i,
                       imgTempOrig, imgTempPert, H_AB, pOrig,
                       tfRecFolder)

    return

def prepare_dataset(datasetType, readFolder, tfRecFolder):
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
    filenames.sort()
    #
    i = 0
    for filename in filenames:
        if "train" in datasetType:
            if i < 33302: # total of 500000
                num = NUM_OF_TRAIN_PERTURBED_IMAGES + 1
            else:
                num = NUM_OF_TRAIN_PERTURBED_IMAGES
        else: # test
            num = NUM_OF_TEST_PERTURBED_IMAGES
        img = cv2.imread(readFolder+filename, 0)
        if img.ndim == 2:
            generate_random_perturbations(datasetType, img, filename, num, tfRecFolder)
        else:
            print("Not a grayscale")
        if math.floor((i*100)/len(filenames)) != math.floor(((i-1)*100)/len(filenames)):
            print(str(math.floor((i*100)/len(filenames)))+'%  '+str(i))
        i = i+1
    print('100%  Done')


def divide_train_test(readFolder, trainFolder, testFolder):
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]

    # 5000 test subjects
    testSelector = random.sample(range(0, len(filenames)), 5000)
        
    i = 0
    trainCounter = 0
    testCounter = 0
    filenames.sort()
    for files in filenames:
        img = cv2.imread(readFolder+files, 0)
        if i in testSelector:
            # test image
            testCounter = testCounter+1
            img = cv2.resize(img, (640, 480))
            cv2.imwrite(testFolder+files, img)
        else:
            # train image
            trainCounter = trainCounter+1
            img = cv2.resize(img, (320, 240))
            cv2.imwrite(trainFolder+files, img)

        if math.floor((i*10)/len(filenames)) != math.floor(((i-1)*10)/len(filenames)):
            print(str(math.floor((i*100)/len(filenames)))+'%  '+str(i))
            print(str(testCounter)+' out of 5000')
            print(str(trainCounter)+' out of '+str(len(filenames)-5000))
        i = i+1



dataRead = "../Data/MSCOCO_orig/"
dataReadGray = "../Data/MSCOCO_gray/"

train320 = "../Data/320_240_train/"
traintfRecordFLD = "../Data/128_train_tfrecords/"


test640 = "../Data/640_480_test/"
testtfRecordFLD = "../Data/128_test_tfrecords/"

""" Divide dataset (87XXXX) to (5000) test and (82XXX) training samples"""
#divide_train_test(dataReadGray, train320, test640)

"""
    Generate more Test Samples
    generate 5,000x5=25,000 Samples
    Total Files = 25,000 orig + 25,000 pert + 25,000 origSq 25,000 HAB = 100,000 
"""
prepare_dataset("test", test640, testtfRecordFLD)
"""
    Generate more Train Samples
    generate  Samples
    Total Files =  orig +  pert + 25,000 HAB = 
"""
#prepare_dataset("train", train320, traintfRecordFLD)
