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

import tfrecord_io

# Global Variables
NUM_OF_TEST_PERTURBED_IMAGES = 5
NUM_OF_TRAIN_PERTURBED_IMAGES = 7



def image_process_subMean_divStd(img):
    out = img - np.mean(img)
    out = out / img.std()
    return out

def image_process_subMean_divStd_n1p1(img):
    out = img - np.mean(img)
    out = out / img.std()
    out = (2*((out-out.min())/(out.max()-out.min())))-1
    return out

def perturb_writer( ID, idx,
                    imgOrig, imgPatchOrig, imgPatchPert, HAB, pOrig,
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
    filename = str(ID) + "_" + str(idx)
    fileID = [ID, idx]
    tfrecord_io.tfrecord_writer(imgOrig, imgPatchOrig, imgPatchPert, pOrig, HAB, tfRecFolder, filename, fileID)

    #imgOp = image_process_subMean_divStd(imgPatchOrig)
    #imgPp = image_process_subMean_divStd(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolder+filename)
    # Tensorflow record in range -1 and 1
    #filename = filenameWrite.replace(".jpg", "_"+ str(idx))
    #imgOp = image_process_subMean_divStd_n1p1(imgPatchOrig)
    #imgPp = image_process_subMean_divStd_n1p1(imgPatchPert)
    #tfrecord_writer(imgOp, imgPp, HAB, pOrig, tfRecFolderN1P1+filename)

    return

def generate_random_perturbations(datasetType, img, ID, num, tfRecFolder):
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
        rndListRowPert = np.asarray(random.sample(range(-32, 32), 4))
        rndListColPert = np.asarray(random.sample(range(-32, 32), 4))
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
        perturb_writer(ID, i,
                       np.asarray(img), imgTempOrig, imgTempPert, H_AB, pOrig,
                       tfRecFolder)
        #mu = np.average(np.linalg.norm(H_AB, axis=0))
        #var = np.var(np.linalg.norm(H_AB, axis=0))
        mu = np.average(H_AB, axis=1)
        var = np.sqrt(np.var(H_AB, axis=1))
    return mu, var

def prepare_dataset(datasetType, readFolder, tfRecFolder):
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
    filenames.sort()
    #
    id = 0
    totalCount = 0
    tMu = np.asarray([0.0, 0.0])
    tVar = np.asarray([0.0, 0.0])
    for filename in filenames:
        if "train" in datasetType:
            if id < 33302: # total of 500000
                num = NUM_OF_TRAIN_PERTURBED_IMAGES + 1
            else:
                num = NUM_OF_TRAIN_PERTURBED_IMAGES
        else: # test
            num = NUM_OF_TEST_PERTURBED_IMAGES
        img = cv2.imread(readFolder+filename, 0)
        if img.ndim == 2:
            mu, var = generate_random_perturbations(datasetType, img, id, num, tfRecFolder)
            tMu = tMu + mu
            tVar = tVar + var
            totalCount = totalCount + (num)
        else:
            print("Not a grayscale")
        if math.floor((id*50)/len(filenames)) != math.floor(((id-1)*50)/len(filenames)):
            print(str(math.floor((id*100)/len(filenames)))+'%  '+str(id))
            print('Perturbation Statistics: MuXY = %.1f, %.1f , VarXY = %.1f, %.1f , Files = %d' % (mu[0], mu[1], var[0], var[1], totalCount))
        id = id+1
    i=id
    print('100%  Done')
    print('Perturbation Statistics: Average MuXY = %.1f, %.1f , VarXY = %.1f, %.1f , Files = %d' % (tMu[0]/i, tMu[1]/i, tVar[0]/i, tVar[1]/i, totalCount))
    print('Perturbation Statistics: Average MuXY = %.1f       , VarXY = %.1f ' % (np.linalg.norm(tMu/i), np.linalg.norm(tVar/i)))


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



dataRead = "../../Data/MSCOCO_orig/"
dataReadGray = "../../Data/MSCOCO_gray/"

train320 = "../../Data/320_240_train/"
traintfRecordFLD = "../../Data/128_train_tfrecords/"


test640 = "../../Data/640_480_test/"
testtfRecordFLD = "../../Data/128_test_tfrecords/"

""" Divide dataset (87XXXX) to (5000) test and (82XXX) training samples"""
#divide_train_test(dataReadGray, train320, test640)

"""
    Generate more Test Samples
    generate 5,000x5=25,000 Samples
    Total Files = 25,000 orig + 25,000 pert + 25,000 origSq 25,000 HAB = 100,000 
"""
#prepare_dataset("test", test640, testtfRecordFLD)
"""
    Generate more Train Samples
    generate  Samples
    Total Files =  orig +  pert + 25,000 HAB = 
"""
prepare_dataset("train", train320, traintfRecordFLD)
