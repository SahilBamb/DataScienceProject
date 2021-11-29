#!pip install h5py==2.9.0

import cv2
import numpy as np
import scipy.io as sc
import os
import h5py
import pickle
import matplotlib.pyplot as plt

def prepDataforCNN(numChannel = 1, feat_norm = False):

    htr  = h5py.File(r'C:\Users\Jo Wood\Documents\GitHub\SVHN_CNN\CNN\train.h5', 'r')
    hts  = h5py.File(r'C:\Users\Jo Wood\Documents\GitHub\SVHN_CNN\CNN\test.h5' , 'r')

    if numChannel == 1:
       digits     = htr["digitsBW"]
       testdigits = hts["digitsBW"]
       negdigits  = htr["negdigitsBW"]
    else:
       digits     = htr["digits"]
       testdigits = hts["digits"]
       negdigits  = htr["negdigits"]

    trainlabs  = htr["labs5"]
    testlabs   = hts["labs5"]
    neglabs    = htr["neglab"]

    digits     = digits[:]
    testdigits = testdigits[:]
    negdigits  = negdigits[:]
    trainlabs  = trainlabs[:]
    testlabs   = testlabs[:]
    neglabs    = neglabs[:]
    negdigits  = negdigits[:]

    seed = 25
    np.random.seed(seed)
    countNeg = 30000
    countX   = 90000

    negIdx = np.random.randint(0,negdigits.shape[0],countNeg)
    numNegTs = np.arange(negdigits.shape[0] - 500,negdigits.shape[0],1)
    numtran = np.arange(0,countX,1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K


    ntrdigits = negdigits[negIdx]
    ntrlab    = neglabs[negIdx,:]
    ntest     = negdigits[numNegTs,:]
    ntslab    = neglabs[numNegTs,:]

    preTrain = np.vstack((digits,ntrdigits)).astype('float32')  #
    preTest =  np.vstack((testdigits,ntest)).astype('float32')  #xtest

    trainlabs = np.vstack((trainlabs,ntrlab)).astype('uint8')  #xtrlab
    testlabs  = np.vstack((testlabs,ntslab)).astype('uint8')   #xtslab
    ind = np.argwhere(trainlabs[:,0]<5)
    ind = ind[:,0]
    preTrain = preTrain[ind,:]
    trainlabs = trainlabs[ind,:]

    nb = np.reshape(np.asarray(trainlabs[:,0] > 0, dtype = 'uint8'),(trainlabs.shape[0],1))
    trl = np.hstack((trainlabs, nb))

    ind = np.argwhere(testlabs[:,0]<5)
    ind = ind[:,0]
    preTest = preTest[ind,:]
    testlabs = testlabs[ind,:]

    nb = np.reshape(np.asarray(testlabs[:,0] > 0, dtype = 'uint8'),(testlabs.shape[0],1))
    tsl = np.hstack((testlabs, nb))

    train = np.float64(preTrain)
    test  = np.float64(preTest)

    for i in range(preTrain.shape[0]):
        if numChannel >1:
            for channel in range(0,numChannel,1):
                train[i][:,:,channel] -= np.mean(preTrain[i][:,:,channel].flatten(),axis = 0)
        else:
            train[i] -= np.mean(preTrain[i].flatten(),axis = 0)

    for i in range(preTest.shape[0]):
        if numChannel > 1:
           for channel in range(0,numChannel,1):
               test[i][:,:,channel] -= np.mean(preTest[i][:,:,channel] .flatten(),axis = 0)
        else:
           test[i] -= np.mean(preTest[i].flatten(),axis = 0)


    if feat_norm:
       M = np.mean(train, axis = 0)
       train = train - M
       sd = np.std(train, axis = 0)
       train = train/sd

       test = test - M
       test = test/sd
       featNorm = {'mean': M, 'std': sd}
       if numChannel > 1:
           with open('BGRnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol = pickle.HIGHEST_PROTOCOL)
       else:
           with open('BWnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol = pickle.HIGHEST_PROTOCOL)

    numtrain = train.shape[0]
    numtest  = test.shape[0]
    row = train.shape[1]
    col = train.shape[2]

    train = np.reshape(train,(numtrain,row,col,numChannel))
    test  = np.reshape(test, (numtest, row,col,numChannel))

    p = 0.9
    seed = 25
    np.random.seed(seed)
    split = np.int32(np.round((p * numtrain)))  #.85

    idx = np.random.permutation(numtrain)
    trIdx = idx[0:split]
    vlIdx = idx[split:numtrain]

    trlab = [ np.reshape(trl[:,0],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,1],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,2],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,3],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,4],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,6],(numtrain,1)).astype('uint8')]

    tslab = [ np.reshape(tsl[:,0],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,1],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,2],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,3],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,4],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,6],(numtest,1)).astype('uint8')]

    ctrlab = [trlab[0][trIdx], trlab[1][trIdx], trlab[2][trIdx], trlab[3][trIdx], trlab[4][trIdx], trlab[5][trIdx]]
    cvlab  = [trlab[0][vlIdx], trlab[1][vlIdx], trlab[2][vlIdx], trlab[3][vlIdx], trlab[4][vlIdx], trlab[5][vlIdx]]
    ctslab = [tslab[0],        tslab[1],        tslab[2],        tslab[3],        tslab[4],        tslab[5]]

    data = {'trainX': train[trIdx], 'trainY': ctrlab,
            'testX':  test,         'testY':  ctslab,
            'valdX':  train[vlIdx], 'valdY':  cvlab}

    return data
