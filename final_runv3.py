import numpy as np
import os

import detection
import helper
#import detection
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import regularizers
import keras.utils as ku
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import pickle
import matplotlib.pyplot as ply

test_dir = "test"
train_dir = "train"
OUTPUT_DIR = "output"

train = os.path.join(train_dir, "train")
test  = os.path.join(test_dir, "test")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)


def designedCNN_Model():
    data = helper.prepDataforCNN(numChannel = 3, feat_norm = True)
    trainX = data["trainX"]
    valdX  = data["valdX"]
    trainY = data["trainY"]
    valdY  = data["valdY"]

    _,row, col,channel = trainX.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 25
    batch_size = 64

    optim = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config = config)

    input = keras.Input(shape=(row,col,channel), name='customModel')
    M = Conv2D(16,(3,3),activation='relu',padding='same',name = 'conv_16_1')(input)
    M = Conv2D(16,(3, 3), activation ='relu', padding='same',name = 'conv_16_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_01')(M)
    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)
    M = Dropout(0.5)(M)

    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_01')(M)
    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_1')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same', name = 'conv2_64_2')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_1')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_2')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)

    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_5')(M)
    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_6')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)
    M = Dropout(0.5)(M)

    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_1')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_2')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_3')(M)

    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_1')(M)
    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides= 1)(M)
    M = Dropout(0.25)(M)


    Mout = Flatten()(M)
    Mout = Dense(2048, activation='relu', name = 'FC1_2048')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC1_1024')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC2_1024')(Mout)


    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(Mout)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(Mout)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(Mout)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(Mout)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(Mout)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(Mout)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM, numB_SM]

    svhnModel = keras.Model(inputs = input, outputs = out)

    lr_metric = get_lr_metric(optim)
    svhnModel.compile(loss = 'sparse_categorical_crossentropy', #ceLoss ,
                      optimizer= optim,
                      metrics=  ['accuracy']) #[])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 2,
                                                  cooldown= 1,
                                                  min_lr = 0.00001)
    svhnModel.summary()
    callback = []
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/designedBGRClassifier_weights.{epoch:02d}-{val_loss:.2f}.h5.hdf5',
                                                   save_weights_only=True,
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    tb = keras.callbacks.TensorBoard(log_dir = 'logs',
                                      write_graph = True,
                                      batch_size = batch_size,
                                      write_images = True)
    es = keras.callbacks.EarlyStopping(monitor= 'loss',  #'dig1_loss',
                                       min_delta=0.000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(tb)
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)

    designHist = svhnModel.fit(x = trainX,
                              y = trainY,
                              batch_size = batch_size,
                              epochs = epochs,
                              verbose=1,
                              shuffle = True,
                              validation_data = (valdX, valdY),
                              callbacks= callback)

    print(designHist.history.keys())
    modName = 'customDesign'
    print(designHist.history.keys())
    createSaveMetricsPlot(designHist,modName,data,svhnModel)


if __name__ == "__main__":
    designedCNN_Model()
