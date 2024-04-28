#!/usr/bin/env python

# baseline cnn model for mnist from tutorial
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# for opening images
from PIL import Image
from numpy import asarray
import numpy as np

# Global variables
RESOLUTION = 480
def map_str_to_int(labels):
    uniques = list(set(labels))
    for i,x in enumerate(labels):
        ind = uniques.index(x)
        labels[i] = ind
    return np.array(labels)
    

def load_dataset_ecg(datalist):
    
    with open(datalist) as datafile:
        lines = [line.rstrip().split(",") for line in datafile]

    # get rid of header
    lines.pop(0)


    labels = []
    files = []


    
    for x in lines:
        labels.append(x[0])
        img = Image.open(x[1]).convert('L')
        #print(img.size)
        # change for resolution
        img = img.resize((RESOLUTION,RESOLUTION))
        img = asarray(img)
        files.append(img)
    files = np.array(files)
    labels = map_str_to_int(labels)
    files = files.reshape((files.shape[0], RESOLUTION, RESOLUTION, 1))
    labels = to_categorical(labels)
    
    return files,labels

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images

    return train_norm, test_norm

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(RESOLUTION, RESOLUTION, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
#        print(dataX[train_ix],dataX[test_ix])
        print(dataX[train_ix].shape,dataX[test_ix].shape)
#        print(dataY[train_ix],dataY[test_ix])
        print(dataY[train_ix].shape,dataY[test_ix].shape)

        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

    # summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    
# run the test harness for evaluating a model
def run_ecg_test_harness():

    
    datalist = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHY/datalist.csv"
    # load dataset ecq
    trainX, trainY = load_dataset_ecg(datalist)
    testX, testY = load_dataset_ecg(datalist)
    i = 0
    model = define_model()
    histories = []
    for W,X,Y,Z in list(zip(trainX,trainY,testX,testY)):                                        
        # prepare pixel data
        tmptrainX, tmptestX = prep_pixels(np.array([W]), np.array([Y]))
        #print("TrainX = ",trainX.shape)
        #print("TrainY = ",trainY.shape)                                                       
        histories.append(model.fit(tmptrainX,np.array([X]),epochs=10, batch_size=32, validation_data=(tmptestX, np.array([Z])), verbose=0))
        i+=1
        print(i,"Out of",len(trainX),"Files trained on")

    scores = []
    i = 1
    for W,X,Y,Z in list(zip(trainX,trainY,testX,testY)):
        tmptrainX, tmptestX = prep_pixels(np.array([W]), np.array([Y]))
        _,acc=model.evaluate(tmptestX,np.array([Z]),verbose=0)
        scores.append(acc)
        print("Evaluated on",i,"out of",len(testX),"elements,",_,acc)
        i+=1
    # prepare pixel data ecg
    #trainX, testX = prep_pixels(trainX, testX)

    #print("TrainX = ",trainX.shape)
    #print("TrainY = ",trainY.shape)
    # evaluate model ecg
    #scores, histories = evaluate_model(trainX, trainY)

    # learning curves ecg
    summarize_diagnostics(histories)

    # summarize estimated performance ecg
    summarize_performance(scores)

def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    
    #print("TrainX = ",trainX.shape)
    #print("TrainY = ",trainY.shape)
    
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)

    # learning curves
    summarize_diagnostics(histories)

    # summarize estimated performance
    summarize_performance(scores)

# entry point, run the test harness
run_ecg_test_harness()
#run_test_harness()
