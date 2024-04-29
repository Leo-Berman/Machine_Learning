#!/usr/bin/env python

# baseline cnn model for mnist from tutorial
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# resnet
from keras import applications
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D
from keras.models import Model

# for opening images
from PIL import Image
from numpy import asarray
import numpy as np

# class hashing
from bitstring import BitArray
# Global variables
RESOLUTION = 480

# testing bottleneck
import time

# maps class names to integers
def map_str_to_int(labels):
    print(labels[0])
    print(labels[1])
    for i,x in enumerate(labels):
        myint = BitArray(bin=x).uint
        labels[i] = myint
    print(list(set(labels)))

    return np.array(labels)

# reads filelists
def read_lists(datalist):
    with open(datalist) as datafile:
        lines = [line.rstrip().split(",") for line in datafile]
    # get rid of header
    lines.pop(0)
    lines.pop(0)
    labels = []
    files = []
    for x in lines:
        labels.append(x[0])
        files.append(x[1])
    return to_categorical(map_str_to_int(labels)),files

# read an image
def read_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((RESOLUTION,RESOLUTION))
    img = asarray(img)
    img = img.reshape(RESOLUTION,RESOLUTION,1)
    return img

# prep pixels
def prep_pixels(image):
    norm = image.astype('float32')
    norm = norm/255.0
    return norm

# define the model
# define resnet model
def define_resnet(num_classes):
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (RESOLUTION,RESOLUTION,1))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def define_base(classnums):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(RESOLUTION, RESOLUTION, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classnums, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
    #plt.boxplot(scores)
    #plt.show()


def main():

    # list to csv images
    datalist = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHY/datalist.csv"
    #datalist = "/home/Desktop/Github_Repos/Machine_Learning/Final_Exam/TEST/datalist.csv"
    labels,files = read_lists(datalist)
    model = define_base(len(labels[0]))
    #model = define_resnet(len(labels[0]))
    modelname = "cbtest"
    print("Model out = ",modelname)
    histories = []
    i = 0
    print(len(labels))
    print(len(files))
    tmplist = []
    tmplabels = []

    

    trainlen = 10
    epochhs = 35
    start = time.time()
    for label,file in list(zip(labels,files)):
        tmplist.append(read_image(file))
        tmplabels.append(label)
        
        if len(tmplist) == trainlen or (i + len(tmplist)==len(files)):
            images = prep_pixels(np.array(tmplist))
            if i == 0:
                history = model.fit(images,np.array(tmplabels),epochs=epochhs,batch_size=32,verbose=0)
            else:
                history = model.fit(images,np.array(tmplabels),epochs=epochhs,batch_size=32,verbose=0,callback=[checkpoint])                
            histories.append(history)    
            
            #model.train_on_batch(images,np.array(tmplabels))

            i+=len(tmplist)
            print(i,"Out of",len(files),"Files trained on")
            tmplist=[]
            tmplabels=[]
            end=time.time()
            print("Time to process",i,"labels = ",end-start)
            checkpoint = ModelCheckpoint(filepath="my_best_"+modelname+"_model.keras", 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
            

    model.save(modelname+'.keras')
    scores = []
    i = 1
    for label,file in list(zip(labels,files)):
        image = prep_pixels(np.array([read_image(file)]))
        _,acc=model.evaluate(image,np.array([label]),verbose=0)
        scores.append(acc)
        print("Evaluated on",i,"out of",len(files),"elements",_,acc)
        i+=1

    # learning curves ecg
    #summarize_diagnostics(histories)

    # summarize estimated performance ecg
    summarize_performance(scores)

if __name__ == "__main__":
    main()
