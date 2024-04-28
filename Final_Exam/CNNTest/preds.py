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
from tensorflow import keras



# for opening images
from PIL import Image
from numpy import asarray
import numpy as np

# class hashing
from bitstring import BitArray

# Global variables
RESOLUTION = 480

# maps class names to integers
def map_str_to_int(labels):
    print(uniques)
    for i,x in enumerate(labels):
        ind = uniques.index(x)
        labels[i] = BitArray(bin=x).int
    return np.array(labels)

# reads filelists
def read_lists(datalist):
    with open(datalist) as datafile:
        lines = [line.rstrip() for line in datafile]
    lines.pop(0)
    return lines

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

def main():

    # list to csv images
    datalist = "/home/tuo54571/Machine_Learning/Final_Exam/TEST/test.list"
    model = "testmodel.keras"
    
    files = read_lists(datalist)
    
    mymodel = keras.models.load_model(model)

    print(mymodel)

    predictions=[]
    for x in files:
        print(x)
        img = prep_pixels(np.array([read_image(x)]))
        prediction = mymodel.predict(img)[0]
        prediction  = prediction.argmax(axis=-1)
        
        predictions.append(prediction)

    print(predictions)


if __name__ == "__main__":
    main()
