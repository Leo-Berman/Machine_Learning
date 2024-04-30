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

from functs import map_str_to_int,read_lines,read_image,prep_pixels

import numpy as np

from datagen import data_gen

RESOLUTION = 480

def main():

    # list to csv images
    model = "/home/tuo54571/Machine_Learning/Final_Exam/CNNTest/models/base_E10_LR_001_0H_1000U.keras"
    mymodel = keras.models.load_model(model)
    dir = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHYPART"
    lbl = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHYPART/datalabels.txt"
    dataset,uninquelabels = data_gen(dir,lbl)

    labels = map_str_to_int(read_lines(lbl))
    
    predictions = mymodel.predict(dataset)
    predictions = [np.argmax(x) for x in predictions]
    corr = 0
    for x,y in list(zip(labels,predictions)):
        print("Actual = ",x,"Pred = ",y)
        if x == y:
            corr+=1
    print("Accuracy rate = ",corr/len(predictions)*100,"%")
if __name__ == "__main__":
    main()
