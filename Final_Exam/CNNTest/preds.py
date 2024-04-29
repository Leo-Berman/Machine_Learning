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

def main():

    # list to csv images
    datalist = "/home/tuo54571/Machine_Learning/Final_Exam/TEST/test.list"
    compare = "/home/tuo54571//Machine_Learning/Final_Exam/TEST/justlabels.txt"
    model = "cbtest.keras"
    
    files = read_lines(datalist)
    comp = map_str_to_int(read_lines(compare)).tolist()
    mymodel = keras.models.load_model(model)

    print(mymodel)

    predictions=[]
    for x in files:
        img = prep_pixels(np.array([read_image(x)]))
        prediction = mymodel.predict(img)
        prediction  = prediction.argmax(axis=-1)[0]
        predictions.append(prediction)

    corr = 0
    for x,y in list(zip(comp,predictions)):
        print("Actual = ",x,"Pred = ",y)
        if x == y:
            corr+=1
    print("Accuracy rate = ",corr/len(comp))
if __name__ == "__main__":
    main()
