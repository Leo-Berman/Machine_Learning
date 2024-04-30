#!/usr/bin/env python
import models

# my custom function for generating a dataset
from datagen import data_gen
from models import define_resnet,define_base

# Global variables
RESOLUTION = 480

# testing bottleneck
import time

def main():

    
    # read the data set
    dir = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHYPART"
    lbl = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHYPART/datalabels.txt"
    images,labels = data_gen(dir,lbl)

    # define model parameters
    modeltype = "base"
    epochhs = 10
    learnrate = .0001
    model = None
    
    # defin the model
    if modeltype == "resnet":
        model = define_resnet(len(labels),RESOLUTION,learningrate=learnrate)
    elif modeltype == "base":
        model = define_base(len(labels),RESOLUTION,learningrate=learnrate)
    else:
        print("invalid model")
    modelname = "models/"+modeltype+"_E"+str(epochhs)+"_LR"+str(learnrate).replace(".","_")[1:]+"_0H_1000U"
    print("Model out = ",modelname+".keras")
    
    # fit and save the model

    start = time.time()
    history = model.fit(images,epochs=epochhs)
    
    end=time.time()
    model.save(modelname+".keras")
    print("model outputted")
    
    datalen=0
    for x in images:
        for y in x:
            for z in y:
                datalen+=1
    datalen = datalen//2
    print("Time to process",datalen,"files = ",end-start,"seconds")

if __name__ == "__main__":
    main()
