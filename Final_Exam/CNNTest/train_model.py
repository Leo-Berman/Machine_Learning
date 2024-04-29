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

    # list to csv images
    datalist = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHY/datalist.csv"
    
    images,labels = data_gen()
    #model = define_base(len(labels),RESOLUTION)
    model = define_resnet(len(labels),RESOLUTION)
    modelname = "resnettest"
    print("Model out = ",modelname)
    histories = []
    i = 0
    tmplist = []
    tmplabels = []




    trainlen = len(labels)
    epochhs = 20
    start = time.time()
    history = model.fit(images,epochs=epochhs)
    histories.append(history)    
    
    end=time.time()
    print("Time to process labels = ",end-start)
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
