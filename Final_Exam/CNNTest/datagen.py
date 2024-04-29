#!/usr/bin/env python

from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from functs import read_lines,map_str_to_int

# class hashing
from bitstring import BitArray


def data_gen():
    
    dirpath = "/home/tuo54571/Machine_Learning/Final_Exam/KERASFORMAT"
    lblpath = "/home/tuo54571/Machine_Learning/Final_Exam/TRAINUNHEALTHY/datalabels.txt"
    
    strlabels = read_lines(lblpath)
    labels = map_str_to_int(strlabels)
    return image_dataset_from_directory(dirpath,labels=labels.tolist(),label_mode="categorical",color_mode="grayscale",image_size=(480,480),batch_size=32),list(set(labels))
    

if __name__ == "__main__":
    main()
