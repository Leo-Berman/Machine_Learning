from PIL import Image
from numpy import asarray
import numpy as np
from bitstring import BitArray

def map_str_to_int(labels):
    for i,x in enumerate(labels):
        myint = BitArray(bin=x).uint
        labels[i] = myint
    print("Unique Labels = ",list(set(labels)))

    return np.array(labels)

def read_lines(myfile):
    with open(myfile) as datafile:
        lines = [line.rstrip() for line in datafile]
    return lines

def read_image(in_image,resolution):
    img = Image.open(in_image).convert('L')
    img = img.resize((resolution,resolution))
    img = asarray(img)
    img = img.reshape(resolution,resolution,1)
    return img

def prep_pixels(image):
    norm = image.astype('float32')
    norm = norm/255.0
    return norm


