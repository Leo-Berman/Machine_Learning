import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
def find_col_range(incol:list):
    return max(incol) - min(incol)

def quantize_list(incol:list):
    list_range = find_col_range(incol)
    ret_list = []
    for x in incol:
        ret_list.append(round((x/list_range)*128))
    
    return ret_list

def calculate_entropy(class_list):
    class_probs = {}
    classes = set(class_list)
    for x in classes:
        class_probs[x]=class_list.count(x)/len(class_list)

    entropy = 0
    for x in range(len(class_probs)):
        entropy += -class_probs[x]*np.log2(class_probs[x])
    
    return entropy
def main():

    # read data in  
    #
    header = ["class","x_coord","y_coord"]
    traindf = pd.read_csv("train.csv",names = header)
    devdf = pd.read_csv("dev.csv",names = header)

    # convert 2d coords to lists
    #
    train_xcoords = traindf["x_coord"]
    train_ycoords = traindf["y_coord"]
    dev_xcoords = devdf["x_coord"]
    dev_ycoords = devdf["y_coord"]

    # quantize lists
    #
    quantized_train_xcoords = quantize_list(train_xcoords)
    quantized_train_ycoords = quantize_list(train_ycoords)
    quantized_dev_xcoords = quantize_list(dev_xcoords)
    quantized_dev_ycoords = quantize_list(dev_ycoords)

    quantized_train_xcoords.extend(quantized_dev_xcoords)
    quantized_train_ycoords.extend(quantized_dev_ycoords)
    total_classes = list(np.hstack([traindf["class"],devdf["class"]]))
    total_x = quantized_train_xcoords
    total_y = quantized_train_ycoords
    total_xy = np.vstack([total_x,total_y])
    # print(total_xy)

    # Find entropy of each vector
    #
    entropy = calculate_entropy(total_classes)

    print(entropy)

if __name__ == "__main__":
    main()