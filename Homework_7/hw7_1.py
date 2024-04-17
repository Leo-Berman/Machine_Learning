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

def calculate_1d_entropy(class_list):

    uniques = set(class_list)
    probs = {}
    for x in uniques:
        probs[x]=class_list.count(x)/len(class_list)

    entropy = 0
    for x in probs:
        entropy-=probs[x]*np.log(probs[x])
    return entropy

def calculate_2d_entropy(class_list1,class_list2):
    
    l = list(zip(class_list1,class_list2))
    uniques = set(l)
    probs = {}
    for x in uniques:
        probs[x]=l.count(x)/len(l)

    entropy = 0
    for x in probs:
        entropy-=probs[x]*np.log(probs[x])
    return entropy

def main():

    # read data in  
    #
    header = ["class","x_coord","y_coord"]
    traindf = pd.read_csv("train.csv",names = header)
    devdf = pd.read_csv("dev.csv",names = header)

    # convert 2d coords to lists
    #
    train_x1 = traindf["x_coord"]
    train_x2 = traindf["y_coord"]
    dev_x1 = devdf["x_coord"]
    dev_x2 = devdf["y_coord"]

    # quantize lists
    #
    quantized_train_x1 = quantize_list(train_x1)
    quantized_train_x2 = quantize_list(train_x2)
    quantized_dev_x1 = quantize_list(dev_x1)
    quantized_dev_x2 = quantize_list(dev_x2)

    rng = np.random.default_rng()
    quantized_unif_x1 = rng.integers(0,129,1000).tolist()
    quantized_unif_x2 = rng.integers(0,129,1000).tolist()
    

    
    
    # Find entropy of each vector
    #
    H_x1_train = calculate_1d_entropy(quantized_train_x1)
    H_x2_train = calculate_1d_entropy(quantized_train_x2)
    H_x1_dev = calculate_1d_entropy(quantized_dev_x1)
    H_x2_dev = calculate_1d_entropy(quantized_dev_x2)
    H_x1_unif = calculate_1d_entropy(quantized_unif_x1)
    H_x2_unif = calculate_1d_entropy(quantized_unif_x2)


    
    H_x1x2_train = calculate_2d_entropy(quantized_train_x1,quantized_train_x2)
    H_x1x2_dev  = calculate_2d_entropy(quantized_dev_x1,quantized_dev_x2)
    H_x1x2_unif  = calculate_2d_entropy(quantized_unif_x1,quantized_unif_x2)

    train_mutualinfo = H_x1_train + H_x2_train - H_x1x2_train
    dev_mutualinfo   = H_x1_dev + H_x2_dev - H_x1x2_dev
    unif_mutualinfo   = H_x1_unif + H_x2_unif - H_x1x2_unif
    
    H_x1x2_train_conditional = H_x1_train - train_mutualinfo
    H_x2x1_train_conditional = H_x2_train - train_mutualinfo
    
    
    H_x1x2_dev_conditional = H_x1_dev - dev_mutualinfo
    H_x2x1_dev_conditional = H_x2_dev - dev_mutualinfo

    H_x1x2_unif_conditional = H_x1_unif - unif_mutualinfo
    H_x2x1_unif_conditional = H_x2_unif - unif_mutualinfo
    
    
    print("Train H(x1)".ljust(20), "= ",H_x1_train)
    print("Train H(x2)".ljust(20), "= ",H_x2_train)
    print("Train H(x1,x2)".ljust(20),"= ",H_x1x2_train)
    print("Train H(x1|x2)".ljust(20),"= ",H_x1x2_train_conditional)
    print("Train H(x2|x1)".ljust(20),"= ",H_x2x1_train_conditional) 
    print("Train I(x1,x2)".ljust(20),"= ",train_mutualinfo)
    print("\n")
    
    print("Dev H(x1)".ljust(20), "= ",H_x1_dev)
    print("Dev H(x2)".ljust(20), "= ",H_x2_dev)
    print("Dev H(x1,x2)".ljust(20),"= ",H_x1x2_dev)
    print("Dev H(x1|x2)".ljust(20),"= ",H_x1x2_dev_conditional)
    print("Dev H(x2|x1)".ljust(20),"= ",H_x2x1_dev_conditional) 
    print("Dev I(x1,x2)".ljust(20),"= ",dev_mutualinfo)
    print("\n")

        
    print("Unif H(x1)".ljust(20), "= ",H_x1_unif)
    print("Unif H(x2)".ljust(20), "= ",H_x2_unif)
    print("Unif H(x1,x2)".ljust(20),"= ",H_x1x2_unif)
    print("Unif H(x1|x2)".ljust(20),"= ",H_x1x2_unif_conditional)
    print("Unif H(x2|x1)".ljust(20),"= ",H_x2x1_unif_conditional) 
    print("Unif I(x1,x2)".ljust(20),"= ",unif_mutualinfo)
    print("\n")
if __name__ == "__main__":
    main()
