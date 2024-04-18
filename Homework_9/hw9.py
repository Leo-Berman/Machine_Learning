#!/usr/bin/env python
from pathlib import Path
import os
import pandas as pd
from sklearn.decomposition import PCA as PCA

def bagging(data):
    pass
    
def get_parent_dir():

    # get the filepath of the driver function
    #
    driver_path = __file__
    
    # convert to a Path datatype
    #
    path = Path(driver_path)

    # get the absolute path of the parent folder and return it
    # 
    parent_folder = path.parent.absolute()
    return parent_folder

def get_data(file_path:str):
    df = pd.read_csv(file_path)
    df.columns = ["Classes","x1","x2"]
    return df

def get_csv(path:str):

    # return the paths of each csv file
    #
    train =   get_data(os.path.join(path,"train.csv"))
    dev =     get_data(os.path.join(path,"dev.csv"))
    eval =    get_data(os.path.join(path,"eval.csv"))
    return train,dev,eval

def main():
    # get the parent directory of the driver function
    #
    parent_folder = get_parent_dir()

    set_10_path = os.path.join(parent_folder, "Set_10")
    set_10_train,set_10_dev,set_10_eval = get_csv(set_10_path)

    print(set_10_train)


if __name__ == "__main__":
    main()
