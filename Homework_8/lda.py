from pathlib import Path
import os
import pandas as pd
import numpy as np
def get_data(filepath:str):
    print(filepath)
    df = pd.read_csv(filepath)
    df.columns=["Classes","x1","x2"]
    return df

def get_parent_dir():
    driver_path = __file__
    path = Path(driver_path)
    parent_folder = path.parent.absolute()
    return parent_folder

def main():
    parent_folder = get_parent_dir()
    data_folder = os.path.join(parent_folder,"Data")

    set_8_path = os.path.join(data_folder, "Set_08")
    set_8_train =   get_data(os.path.join(set_8_path,"train.csv"))
    set_8_dev =     get_data(os.path.join(set_8_path,"dev.csv"))
    set_8_eval =    get_data(os.path.join(set_8_path,"eval.csv"))

    set_9_path = os.path.join(data_folder, "Set_09")
    set_9_train =   get_data(os.path.join(set_9_path,"train.csv"))
    set_9_dev =     get_data(os.path.join(set_9_path,"dev.csv"))
    set_9_eval =    get_data(os.path.join(set_9_path,"eval.csv"))

    set_10_path = os.path.join(data_folder, "Set_10")
    set_10_train =   get_data(os.path.join(set_10_path,"train.csv"))
    set_10_dev =     get_data(os.path.join(set_10_path,"dev.csv"))
    set_10_eval =    get_data(os.path.join(set_10_path,"eval.csv"))
    
if __name__ == "__main__":
    main()