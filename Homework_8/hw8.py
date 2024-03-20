from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RNF
def get_classes_features(data:pd.DataFrame):

    # get the classes into a 1d numpy array
    #
    classes   =   np.array(data['Classes'].to_list()).ravel()

    # get the feature vectors into a 2 column n row array and return them
    #
    feature_vectors  =   np.stack([np.array(data['x1']),np.array(data['x2'])],axis=1)
    return classes,feature_vectors

def score_model(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str,model_type:str):
    
    
    train_classes,train_features = get_classes_features(train)
    dev_classes,dev_features = get_classes_features(dev)
    eval_classes,eval_features = get_classes_features(eval)
    
    # fit the model
    #
    if model_type == "LDA":
        model = LDA()
    elif model_type == "RNF":
        model = RNF()
    else:
        print("Invalid model type")
        return
    
    model.fit(train_features,train_classes)
    
    # score the data
    #
    train_score = model.score(train_features,train_classes)
    dev_score = model.score(dev_features,dev_classes)
    eval_score = model.score(eval_features,eval_classes)

    # print the data 
    #
    print(name,model_type,":\n\tTraining : ", train_score,"\n\tDevelopment : ",dev_score,"\n\tEvaluation : ",eval_score)

def score_knn(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str):
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

    # create a path to the data
    data_folder = os.path.join(parent_folder,"Data")

    # extract the train, dev, and eval for each dataset
    #
    set_8_path = os.path.join(data_folder, "Set_08")
    set_8_train,set_8_dev,set_8_eval = get_csv(set_8_path)

    set_9_path = os.path.join(data_folder, "Set_09")
    set_9_train,set_9_dev,set_9_eval = get_csv(set_9_path)

    set_10_path = os.path.join(data_folder, "Set_10")
    set_10_train,set_10_dev,set_10_eval = get_csv(set_10_path)

    # score the lda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "LDA")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "LDA")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "LDA")

    # score the lda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "KNN")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "KNN")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "KNN")

    # score the lda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "RNF")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "RNF")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "RNF")
if __name__ == "__main__":
    main()