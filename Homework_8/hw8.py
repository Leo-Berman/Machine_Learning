#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.cluster import KMeans as KNM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RNF
from sklearn.decomposition import PCA as PCA
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
    elif model_type == "QDA":
        model = QDA()
    elif model_type == "PCA":
        pca = PCA(n_components=1)
        pca.fit(train_features,train_classes)
        train_features = pca.transform(train_features)
        dev_features = pca.transform(dev_features)
        eval_features = pca.transform(eval_features)
        model = QDA()
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

def plot_knn(x_axis:list,y_axis:list,name:str):
    plt.plot(x_axis,y_axis)
    plt.ylim((0,1))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of nearest neighbors")
    plt.title(name)
    plt.savefig("KNN_"+name+".png")
    plt.cla()

def plot_knm(x_axis:list,y_axis:list,name:str):
    plt.plot(x_axis,y_axis)
    plt.ylim((0,1))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of clusters")
    plt.title(name)
    plt.savefig("KNM_"+name+".png")
    plt.cla()

def score_knn(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str):
    train_classes,train_features = get_classes_features(train)
    dev_classes,dev_features = get_classes_features(dev)
    eval_classes,eval_features = get_classes_features(eval)
    
    train_scores = []
    dev_scores = []
    eval_scores = []
    k_neighbors = []
    x_axis = range(1,50)
    
    for i in x_axis:
        model = KNN(n_neighbors = i)
        model.fit(train_features,train_classes)
        
        train_scores.append(model.score(train_features,train_classes))
        dev_scores.append(model.score(dev_features,dev_classes))
        eval_scores.append(model.score(eval_features,eval_classes))
        k_neighbors.append(i)
    
    
    plot_knn(x_axis,train_scores,"Training"+name)
    plot_knn(x_axis,dev_scores,"Development"+name)
    plot_knn(x_axis,eval_scores,"Evaluation"+name)

    
    ideal_dev = dev_scores.index(max(dev_scores))
    print(name,"KNN",":\n\tTraining (Number of Neighbors = ",ideal_dev,") : ", train_scores[ideal_dev],"\n\tDevelopment (Number of Neighbors = ",ideal_dev,") : ", dev_scores[ideal_dev],"\n\tEvaluation (Number of Neighbors = ",ideal_dev,") : ", eval_scores[ideal_dev])

def score_knm(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str):
    train_classes,train_features = get_classes_features(train)
    dev_classes,dev_features = get_classes_features(dev)
    eval_classes,eval_features = get_classes_features(eval)
    
    train_scores = []
    dev_scores = []
    eval_scores = []
    k_clusters = []
    x_axis = range(1,50)
    
    for i in x_axis:
        model = KNM(n_clusters = i+1)
        model.fit(train_features,train_classes)

        train_preds = model.predict(train_features)
        dev_preds = model.predict(dev_features)
        eval_preds = model.predict(eval_features)

        train_wrongs = 0
        dev_wrongs = 0
        eval_wrongs = 0

        for j,x in enumerate(train_preds):
            if x != train_classes[j]:
                train_wrongs+=1
        train_scores.append(1-(train_wrongs/len(train_classes)))

        for j,x in enumerate(dev_preds):
            if x != dev_classes[j]:
                dev_wrongs+=1
        dev_scores.append(1-(dev_wrongs/len(dev_classes)))
        
        for j,x in enumerate(eval_preds):
            if x != eval_classes[j]:
                eval_wrongs+=1
        eval_scores.append(1-(eval_wrongs/len(dev_classes)))
        

        k_clusters.append(i+1)
    
    
    plot_knm(x_axis,train_scores,"Training"+name)
    plot_knm(x_axis,dev_scores,"Development"+name)
    plot_knm(x_axis,eval_scores,"Evaluation"+name)
    
    
    ideal_dev = dev_scores.index(max(dev_scores))
    print(name,"KNM",":\n\tTraining (Number of Clusters = ",ideal_dev,") : ", train_scores[ideal_dev],"\n\tDevelopment (Number of Clusters = ",ideal_dev,") : ", dev_scores[ideal_dev],"\n\tEvaluation (Number of Clusters = ",ideal_dev,") : ", eval_scores[ideal_dev])


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

    # create a path to the data
    data_folder = os.path.join(parent_folder,"Data")

    # extract the train, dev, and eval for each dataset
    #
    set_8_path = os.path.join(data_folder, "Set_08")
    set_8_train,set_8_dev,set_8_eval = get_csv(set_8_path)
    set_8_traindev = np.vstack([set_8_train,set_8_dev])
    print("Set 8 TrainDev = ",set_8_traindev)
    set_9_path = os.path.join(data_folder, "Set_09")
    set_9_train,set_9_dev,set_9_eval = get_csv(set_9_path)

    set_10_path = os.path.join(data_folder, "Set_10")
    set_10_train,set_10_dev,set_10_eval = get_csv(set_10_path)

    # score the pca for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "PCA")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "PCA")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "PCA")

    
    # score the lda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "LDA")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "LDA")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "LDA")

    # score the qda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "QDA")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "QDA")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "QDA")
    
    # score the knn for each set
    #
    score_knn(set_8_train,set_8_dev,set_8_eval,"Set 08")
    score_knn(set_9_train,set_9_dev,set_9_eval,"Set 09")
    score_knn(set_10_train,set_10_dev,set_10_eval,"Set 10")

    # score the knm for each set
    #
    score_knm(set_8_train,set_8_dev,set_8_eval,"Set 08")
    score_knm(set_9_train,set_9_dev,set_9_eval,"Set 09")
    score_knm(set_10_train,set_10_dev,set_10_eval,"Set 10")

    # score the rnf for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "RNF")
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "RNF")
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "RNF")

    
if __name__ == "__main__":
    main()
