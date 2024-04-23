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
import PCA
def plot_by_class(indata):
    labels,data = get_classes_features(indata)
    classes = {}
    for i,x in enumerate(labels):
        if x not in classes:
            classes[x] = [data[i].tolist()]
        else:
            classes[x].append(data[i].tolist())
    colors = ['red','blue','green']
    for x in classes:
        xax,yax = zip(*classes[x])
        plt.scatter(xax,yax)

def plot_decisions(train,eval,modeltype,name):

    train_classes,train_features = get_classes_features(train)
    eval_classes,eval_features = get_classes_features(eval)


    x_max = max(eval_features[:,0])
    x_min = min(eval_features[:,0])
    y_max = max(eval_features[:,1])
    y_min = min(eval_features[:,1])

    xx = np.linspace(x_max,x_min,40)
    yy = np.linspace(y_max,y_min,40)
    
    xvec,yvec = np.meshgrid(xx,yy)
    
    model = None
    match modeltype:
        case "QDA":
            model = QDA()
            model.fit(train_features,train_classes)
            print("QDA Trained")
        case "RNF":
            model = RNF()
            model.fit(train_features,train_classes)
            print("RNF Trained")
        case "KNN":
            model = KNN(n_neighbors=1)
            print("KNN Trained")
            model.fit(train_features,train_classes)
        case "KNM":
            model = KNM(n_clusters=1)
            print("KNM Trained")
            model.fit(train_features,train_classes)
        case "LDA":
            model = LDA()
            print("LDA Trained")
            model.fit(train_features,train_classes)
        case "PCA":
            model = train_pca(train_features,train_classes)
            print("PCA Trained")
        case _:
            print("Invalid model")
            return
    print("Past match")
    pinkx = []
    pinky = []
    purpx = []
    purpy = []
    grayx = []
    grayy = []
    plot_by_class(eval)
    for i,x in enumerate(xvec[0]):
        for y1 in yvec:
            for y in y1:
                prediction = model.predict(np.array([x,y]).reshape(1,-1))
                
                
                if prediction == 0:
                    pinkx.append(x)
                    pinky.append(y)
                if prediction == 1:
                    purpx.append(x)
                    purpy.append(y)
                if prediction == 2:
                    grayx.append(x)
                    grayy.append(y)
        print(str(i) + "/" + str(len(xvec[0])) + " done")
    plt.scatter(pinkx,pinky,color="pink",alpha=.01)
    print("pink plotted")
    plt.scatter(purpx,purpy,color="purple",alpha=.01)
    print("purp plotted")
    plt.scatter(grayx,grayy,color="gray",alpha=.01)
    print("gray pltoted")
    plt.savefig(name+modeltype+"decisions.png")
    print(name+modeltype+"decisions.png has been saved")
def get_classes_features(data:pd.DataFrame):

    # get the classes into a 1d numpy array
    #
    classes   =   np.array(data['Classes'].to_list()).ravel()

    # get the feature vectors into a 2 column n row array and return them
    #
    feature_vectors  =   np.stack([np.array(data['x1']),np.array(data['x2'])],axis=1)
    return classes,feature_vectors

def score_pca(data,labels,model):
    mydata = np.array(list(zip(labels,data[:,0],data[:,1])))
    return model.eval(newdata=mydata)

def train_pca(data,labels):
    model = PCA.custom_PCA()
    mydata = np.array(list(zip(labels,data[:,0],data[:,1])))
    model.train(mydata)
    return model

def score_model(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str,model_type:str,traindevflag=False):
    
    
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
        model = None
        if traindevflag == False:
            model = train_pca(train_features,train_classes)
        else:
            feats = np.vstack((train_features,dev_features))
            classes = np.hstack((train_classes,dev_classes))
            model = train_pca(feats,classes)

        train_score = score_pca(train_features,train_classes,model)
        dev_score = score_pca(dev_features,dev_classes,model)
        eval_score = score_pca(eval_features,eval_classes,model)
        # print the data 
        #
        print("traindev" if traindevflag else "",name,model_type,":\n\tTraining : ", train_score,"\n\tDevelopment : ",dev_score,"\n\tEvaluation : ",eval_score)

        return

    else:
        print("Invalid model type")
        return
    if traindevflag == False:
        model.fit(train_features,train_classes)
    else:
        feats = np.vstack((train_features,dev_features))
        classes = np.hstack((train_classes,dev_classes))
        model.fit(feats,classes)    
    # score the data
    #
    train_score = model.score(train_features,train_classes)
    dev_score = model.score(dev_features,dev_classes)
    eval_score = model.score(eval_features,eval_classes)

    # print the data 
    #
    print("traindev" if traindevflag else "",name,model_type,":\n\tTraining : ", train_score,"\n\tDevelopment : ",dev_score,"\n\tEvaluation : ",eval_score)

def plot_knn(x_axis:list,y_axis:list,name:str,traindevflag=False):
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

def score_knn(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str,traindevflag=False):
    train_classes,train_features = get_classes_features(train)
    dev_classes,dev_features = get_classes_features(dev)
    eval_classes,eval_features = get_classes_features(eval)
    
    train_scores = []
    dev_scores = []
    eval_scores = []
    k_neighbors = []
    x_axis = range(1,50)
    
    for i in x_axis:
        model = KNN(n_neighbors = i+1)
        if traindevflag==False:
             model.fit(train_features,train_classes)
        else:
             feats = np.vstack((train_features,dev_features))
             classes = np.hstack((train_classes,dev_classes))
             model.fit(feats,classes)
        
        train_scores.append(model.score(train_features,train_classes))
        dev_scores.append(model.score(dev_features,dev_classes))
        eval_scores.append(model.score(eval_features,eval_classes))
        k_neighbors.append(i)
    
    
    #plot_knn(x_axis,train_scores,"Training"+name)
    #plot_knn(x_axis,dev_scores,"Development"+name)
    #plot_knn(x_axis,eval_scores,"Evaluation"+name)

    
    ideal_dev = dev_scores.index(max(dev_scores))
    print("traindev" if traindevflag else "",name,"KNN",":\n\tTraining (Number of Neighbors = ",ideal_dev,") : ", train_scores[ideal_dev],"\n\tDevelopment (Number of Neighbors = ",ideal_dev,") : ", dev_scores[ideal_dev],"\n\tEvaluation (Number of Neighbors = ",ideal_dev,") : ", eval_scores[ideal_dev])

def score_knm(train:pd.DataFrame,dev:pd.DataFrame,eval:pd.DataFrame,name:str,traindevflag=False):
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
        if traindevflag==False:
             model.fit(train_features,train_classes)
        else:
             feats = np.vstack((train_features,dev_features))
             classes = np.hstack((train_classes,dev_classes))
             model.fit(feats,classes)
        
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
    
    
    #plot_knm(x_axis,train_scores,"Training"+name)
    #plot_knm(x_axis,dev_scores,"Development"+name)
    #plot_knm(x_axis,eval_scores,"Evaluation"+name)
    
    
    ideal_dev = dev_scores.index(max(dev_scores))
    print("traindev" if traindevflag else "",name,"KNM",":\n\tTraining (Number of Clusters = ",ideal_dev,") : ", train_scores[ideal_dev],"\n\tDevelopment (Number of Clusters = ",ideal_dev,") : ", dev_scores[ideal_dev],"\n\tEvaluation (Number of Clusters = ",ideal_dev,") : ", eval_scores[ideal_dev])


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
    set_9_path = os.path.join(data_folder, "Set_09")
    set_9_train,set_9_dev,set_9_eval = get_csv(set_9_path)

    set_10_path = os.path.join(data_folder, "Set_10")
    set_10_train,set_10_dev,set_10_eval = get_csv(set_10_path)

    '''
    # Train on train
    
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
    
    # Train on traindev
     # score the pca for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "PCA",traindevflag=True)
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "PCA",traindevflag=True)
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "PCA",traindevflag=True)
    
    
    # score the lda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "LDA",traindevflag=True)
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "LDA",traindevflag=True)
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "LDA",traindevflag=True)

    # score the qda for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "QDA",traindevflag=True)
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "QDA",traindevflag=True)
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "QDA",traindevflag=True)
    
    # score the knn for each set
    #
    score_knn(set_8_train,set_8_dev,set_8_eval,"Set 08",traindevflag=True)
    score_knn(set_9_train,set_9_dev,set_9_eval,"Set 09",traindevflag=True)
    score_knn(set_10_train,set_10_dev,set_10_eval,"Set 10",traindevflag=True)

    # score the knm for each set
    #
    score_knm(set_8_train,set_8_dev,set_8_eval,"Set 08",traindevflag=True)
    score_knm(set_9_train,set_9_dev,set_9_eval,"Set 09",traindevflag=True)
    score_knm(set_10_train,set_10_dev,set_10_eval,"Set 10",traindevflag=True)

    # score the rnf for each set
    #
    score_model(set_8_train,set_8_dev,set_8_eval,"Set 08", "RNF",traindevflag=True)
    score_model(set_9_train,set_9_dev,set_9_eval,"Set 09", "RNF",traindevflag=True)
    score_model(set_10_train,set_10_dev,set_10_eval,"Set 10", "RNF",traindevflag=True)



'''
    plot_by_class(set_8_train)
    plt.savefig('TrainSet 08.png')
    plt.cla()
    plot_by_class(set_9_train)
    plt.savefig('TrainSet 09.png')
    plt.cla()
    plot_by_class(set_10_train)
    plt.savefig('TrainSet 10.png')
    plt.cla()
    
    plot_by_class(set_8_dev)
    plt.savefig('DevSet 08.png')
    plt.cla()
    plot_by_class(set_9_dev)
    plt.savefig('DevSet 09.png')
    plt.cla()
    plot_by_class(set_10_dev)
    plt.savefig('DevSet 10.png')
    plt.cla()


    plot_by_class(set_8_eval)
    plt.savefig('EvalSet 08.png')    
    plt.cla()
    plot_by_class(set_9_eval)
    plt.savefig('EvalSet 09.png')
    plt.cla()
    plot_by_class(set_10_eval)
    plt.savefig('EvalSet 10.png')
    plt.cla()

    plot_decisions(set_8_train,set_8_eval,'PCA','set8')
    plot_decisions(set_8_train,set_8_eval,'QDA','set8')
    plot_decisions(set_8_train,set_8_eval,'RNF','set8')
    plot_decisions(set_8_train,set_8_eval,'KNN','set8')
    plot_decisions(set_8_train,set_8_eval,'KNM','set8')
    plot_decisions(set_8_train,set_8_eval,'LDA','set8')
    plot_decisions(set_8_train,set_8_eval,'PCA','set9')
    plot_decisions(set_8_train,set_8_eval,'QDA','set9')
    plot_decisions(set_8_train,set_8_eval,'RNF','set9')
    plot_decisions(set_8_train,set_8_eval,'KNN','set9')
    plot_decisions(set_8_train,set_8_eval,'KNM','set9')
    plot_decisions(set_8_train,set_8_eval,'LDA','set9')
    plot_decisions(set_8_train,set_8_eval,'PCA','set10')
    plot_decisions(set_8_train,set_8_eval,'QDA','set10')
    plot_decisions(set_8_train,set_8_eval,'RNF','set10')
    plot_decisions(set_8_train,set_8_eval,'KNN','set10')
    plot_decisions(set_8_train,set_8_eval,'KNM','set10')
    plot_decisions(set_8_train,set_8_eval,'LDA','set10')
    
if __name__ == "__main__":
    main()
