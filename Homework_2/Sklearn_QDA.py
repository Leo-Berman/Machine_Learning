from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
import pandas as pd
import sys
def hw_data():
    # read in the data from the csv files
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    
    train_coords = np.array(list(zip(train[:,1],train[:,2])))
    eval_coords = np.array(list(zip(eval[:,1],eval[:,2])))

    # Set and train the algorithm
    algo = QDA()
    algo.fit(train_coords,train[:,0])

    # Evaluate the algorithm
    print("Evaluation accuracy rate: ",1-algo.score(eval_coords,eval[:,0]))
    print("Training accuracy rate: ",1-algo.score(train_coords,train[:,0]))

def debug_data():
    # read in the data from the csv files
    train = pd.read_csv("debug2.csv",comment = "#").to_numpy()
    eval = pd.read_csv("debug2.csv", comment = "#").to_numpy()
    
    train_coords = np.array(list(zip(train[:,1],train[:,2])))
    eval_coords = np.array(list(zip(eval[:,1],eval[:,2])))

    # Set and train the algorithm
    algo = QDA()
    algo.fit(train_coords,train[:,0])

    # Evaluate the algorithm
    print("Evaluation accuracy rate: ",1-algo.score(eval_coords,eval[:,0]))
    print("Training accuracy rate: ",1-algo.score(train_coords,train[:,0]))
def main():
    
    if len(sys.argv) == 1:
        hw_data()
    elif sys.argv[1] == "-debug":
        debug_data()
    else:
        print("Use Sklearn_QDA.py as:\n'python3 Sklearn_QDA.py -debug' or 'python3 Sklearn QDA.py'")


main()