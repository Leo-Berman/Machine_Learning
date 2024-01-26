from sklearn.gaussian_process import GuassianProcessClassifier as GPC
import numpy as np
import pandas as pd

def main():

    # read in the data from the csv files
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    
    train_coords = np.array(list(zip(train[:,1],train[:,2])))
    eval_coords = np.array(list(zip(eval[:,1],eval[:,2])))

    # Set and train the algorithm
    algo = QDA()
    print(algo.fit(train_coords,train[:,0]))

    # Evaluate the algorithm
    print("Evaluation accuracy rate: ",1-algo.score(eval_coords,eval[:,0]))
    print("Training accuracy rate: ",1-algo.score(train_coords,train[:,0]))


main()