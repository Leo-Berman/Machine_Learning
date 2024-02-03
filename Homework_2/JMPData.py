import pandas as pd
import numpy as np
def main():
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()

    col0 = np.hstack((train[:,0],eval[:,0]))
    col1 = np.hstack((train[:,1],eval[:,1]))
    col2 = np.hstack((train[:,2],eval[:,2]))
    col3 = ["train"]*len(train) + ["eval"]*len(eval)
    print(col0)
    d = {
        "classes":col0,
        "x":col1,
        "y":col2,
        "Validation":col3
    }
    d = pd.DataFrame(data = d)
    d.to_csv('JMPData.csv',sep =',',index = False,encoding='utf-8')
main()