#!/usr/bin/env python

import matplotlib.pyplot as plt
import  numpy as np
TESTDAT = "/data/isip/data/tnmg_code/v1.0.0/data/S2910000/2919/2919827/TNMG2919827_N1.dat"
TESTHEA = "/data/isip/data/tnmg_code/v1.0.0/data/S2910000/2919/2919827/TNMG2919827_N1.hea"

def data_read(inpath):
    myints = np.fromfile(inpath, dtype = np.int16)
    feature_vectors = np.array([myints[0],myints[1],myints[2],myints[3],myints[4],myints[5],myints[6],myints[7]])
    for i in range(8,len(myints),8):
        applist = []
        for j in range(8):
            applist.append(myints[i+j])

        feature_vectors = np.vstack([feature_vectors,applist])
    return feature_vectors

def attach_labels(inpath,feature_vectors):
    pass
def main():
    features = data_read(TESTDAT)
    print(features)

if __name__ == "__main__":
    main()
