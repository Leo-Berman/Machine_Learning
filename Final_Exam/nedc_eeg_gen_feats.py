#!/usr/bin/env python

import matplotlib.pyplot as plt
import  numpy as np
import nedc_cmdl_parser
import nedc_file_tools
import nedc_mladp_fileio_tools as fileio_tools
import polars
from scipy.fftpack import dct as dct

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

def data_read_with_sum(inpath):
    myints = np.fromfile(inpath, dtype = np.int16)
    feature_vector = []
    for i in range(0,len(myints),8):
        mysum = 0
        for j in range(8):
            mysum+=myints[i+j]
        feature_vector.append(mysum)
    return feature_vector

def data_read_separate(inpath):
    feature_vectors = data_read(inpath)
    feature_vectors = np.transpose(feature_vectors)
    ret = []
    for x in feature_vectors:


        toext = dct(feature_vectors[0:])[0][0:10].tolist()

        ret.extend(toext)
    
    return ret
    

def main():
    args_usage = 'nedc_eeg_gen_feats.usage'
    args_help = 'nedc_eeg_gen_feats.help'

    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters                                                                              
    #                                                                                               
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    
    
    feature_list = parsed_parameters['feats']
    label_list = parsed_parameters['labels']
    output_file = parsed_parameters['output_file']
    feat_files = fileio_tools.read_file_lists(feature_list)
    labels = fileio_tools.read_file_lists(label_list)
    labels = [x.replace(',','') for x in labels]
    print("Feature list = ",feature_list)
    print("Label list = ",label_list)
    feats = []
    for i,x in enumerate(feat_files):
        feats.append(data_read_separate(x))
        print(str(i+1)+" files processed")

        
    towrite = []
    for x,y in list(zip(labels,feats)):
        app = [x]
        app.extend(y)
        towrite.append(app)

        
    my_schema = []
    
    for i in range(len(towrite[0])):
        if i == 0:
            my_schema.append('label')
        else:
            my_schema.append(str(i-1))

    df = polars.DataFrame(towrite,schema=my_schema)
    df.write_csv(output_file)
if __name__ == "__main__":
    main()
