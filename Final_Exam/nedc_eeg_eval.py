#!/usr/bin/env python
#
# file: /data/isip/exp/tuh_dpath/exp_0280/nedc_mladp/src/util/nedc_train_model/train_model.py
#
# revision history:
#
# 
#
# This is a Python version of the C++ utility nedc_print_header.
#------------------------------------------------------------------------------

"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
import os
import joblib
import polars

import nedc_mladp_fileio_tools as fileio_tools

#picone
import nedc_file_tools

def read_data(infile):
    df = polars.read_csv(infile,infer_schema_length=0).with_columns(polars.exclude("label").cast(float))
    labels = df.select("label").to_series().to_list()
    df = df.drop("label")
    rows = df.rows()

    feats = []
    for x in rows:
        feats.append(list(x))

    return feats,labels
def main():

    # set argument parsing
    #
    args_usage = "nedc_eeg_eval.usage"
    args_help = "nedc_eeg_eval.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval")
    data_file=parsed_parameters['data_file']
    modelpath=parsed_parameters['modelpath']

    
    mydata,labels = read_data(data_file)


    
    model=joblib.load(modelpath)

    print(modelpath + "score = ",model.score(mydata,labels))

    

if __name__ == "__main__":
    main()
