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
import torch
import torchvision
import torchvision.transforms as transforms
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
    args_usage = "nedc_eeg_train.usage"
    args_help = "nedc_eeg_train.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train")
    model_type=parsed_parameters['model_type']
    data_file=parsed_parameters['data_file']
    model_output_path=parsed_parameters['model_output_path']
    compression=int(parsed_parameters['compression'])

    
    mydata,labels = read_data(data_file)


    
    # Fit the model
    #
    model = None
    if model_type == "RNF":
        model = RNF()
        model.fit(mydata, labels)
        print("Trained on ",len(labels),"features")
        print("Score on training data = ",model.score(mydata,labels))
    
        # dump the model there
        #
        joblib.dump(model,model_output_path,compress=compression)
    elif model_type = "CNN":
        pass
    else:
        print("No model supplied")
        return

    

if __name__ == "__main__":
    main()
