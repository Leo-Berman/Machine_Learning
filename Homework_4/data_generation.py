import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sys
import os

def plot_histogram(data,name):
    # Plotting a basic histogram
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')
    
    # Display the plot
    plt.savefig(name+".png")
    
def generate_data(means,variances,nelems):
    ret = []
    [ret.append(np.random.normal(loc = x,scale = y,size = nelems)) for x,y in zip(means,variances)]
    return ret

def main():
    # number of elements
    nelems = 10000

    # declare means and variances
    means = [-2.0,0.0,2.0]
    variances = [1.0,0.5,1.0]
    
    # generate data vectors
    vectors = generate_data(means,variances,nelems)

    

main()
