import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sys
import os
from scipy.stats import norm

def plot_histogram(data,name):
    # Plotting a basic histogram
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')
    
    # Display the plot
    #plt.savefig(name+".png")
    #print("saved")
def generate_data(means,variances,nelems):
    ret = []
    [ret.append(np.random.normal(loc = x,scale = y,size = nelems)) for x,y in zip(means,variances)]
    return ret

def main():
    # number of elements, means and variances
    nelems = 10000
    means = [-2.0,0.0,2.0]
    variances = [1.0,0.5,1.0]
    
    # generate data vectors
    vectors = generate_data(means,variances,nelems)

    # concatentae all vectors into 30,000 element list
    total_points = []
    [total_points.extend(x) for x in vectors]
    print(len(total_points))

    # calculate the pdf
    pdf = norm.pdf(total_points, np.mean(total_points), np.mean(variances))
    plot_histogram(total_points,"q1")
    plt.plot(total_points, pdf, label='Distribuzione Normale')
    plt.title('PDF of the Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    

main()
