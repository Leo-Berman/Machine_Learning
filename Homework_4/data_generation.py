import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
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
    

def plot_pdf(data,name):

    # sort the data
    data.sort()

    # calculate the pdf
    mu,sigma = norm.fit(data)
    pdf = norm.pdf(data, loc=mu, scale=sigma)

    # plot normalized histogram of the data
    plt.hist(data,bins = 60, density = True)

    # plot the pdf of the data
    plt.plot(data, pdf, label='Normal Distribution')
    plt.title('PDF of the Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()

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


    #plot_pdf(total_points,"q1")
    #plt.cla()

    # single mixture
    GM_1 = mixture.GaussianMixture(1)
    print(GM_1.fit(total_points))
    

main()
