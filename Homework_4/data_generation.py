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
    

def generate_data(means,variances,nelems):
    ret = []
    [ret.append(np.random.normal(loc = x,scale = y,size = nelems)) for x,y in zip(means,variances)]
    return ret

def generate_Gaussian_Mixture(data, components = 1,color = 'black'):
    GM_1 = mixture.GaussianMixture(components)
    GM_1.fit(data)
    weights = GM_1.weights_
    means = GM_1.means_
    covars = GM_1.covariances_
    plt.hist(data, color = 'gray', bins=30, density=True, alpha = 0.5)
    for i in range(len(means)):
        plt.plot(sorted(data),weights[i]*norm.pdf(sorted(data),means[i],np.sqrt(covars[i])), c=color)
    

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
    total_points = np.array(total_points).reshape(-1,1)

    #Q1
    #plot_pdf(total_points,"q1")
    #plt.cla()

    # single mixture
    

    generate_Gaussian_Mixture(total_points,1,'red')
    generate_Gaussian_Mixture(total_points,2,'green')
    generate_Gaussian_Mixture(total_points,3,'blue')
    generate_Gaussian_Mixture(total_points,4,'yellow')
    plt.show()
main()
