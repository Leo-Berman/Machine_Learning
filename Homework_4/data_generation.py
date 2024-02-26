import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

def generate_1d_Gaussian_Mixture_pdf(data, components = 1,color = 'black'):
    GM_1 = mixture.GaussianMixture(components)
    GM_1.fit(data)
    weights = GM_1.weights_
    means = GM_1.means_
    covars = GM_1.covariances_
    plt.hist(data, color = 'gray', bins=30, density=True, alpha = 0.5)
    for i in range(len(means)):
        plt.plot(sorted(data),weights[i]*norm.pdf(sorted(data),means[i],np.sqrt(covars[i])), c=color)

def generate_2d_Gaussian_Mixture_pdf(data, components = 1,color = 'black'):
    GM_1 = mixture.GaussianMixture(components)
    GM_1.fit(data)
    weights = GM_1.weights_
    means = GM_1.means_
    covars = GM_1.covariances_
    x = np.linspace(-2, 2)
    y = np.linspace(-2, 2)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -GM_1.score_samples(XX)
    Z = Z.reshape(X.shape)
    CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
    plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(data[:, 0], data[:, 1], 0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    
    
def generate_Gaussian_Mixture_logprob(data, components = 1, color = 'black'):
    GM_1 = mixture.GaussianMixture(components)
    GM_1.fit(data)
    data = np.array(data[0]).reshape(-1,1)
    likelihood = -(GM_1.score(data))
    return likelihood

def q1(total_points):
    generate_1d_Gaussian_Mixture_pdf(total_points,1,'red')
    plt.savefig("q1.png")
    plt.cla()

def q2(total_points):
    # 1 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,1,'red')
    # 2 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,2,'green')
    # 3 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,3,'blue')
    plt.savefig("q2.png")
    plt.cla()

def q3(total_points):
    # 1 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,1,'red')
    # 2 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,2,'green')
    # 4 mixture
    generate_1d_Gaussian_Mixture_pdf(total_points,4,'yellow')
    plt.savefig("q3.png")
    plt.cla()

def q4(total_points):
    logprobs = []
    print(total_points)
    for i in range(10):
        logprobs.append(generate_Gaussian_Mixture_logprob(total_points,i+1))
    print(logprobs)
    x = np.linspace(1,10,10)
    plt.scatter(x,logprobs)
    plt.savefig("q4.png")

def separate_classes(data):
    dogs = []
    cats = []
    for x in data:
        if x[0] == "dogs":
            dogs.append([x[1],x[2]])
        else:
            cats.append([x[1],x[2]])
    return np.array(dogs).reshape(len(dogs),2),np.array(cats).reshape(len(cats),2)

def q5():
    # read in the data from the csv files
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    
    traindogs,traincats = separate_classes(train)
    evaldogs,evalcats = separate_classes(eval)
    
    dogslogprobs = []
    catslogprobs = []
    for i in range(10):
        dogslogprobs.append(generate_Gaussian_Mixture_logprob(traindogs,i+1))
        catslogprobs.append(generate_Gaussian_Mixture_logprob(traincats,i+1))
    x = np.linspace(1,10,10)
    plt.scatter(x,dogslogprobs,label="dogs")
    plt.scatter(x,catslogprobs,label="cats")
    plt.legend(["dogs","cats"])
    plt.savefig("q5.png")

    
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


    q1(total_points)
    q2(total_points)
    q3(total_points)
    q4(total_points)
    plt.cla()
    # q5()
    
main()
