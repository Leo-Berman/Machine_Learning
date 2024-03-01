import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import os

# generate data points
def generate_data(x1=0,x2=1,y1=0,y2=1):

    # create lists to store values
    xpoints1 = []
    ypoints1 = []
    xpoints2 = []
    ypoints2 = []
    for i in range(5000):
        xpoints1.append(random.uniform(x1,x2))
        ypoints1.append(random.uniform(x1,x2))
        xpoints2.append(random.uniform(y1,y2))
        ypoints2.append(random.uniform(y1,y2))
    
    # return lists of lists
    return list(map(list, zip(xpoints1, ypoints1))),list(map(list, zip(xpoints2, ypoints2)))
    
def plot_decision_surface(model):
    min1 = -3
    max1 = 4
    min2 = -3
    max2 = 4
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class


def main():

    

    # remove frames from previous
    # os.system("rm this*.png")
    alphas = [-1,-.75,-.5,-.25,0,.25,.5,.75,1]
    
    
    for x in alphas:
        # append to the axis
        xaxis = []
        errors = []
        
        for i in range(101):
            class1,class2 = generate_data(0,1,x,1+x)

            # generate the labels
            labels = [0]*len(class1) + [1]*len(class1)

            
            # concatenate data
            data = class1.copy()
            data.extend(class2)

            # create a QDA model
            model = QDA(tol=1e-7)

            # set the priors
            model.priors=[i/100,1-i/100]
            xaxis.append(i/100)
            #
            # weights = [model.priors[0]]*len(class1)+[model.priors[1]]*len(class2)
            print(model.priors)
            # fit model to the data
            model.fit(data,labels)

            model.priors=[i/100,1-i/100]

            # errors.append(1-model.score(data,labels,weights))

            errors.append(1-model.score(data,labels))

        

            # create lists for each x and y for each data set
            # for plotting the scatter
            class1_x = []
            class1_y = []
            class2_x = []
            class2_y = []
            for i in range(len(class1)):
                class1_x.append(class1[i][0])
                class1_y.append(class1[i][1])
                class2_x.append(class2[i][0])
                class2_y.append(class2[i][1])


            plot_decision_surface(model)
            # plot the two datasets
            plt.scatter(class1_x,class1_y,color = "red")
            plt.scatter(class2_x,class2_y,color = "blue")

            
            plt.title("Alpha = " + str(x) + " P[w1] = "+str(model.priors[0]))


            figurename = "Alpha = " + str(x) + " P[w1] = "+str(model.priors[0])+".png"
            # print(figurename)
            # save the plots as images
            plt.savefig(figurename)

            # clear plots
            plt.cla()

        # plot the probability of errors
        plt.plot(xaxis,errors)

        # label axis
        plt.xlabel("P[w1]")
        plt.ylabel("P[E]")

        # label the middle point
        iterations = 100
        # save the file
        plt.title("Alpha = " + str(x))
        plt.ylim((0,1))
        plt.savefig("Plot_Alpha = " + str(x) + ".png")
        plt.cla()


main()


