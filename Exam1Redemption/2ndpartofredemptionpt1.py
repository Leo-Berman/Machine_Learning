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
    min1 = -2
    max1 = 3
    min2 = -2
    max2 = 3
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
    for j in range(5):
        # lists for probability of error
        probability_error = []

        # list for xaxis
        xaxis = []

        # resolution
        iterations = 8
        # iterate through the resolutions
        for i in range(iterations):
            k = i+1
            # print(i)
            # generate the data
            class1,class2 = generate_data(0,1,-2+(i/(iterations/4)),-1+(i/(iterations/4)))

            # generate the labels
            labels = [0]*len(class1) + [1]*len(class1)

            
            # concatenate data
            data = class1.copy()
            data.extend(class2)

            # create a QDA model
            model = QDA(tol=1e-7)

            # set the priors
            model.priors=[1-(j*.25),j*.25]
                
            #
            weights = [model.priors[0]]*len(class1)+[model.priors[1]]*len(class2)
            # fit model to the data
            model.fit(data,labels)

            model.priors=[1-(j*.25),j*.25]
            

            # probability_error.append(1-model.score(data,labels,weights))


            probability_error.append(1-model.score(data,labels))
                                                   
            # append to the axis
            xaxis.append(-2 + i/(iterations/4))

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

            
            plt.title("Alpha = " + str(-2 + (k)/(iterations/4))+"P[w1] = "+str(1-(j*.25)))


            figurename = str(round(1-(j*.25),3))+"_"+str(k)+".png"
            # print(figurename)
            # save the plots as images
            plt.savefig(figurename)

            # clear plots
            plt.cla()

        # plot the probability of errors
        plt.plot(xaxis,probability_error)

        # label axis
        plt.xlabel("alpha")
        plt.ylabel("P[E]")

        # label the middle point
        plt.text(xaxis[iterations//2],probability_error[iterations//2],str(xaxis[iterations//2]) + "," + str(probability_error[iterations//2]))
        plt.text(xaxis[iterations//4*3],probability_error[iterations//4*3],str(xaxis[iterations//4*3]) + "," + str(probability_error[iterations//4*3]))
        # save the file
        plt.title("P[w1] = "+str(1-(j*.25)))
        plt.savefig("resultsprior1_"+str(1-(j*.25))+".png")
        plt.cla()


main()


