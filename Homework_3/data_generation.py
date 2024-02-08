import numpy
import pandas
import matplotlib.pyplot as plt
import sys
def plot_histogram(data,name):
    # Plotting a basic histogram
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')
    
    # Display the plot
    plt.savefig(name+".png")
def cascading_arrow(data,ylower=-2,yhigher=2):
    
    # generate x_axis
    x_axis = numpy.linspace(1,10**6,10**6)
    
    # create subplots
    fig,ax = plt.subplots()

    # plot the data
    ax.plot(x_axis,data)

    # set the x-axis to a logarithmic scale
    plt.xscale("log")

    # set bounds
    plt.ylim(ylower,yhigher)
    plt.xlim(0,10**7)

    #iterate through 1,5,10,50,100,500...5*10^5,10**6
    index = 1
    for i in range(13):

        # scientific notation to be efficient with space
        if i > 10:
            pltstr = '('+"{:.0e}".format(round(x_axis[index-1],0))+','+str(round(data[index-1],2))+')'
        else:
            pltstr = '('+str(round(x_axis[index-1],0))+','+str(round(data[index-1],2))+')'
        
        # annotate stacked cascading arrows with text centered
        ha = 'center'
        if i == 0:
            xloc = x_axis[index-1]+2
        else:
            xloc = x_axis[index-1]

        if (i%2) == 0:
            ax.annotate(pltstr,xy=(x_axis[index-1],data[index-1]),xytext=(xloc,ylower+((yhigher-ylower)/14)+(i*((yhigher-ylower)/42))),arrowprops=dict(facecolor='black', shrink=.05), horizontalalignment=ha)
            index*=5
        else:
            ax.annotate(pltstr,xy=(x_axis[index-1],data[index-1]),xytext=(xloc,yhigher-((yhigher-ylower)/14)-(i*((yhigher-ylower)/42))),arrowprops=dict(facecolor='black', shrink=.2), horizontalalignment=ha)
            index*=2

def q2a(data):
    cascading_arrow(data,-1,3)    
    #plt.savefig("Q2a.png")
    plt.cla()
    print("Question 2a No. of Points = 110: ", data[110])
    

def q2b(data):
    
    # find the mean of the first 6 sets summed together
    mean_means_data = numpy.array(data[.90])
    for i in range(5):
        mean_means_data+=numpy.array(data[round(.92+(i*.02),2)])
    mean_means_data/=6
    plot_data = []
    sum = 0

    # find the MLE of the mean of the first 6 sets summed together
    for i,x in enumerate(mean_means_data):
        sum+=x
        plot_data.append(sum/(i+1))
    
    # plot the points
    cascading_arrow(plot_data,.5,1.5)

    plt.savefig('Q2b.png')
    plt.cla()
    print("Question 2b No. of Points = 110: ", plot_data[int(110/6)])
    
def q3a(data):
    # find the mean of the all the sets summed together
    mean_means_data = numpy.array(data[.90])
    for i in range(10):
        mean_means_data+=numpy.array(data[round(.92+(i*.02),2)])
    mean_means_data/=11
    plot_data = []
    sum = 0
    
    # find the MLE of the mean of the first 11 sets summed together
    for i,x in enumerate(mean_means_data):
        sum+=x
        plot_data.append(sum/(i+1))
    
    # plot the points
    cascading_arrow(plot_data,.5,1.5)

    #plt.savefig('Q3a.png')
    plt.cla()
    print("Questoin 3a No. of Points = 110: ",plot_data[10])

def q4a(data):
    #    print(data)
    guess = 2
    variance = 1
    plot_data =[]

    # find the mean of the all the sets summed together
    mean_means_data = numpy.array(data[.90])
    for i in range(10):
        mean_means_data+=numpy.array(data[round(.92+(i*.02),2)])
    mean_means_data/=11
    plot_data = []
    mysum = 0
    

    # find the MLE of the mean of the first 11 sets summed together
    for j,x in enumerate(mean_means_data):
        i = j+1
        mysum+=x
        mean = mysum/(i)
        #prior=numpy.log(    (2*numpy.pi*variance**2)**-(.5)     *      numpy.exp(   -1*(mean-guess)**2/(2*variance**2)))
        tau_n=(i/1+1)**-1
        mu_n = tau_n* ( i * mean+2)
        posterior=numpy.log(    (2*numpy.pi*tau_n**2)**-(.5)     *      numpy.exp(   -1*(mean-mu_n)**2/(2*tau_n**2)))                        
        plot_data.append(posterior)
    cascading_arrow(plot_data,0,15)
    plt.show()
    
def figure_generation(data):
    # generate x_axis
    x_axis = numpy.linspace(1,10**6,10**6)
    
    # create subplots
    fig,ax = plt.subplots()

    for x in data:
        # plot the data
        ax.plot(x_axis,data[x])
    
    # set the x-axis to a logarithmic scale
    plt.xscale("log")
    #plt.show()
    plt.savefig('SeparateMeans.png')
    plt.xlim(10**5,10**6)
    plt.ylim(.8,1.2)
    plt.savefig('SeparateMeans_Zoomed.png')
def main():
    # number of elements
    nelem = 10**6
    
    # generate the points for each set
    vectors = []
    for i in range(11):
        vectors.append(numpy.random.normal(loc = .9+(i*.02),scale = 1,size = nelem))
    
    # create a dictionary to hold each set's means
    mean_plot_points = {}
    
    # iterate through all sets
    for i,x in enumerate(vectors):

        # keep track of the mean of that set
        mysum = 0

        # Add the set to the dictionary
        mean_plot_points[round(.90+(.02*i),2)] = []
        for j,y in enumerate(x):
            # keep track of the mean and append each mean point
            mysum+=y
            mean_plot_points[round(.90+(.02*i),2)].append(mysum/(j+1))
            

    
    #q2a(mean_plot_points[1.00])
    #q2b(mean_plot_points)
    #q3a(mean_plot_points)
    #figure_generation(mean_plot_points)    
    q4a(mean_plot_points)

    '''
    d_2 = {
        'col0':classes, 
        'col1':cov_2_col1,
        'col2':cov_2_col2,
        }
    df_1 = pandas.DataFrame(data=d_1)
    df_2 = pandas.DataFrame(data = d_2)
    df_1.to_csv('cov_1.csv' ,sep = ',',index=False, encoding='utf-8')
    df_2.to_csv('cov_2.csv' ,sep = ',',index=False, encoding='utf-8')'''
main()
