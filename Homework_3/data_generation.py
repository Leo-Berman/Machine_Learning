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

def main():
    # number of elements
    nelem = 10**6
    
    # generate the sets
    points = []
    for i in range(11):
        points.append(numpy.random.normal(loc = .9+(i*.02),scale = 1,size = nelem))
    
    
    mean100_plot_points = []
    mean100_sum= 0

    
    # Calculate mean using MLE
    for i,x in enumerate(points[5]):
        mean100_sum+=x
        mean100_plot_points.append(mean100_sum/(i+1))
    x_axis = numpy.linspace(1,10**6,10**6)
    
    fig,ax = plt.subplots()
    ax.plot(x_axis,mean100_plot_points)
    plt.xscale("log")
    plt.ylim(-3,4)
    

    
    index = 1
    for i in range(12):
        
        pltstr = '('+"{:.0e}".format(round(x_axis[index-1],0))+','+str(round(mean100_plot_points[index-1],2))+')'
        if (i%2) == 0:
            ax.annotate(pltstr,xy=(x_axis[index-1],mean100_plot_points[index-1]),xytext=(x_axis[index-1],-2.5+(i*.15)),arrowprops=dict(facecolor='black', shrink=.05))
            index*=5
        else:
            ax.annotate(pltstr,xy=(x_axis[index-1],mean100_plot_points[index-1]),xytext=(x_axis[index-1],3.5-(i*.15)),arrowprops=dict(facecolor='black', shrink=.2))
            index*=2
    plt.show()
        


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