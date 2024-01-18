import numpy
import pandas
def Gen_2d_Gaussian(mean,cov,n_elem):
        return numpy.random.multivariate_normal(mean,cov,n_elem)

def main():
    # declare means for each class
    mean_1 = [1,1]
    mean_2 = [1,-1]
    mean_3 = [-1,-1]
    mean_4 = [-1,1]

    # declare covariance matrix
    cov = [[0.1, 0],[0,0.1]]

    # generate the gaussians
    class_1 = numpy.array(Gen_2d_Gaussian(mean_1,cov,10000).tolist()).reshape((2,10000))
    print(class_1[0])
    
    
    d = {'col0':["1"]*10000, 'col1':class_1[0], 'col2':class_1[1]}
    df = pandas.DataFrame(data=d)
    df.to_csv('/home/leo/Desktop/Github_Repos/Machine_Learning/Homework_1/class1.csv' ,sep = ',',index=False, encoding='utf-8')
    
main()