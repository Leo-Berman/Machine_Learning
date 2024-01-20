import numpy
import pandas
def Gen_2d_Gaussian(mean,cov,n_elem):
        return numpy.random.multivariate_normal(mean,cov,n_elem)

def main():
    # number of elements
    number_elements = 100

    # declare means for each class
    mean_1 = [1,1]
    mean_2 = [1,-1]
    mean_3 = [-1,-1]
    mean_4 = [-1,1]

    # declare covariance matrixes
    cov_1 = [[0.1, 0],[0,0.1]]
    cov_2 = [[1, 0],[0,1]]

    # generate the gaussians
    cov_1_class_1 = numpy.array(Gen_2d_Gaussian(mean_1,cov_1,number_elements).tolist()).reshape((2,number_elements))
    cov_1_class_2 = numpy.array(Gen_2d_Gaussian(mean_2,cov_1,number_elements).tolist()).reshape((2,number_elements))
    cov_1_class_3 = numpy.array(Gen_2d_Gaussian(mean_3,cov_1,number_elements).tolist()).reshape((2,number_elements))
    cov_1_class_4 = numpy.array(Gen_2d_Gaussian(mean_4,cov_1,number_elements).tolist()).reshape((2,number_elements))
    
    cov_2_class_1 = numpy.array(Gen_2d_Gaussian(mean_1,cov_2,number_elements).tolist()).reshape((2,number_elements))
    cov_2_class_2 = numpy.array(Gen_2d_Gaussian(mean_2,cov_2,number_elements).tolist()).reshape((2,number_elements))
    cov_2_class_3 = numpy.array(Gen_2d_Gaussian(mean_3,cov_2,number_elements).tolist()).reshape((2,number_elements))
    cov_2_class_4 = numpy.array(Gen_2d_Gaussian(mean_4,cov_2,number_elements).tolist()).reshape((2,number_elements))
    
    classes = ["1"]*number_elements+["2"]*number_elements+["3"]*number_elements+["4"]*number_elements
    cov_1_col1 = numpy.concatenate((cov_1_class_1[0], cov_1_class_2[0], cov_1_class_3[0], cov_1_class_4[0]))
    cov_1_col2 = numpy.concatenate((cov_1_class_1[1], cov_1_class_2[1], cov_1_class_3[1], cov_1_class_4[1]))
    cov_2_col1 = numpy.concatenate((cov_2_class_1[0], cov_2_class_2[0], cov_2_class_3[0], cov_2_class_4[0]))
    cov_2_col2 = numpy.concatenate((cov_2_class_1[1], cov_2_class_2[1], cov_2_class_3[1], cov_2_class_4[1]))


    d_1 = {
        'col0':classes, 
        'col1':cov_1_col1,
        'col2':cov_1_col2,

    }
    
    d_2 = {
        'col0':classes, 
        'col1':cov_2_col1,
        'col2':cov_2_col2,
    }

    df_1 = pandas.DataFrame(data=d_1)
    df_2 = pandas.DataFrame(data = d_2)
    df_1.to_csv('/home/leo/Desktop/Github_Repos/Machine_Learning/Homework_1/cov_1.csv' ,sep = ',',index=False, encoding='utf-8')
    df_2.to_csv('/home/leo/Desktop/Github_Repos/Machine_Learning/Homework_1/cov_2.csv' ,sep = ',',index=False, encoding='utf-8')
main()