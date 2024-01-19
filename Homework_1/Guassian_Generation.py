import numpy
import pandas
def Gen_2d_Gaussian(mean,cov,n_elem):
        return numpy.random.multivariate_normal(mean,cov,n_elem)

def main():
    # number of elements
    number_elements = 1000

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
    
    
    d_1 = {
        'col0':["1"]*number_elements, 
        'col1':cov_1_class_1[0], 
        'col2':cov_1_class_1[1],

        'col3':["2"]*number_elements, 
        'col4':cov_1_class_2[0], 
        'col5':cov_1_class_2[1],

        'col6':["3"]*number_elements, 
        'col7':cov_1_class_3[0], 
        'col8':cov_1_class_3[1],
          
        'col9':["4"]*number_elements, 
        'col10':cov_1_class_4[0], 
        'col11':cov_1_class_4[1]
    }
    
    d_2 = {
        'col0':["1"]*number_elements, 
        'col1':cov_2_class_1[0], 
        'col2':cov_2_class_1[1],

        'col3':["2"]*number_elements, 
        'col4':cov_2_class_2[0], 
        'col5':cov_2_class_2[1],

        'col6':["3"]*number_elements, 
        'col7':cov_2_class_3[0], 
        'col8':cov_2_class_3[1],
          
        'col9':["4"]*number_elements, 
        'col10':cov_2_class_4[0], 
        'col11':cov_2_class_4[1]
    }

    df_1 = pandas.DataFrame(data=d_1)
    df_2 = pandas.DataFrame(data = d_2)
    df_1.to_csv('/home/leo/Desktop/Github_Repos/Machine_Learning/Homework_1/cov_1.csv' ,sep = ',',index=False, encoding='utf-8')
    df_2.to_csv('/home/leo/Desktop/Github_Repos/Machine_Learning/Homework_1/cov_2.csv' ,sep = ',',index=False, encoding='utf-8')
main()