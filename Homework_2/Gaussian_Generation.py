import numpy
import pandas
import matplotlib.pyplot as plt
# generate multivariate guassians
def GMG(mean,cov,nelem):
    return numpy.random.multivariate_normal(mean,cov,nelem)

# calculate eigenvectors
def CE(inmat):
    return numpy.linalg.eig(inmat)

def main():
    # number of elements
    number_elements = 5000

   # declare mean for all covariance matrixes
    mean = [0,0]

    # declare covariance matrixes
    cov_0 = [[1,0],[0,1]]
    cov_1 = [[5,0],[0,2]]
    cov_2 = [[2,0],[0,5]]
    cov_3 = [[1,.5],[.5,1]]
    cov_4 = [[1,-.5],[-.5,1]]
    cov_5 = [[5,.5],[.5,2]]
    cov_6 = [[5,-.5],[-.5,2]]

    # concatenate all covariance matrixes into a list
    cov_list = [cov_0,cov_1,cov_2,cov_3,cov_4,cov_5,cov_6]

    # declare list for points to go into
    points_list = []

    # declare list of eigenvectors and eIgenvalues
    eigenvectors_list = []
    eigenvalues_list = []

    # Colors list
    colors_list = ['r','b','y','g','o']

    # Generate guassians for allcovariance matrixes
    for i in range(len(cov_list)):
        # Generate Guassian points
        points_list.append(GMG(mean,cov_list[i],number_elements))
        
        # Plot the points
        plt.plot(points_list[i][:,0], points_list[i][:,1], '.', alpha = 0.5, zorder = 0)
        
        # Calculate covariance matrix eigenvectors and eigenvalues
        eigenvalues,eigenvectors = CE(cov_list[i])
        eigenvectors_list.append(eigenvectors)
        eigenvalues_list.append(eigenvalues)

        print("Eigenvalues = ",eigenvalues)
        # Plot the eigenvectors
        for j in range(len(eigenvectors)):
            print(eigenvalues[j])     
            plt.quiver(*mean, *(eigenvectors[:,j]*eigenvalues[j]), scale = 18, color = colors_list[j], zorder = 10)
        
        
        #plt.axis('equal')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.grid()
        #plt.show()
        plotname = "Cov_" + str(i) + ".png"
        plt.savefig(plotname)
        plt.clf()
        plt.cla()
       
        
    '''
    cov_1_col1 = cov_1_class_1[:,0].tolist() + cov_1_class_2[:,0].tolist() + cov_1_class_3[:,0].tolist() + cov_1_class_4[:,0].tolist()
    cov_1_col2 = cov_1_class_1[:,1].tolist() + cov_1_class_2[:,1].tolist() + cov_1_class_3[:,1].tolist() + cov_1_class_4[:,1].tolist()
    cov_2_col1 = cov_2_class_1[:,0].tolist() + cov_2_class_2[:,0].tolist() + cov_2_class_3[:,0].tolist() + cov_2_class_4[:,0].tolist()
    cov_2_col2 = cov_2_class_1[:,1].tolist() + cov_2_class_2[:,1].tolist() + cov_2_class_3[:,1].tolist() + cov_2_class_4[:,1].tolist()
    
    plt.plot(cov_1_class_1[:, 0], cov_1_class_1[:, 1], '.', alpha=0.5)
    plt.plot(cov_1_class_2[:, 0], cov_1_class_2[:, 1], '.', alpha=0.5)
    plt.plot(cov_1_class_3[:, 0], cov_1_class_3[:, 1], '.', alpha=0.5)
    plt.plot(cov_1_class_4[:, 0], cov_1_class_4[:, 1], '.', alpha=0.5)
    plt.axis('equal')
    plt.grid()
    plt.show()
    
    
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
    df_1.to_csv('cov_1.csv' ,sep = ',',index=False, encoding='utf-8')
    df_2.to_csv('cov_2.csv' ,sep = ',',index=False, encoding='utf-8')

    debug_csv()'''
main()
