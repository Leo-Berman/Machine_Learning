import numpy as np
def main():
    # declare means for each class
    mean_1 = [1,1]
    mean_2 = [1,-1]
    mean_3 = [-1,-1]
    mean_4 = [-1,1]

    # declare covariance matrix
    cov = [[0.1, 0],[0,0.1]]

    # generate the gaussians
    np.random.Generator.multivariate_normal(mean_1,cov)
main()