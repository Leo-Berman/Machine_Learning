
import numpy as np
import pandas as pd
import sys
class Feature:

    # initialize values
    def __init__(self,inname):
        self.class_name =       inname
        self.number_elements =  0
        self.x_mean =           None
        self.y_mean =           None
        self.x_sd =             None
        self.y_sd =             None
        self.data =             np.array([])
        self.cov =              None
        self.cov_mat =          None
        self.x_var =            None
        self.y_var =            None
        self.mean_vec =         None
    
    # printing self
    def print_self(self):
        print(self.class_name,":")
        print("Number of elements = ", self.number_elements)
        print("Mean Vector = ",self.mean_vec)
        print("Covariance = ",self.cov)
        print("Covariance Matrix = \n",self.cov_mat)


    # calculating the standard deviation
    def calculate_sd(self):
        x_sums_squared = 0
        y_sums_squared = 0
        for x in self.data:
            x_sums_squared += (-self.x_mean + x[0])**2
            y_sums_squared += (-self.y_mean + x[1])**2
        self.x_sd = np.sqrt(x_sums_squared/(self.number_elements-1))
        self.y_sd = np.sqrt(y_sums_squared/(self.number_elements-1))

    # Add in feature data
    def add_data(self,indata):
        #print(indata)
        if len(self.data) == 0:
            self.data = indata
        else:
            self.data=np.vstack([self.data, indata])
        #print(self.data)
        self.number_elements+=1

    # Calculating the mean of each feature of the class
    def calculate_mean(self):
        x_sum = 0
        y_sum = 0
        for x in self.data:
            x_sum+=x[0]
            y_sum+=x[1]
        self.x_mean = x_sum/self.number_elements
        self.y_mean = y_sum/self.number_elements

    def calculate_mean_vector(self):
        self.calculate_mean()
        self.mean_vec = np.array([self.x_mean,self.y_mean])

    # Calculating covariance
    def calculate_covariance(self):
        work = 0
        for x in self.data:
            work+=(x[0] - self.x_mean) * (x[1] - self.y_mean)
        self.cov = work / (self.number_elements-1)

    # Calculating covariance matrix
    def calculate_covariance_matrix(self):
        self.calculate_covariance()
        self.calculate_variance()
        self.cov_mat = np.array([[self.x_var, self.cov],[self.cov, self.y_var]])

        #print("Hand calc = \n",self.cov_mat)
        #print(self.data[:,0])  
        #print("Numpy = \n",np.cov(self.data[:,0],self.data[:,1]))
    # Calculating population variance
    def calculate_variance(self):
        x_sum = 0
        y_sum = 0
        for x in self.data:
            x_sum+=(x[0]-self.x_mean)**2
            y_sum+=(x[1]-self.y_mean)**2

        self.x_var = x_sum/(self.number_elements-1)
        self.y_var = y_sum/(self.number_elements-1)

class Custom_Gaussian:
    
    # constructor
    def __init__(self):
        # keep track of classes
        self.classes = {}

        # keep track of number of elements
        self.number_elements = 0

        # Keeping track of totals
        self.value_totals = {}

    # training the model
    def train(self,training_data):
        
        # iterate through all data
        for i in range(len(training_data)):
            
            # check if there is already a feature for the class if not make one
            if training_data[i][0] not in self.classes:
                #print(training_data[i][0])

                # total values and make new features
                self.classes[training_data[i][0]]=Feature(training_data[i][0])
            
            # Add data to feature
            self.classes[training_data[i][0]].add_data(np.array([float(training_data[i][1]),float(training_data[i][2])]))

            # Keep track of total number of elements
            self.number_elements+=1

        # Call functions for calculating means and standard deviations
        for x in self.classes:
            self.classes[x].calculate_mean_vector()
            self.classes[x].calculate_covariance_matrix()

    # printing data from the class
    def print_classes(self):
        for x in self.classes:
            self.classes[x].print_self()
            print("")

    # calculate probabilities for individual classes
    def return_probability(self,prior,inclass,inx,iny):
        guess = None
        guess_prob = -1*float("inf")
        # Iterate through all classes and calculate each features probability density
        for x in self.classes:
            
            # Mean vector            
            mean_vector = self.classes[x].mean_vec

            # Feature vector
            feature_vector = np.array([inx,iny])

            # Covariance matrix
            covariance_matrix = self.classes[x].cov_mat
            
            # Grab inverse matrix calculate once
            #
            # Print for debugging
            #print(x,"mean vector = \n",mean_vector,"\n")
            #print("feature vector = \n",feature_vector,"\n")
            #print(x,"covariance matrix = \n",covariance_matrix,"\n")

            #print("DEBUG = \n",np.dot(feature_vector-mean_vector,np.linalg.inv(covariance_matrix)))
            # Elements for calculating probability density
            ele1_1 = np.matmul(np.transpose(feature_vector-mean_vector),np.linalg.inv(covariance_matrix))
            ele1_2 = feature_vector-mean_vector
            ele1 = -.5 * np.matmul(ele1_1,ele1_2)
            #print("Element 1 = \n",ele1,"\n")
            ele2 = -.5 * np.log(2*np.pi)
            #print("Element 2 = \n",ele2,"\n")
            ele3 = -.5 * (np.log(np.linalg.det(covariance_matrix)))
            #print("Element 3 = \n",ele3,"\n")
            if x == "dogs":
                #print("dogs")
                ele4 = prior
            else:
                #print("cats")
                ele4 = 1-prior
            #ele4 = np.log(.5)
            #print("Element 4 = \n",ele4,"\n")

            # Calculate probability density
            total_prob = ele1 + ele2 + ele3 + ele4
            #print("score = \n",total_prob,"\n")

            # If probability is greater than last guess set as new guess
            if total_prob > guess_prob:
                guess_prob = total_prob
                guess = x
        
        # Check to see if guess was correct or incorrect
        if guess == inclass:
            return True
        else:
            return False
    
    # Evaluate data
    def eval(self,prior,newdata):
        
        # Keep track of guesses and return the correct/total
        total_correct = 0
        total = 0

        for x in newdata:
            total +=1
            # Call the return p robability function which guesses and returns True 
            # When correct and False when not
            if self.return_probability(prior,x[0],float(x[1]),float(x[2])) == True:
                total_correct += 1
            #print(prior)
            #print(total)
            
            
        accuracy_rate = total_correct/total
        return accuracy_rate

def hw_data():
    # read data in and turn it into a 3 column numpy array
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    train = np.array(list(zip(train[:,0],train[:,1],train[:,2])))
    eval = np.array(list(zip(eval[:,0],eval[:,1],eval[:,2])))
    print(eval)
    # initialize model
    my_gauss = Custom_Gaussian()
    my_gauss.train(train)
    #my_gauss.train(eval)
    my_gauss.print_classes()
    #print("Test Case = ",my_gauss.eval([["dogs",-50,-50]]))
    prior_values = []
    evaluation_scores = []
    for x in range(100):
        evaluation_scores.append(1-my_gauss.eval(1,eval))
        prior_values.append(x/100)
    d = {
        "priors":prior_values,
        "eval error rates":evaluation_scores,
    }
    d = pd.DataFrame(data = d)
    d.to_csv('PythonQDA_WrongPriors.csv',sep =',',index = False,encoding='utf-8')

def debug_data():
    # read data in and turn it into a 3 column numpy array
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval_joe.csv", comment = "#").to_numpy()
    train = np.array(list(zip(train[:,0],train[:,1],train[:,2])))
    eval = np.array(list(zip(eval[:,0],eval[:,1],eval[:,2])))

    #print(train)
    # initialize model
    my_gauss = Custom_Gaussian()
    my_gauss.train(train)
    #my_gauss.train(eval)
    my_gauss.print_classes()
    #print("Test Case = ",my_gauss.eval([["1",-1,-1]]))
    
    print("Evaluation accuracy rate = ", 1-my_gauss.eval(eval))
    print("Training accuracy rate = ", 1-my_gauss.eval(train))
def main():
    if len(sys.argv) == 1:
        hw_data()
    elif sys.argv[1] == "-debug":
        debug_data()
    else:
        print("Use Sklearn_QDA.py as:\n'python3 Sklearn_QDA.py -debug' or 'python3 Sklearn QDA.py'")
    


main()