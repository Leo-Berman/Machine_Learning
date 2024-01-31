
import numpy as np
import pandas as pd
class Feature:

    # initialize values
    def __init__(self,inname):
        self.class_name = inname
        self.number_elements = 0
        self.x_mean = None
        self.y_mean = None
        self.x_sd = None
        self.y_sd = None
        self.data = []
        self.cov = 0
        self.cov_mat = []
    
    # printing self
    def print_self(self):
        print(self.class_name,":")
        print("Number of elements = ", self.number_elements)
        print("X mean = ",self.x_mean)
        print("X Standard Deviation = ",self.x_sd)
        print("Y mean = ",self.y_mean)
        print("Y Standard Deviation = ",self.y_sd)
        print("Covariance = ",self.cov)
        print("Covraiance Matrix = \n",self.cov_mat)


    # calculating the standard deviation
    def calculate_sd(self):
        x_sums_squared = 0
        y_sums_squared = 0
        for x in self.data:
            x_sums_squared += (self.x_mean - x[0])**2
            y_sums_squared += (self.y_mean - x[1])**2
        self.x_sd = np.sqrt(x_sums_squared/(self.number_elements-1))
        self.y_sd = np.sqrt(y_sums_squared/(self.number_elements-1))

    # Add in feature data
    def add_data(self,indata):
        self.data.append(indata)
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
        
    # Calculating covariance
    def calculate_covariance(self):
        work = 0
        for x in self.data:
            work+=(x[0]-self.x_mean) * (x[1]-self.y_mean)
        self.cov = work/(self.number_elements-1)

    # Calculating covariance matrix
    def calculate_covariance_matrix(self):
        self.cov_mat = np.array([[self.x_sd, self.cov],[self.cov, self.y_sd]])

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
            self.classes[training_data[i][0]].add_data([float(training_data[i][1]),float(training_data[i][2])])

            # Keep track of total number of elements
            self.number_elements+=1

        # Call functions for calculating means and standard deviations
        for x in self.classes:
            self.classes[x].calculate_mean()
            self.classes[x].calculate_sd()
            self.classes[x].calculate_covariance()
            self.classes[x].calculate_covariance_matrix()

    # printing data from the class
    def print_classes(self):
        for x in self.classes:
            self.classes[x].print_self()
            print("")

    # calculate probabilities for individual classes
    def return_probability(self,inclass,inx,iny):
        guess_elements = -1
        guess = None

        # Iterate through the data and whichever one has more elements is my
        # default guess
        for x in self.classes:
            if self.classes[x].number_elements > guess_elements:
                guess_elements = self.classes[x].number_elements
                guess = x
                guess_prob = -1
        
        # Iterate through all classes and calculate each features probability density
        for x in self.classes:

            # Mean vector            
            x_mean = self.classes[x].x_mean
            y_mean = self.classes[x].y_mean
            mean_vector = np.array([x_mean,y_mean])

            # Feature vector
            feature_vector = np.array([inx,iny])

            # Covariance matrix
            covariance_matrix = self.classes[x].cov_mat
            
            # Print for debugging
            #print("mean vector = \n",mean_vector,"\n")
            #print("feature vector = \n",feature_vector,"\n")
            #print("covariance matrix = \n",covariance_matrix,"\n")

            #print("DEBUG = \n",np.dot(feature_vector-mean_vector,np.linalg.inv(covariance_matrix)))
            # Elements for calculating probability density
            ele1 = -1*.5*np.dot(np.dot(np.transpose(feature_vector-mean_vector),np.linalg.inv(covariance_matrix)),(feature_vector-mean_vector))
            #print("Element 1 = \n",ele1,"\n")
            ele2 = -1*.5*np.log(2*np.pi)
            #print("Element 2 = \n",ele2,"\n")
            ele3 = -1*.5*np.linalg.det(np.log(covariance_matrix))
            #print("Element 3 = \n",ele3,"\n")
            ele4 = -1*.5*np.log(1/self.number_elements)
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
    def eval(self,newdata):
        
        # Keep track of guesses and return the correct/total
        total_correct = 0
        total_wrong = 0
        for x in newdata:
            if (self.return_probability(x[0],float(x[1]),float(x[2]))) == True:
                total_correct += 1
            else:
                total_wrong += 1
        accuracy_rate = total_correct/self.number_elements
        return accuracy_rate

def main():
    
    # read data in and turn it into a 3 column numpy array
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    train = np.array(list(zip(train[:,0],train[:,1],train[:,2])))
    eval = np.array(list(zip(eval[:,0],eval[:,1],eval[:,2])))

    # initialize model
    my_gauss = Custom_Gaussian()
    my_gauss.train(train)
    my_gauss.train(eval)
    my_gauss.print_classes()
    my_gauss.eval([["dogs",0.002100,-0.434914]])
    
    print("Evaluation data accuracy = ", 1-my_gauss.eval(eval))
    print("Training data accuracy = ", 1-my_gauss.eval(train))
    


main()