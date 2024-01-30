
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
    
    # printing self
    def print_self(self):
        print(self.class_name,":")
        print("Number of elements = ", self.number_elements)
        print("X mean = ",self.x_mean)
        print("X Standard Deviation = ",self.x_sd)
        print("Y mean = ",self.y_mean)
        print("Y Standard Deviation = ",self.y_sd)

    # calculating the standard deviation
    def calculate_sd(self):
        x_sums_squared = 0
        y_sums_squared = 0
        
        for x in self.data:
            x_sums_squared += (self.x_mean - x[0])**2
            y_sums_squared += (self.y_mean - x[1])**2
        self.x_sd = np.sqrt(x_sums_squared/(self.number_elements-1))
        self.y_sd = np.sqrt(y_sums_squared/(self.number_elements-1))

class Custom_Gaussian:
    # load data into a class
    def __init__(self,indata):
        self.data = indata
        self.classes = {}
        self.number_elements = 0
    
    # training the model
    def train(self):
        
        # Keeping track of totals
        value_totals = {}

        # iterate through all data
        for i in range(len(self.data)):
            
            # check if there is already a feature for the class if not make one
            if self.data[i][0] not in self.classes:
                #print(self.data[i][0])

                # total values and make new features
                self.classes[self.data[i][0]]=Feature(self.data[i][0])
                value_totals[self.data[i][0]+"_x"] = float(self.data[i][1])
                value_totals[self.data[i][0]+"_y"] = float(self.data[i][2])

            else:
                # add valuess to track means
                value_totals[self.data[i][0]+"_x"] += float(self.data[i][1])
                value_totals[self.data[i][0]+"_y"] += float(self.data[i][2])
            
            # Add data to class and keep track of number of elements
            self.classes[self.data[i][0]].number_elements+=1
            self.classes[self.data[i][0]].data.append([float(self.data[i][1]),float(self.data[i][2])])
            self.number_elements+=1

        # calculate means and standard deviations
        for x in self.classes:
            self.classes[x].x_mean = value_totals[x+"_x"]/self.classes[x].number_elements
            self.classes[x].y_mean = value_totals[x+"_y"]/self.classes[x].number_elements
            self.classes[x].calculate_sd()

    # printing data from the class (features)
    def print_classes(self):
        for x in self.classes:
            self.classes[x].print_self()
            print("")

    # calculate probabilities for individual classes
    def return_probability(self,inclass,inx,iny):
        guess_elements = -1
        guess = None
        for x in self.classes:
            if self.classes[x].number_elements > guess_elements:
                guess_elements = self.classes[x].number_elements
                guess = x
                guess_prob = -1
        
        for x in self.classes:
            x_mean = self.classes[x].x_mean
            y_mean = self.classes[x].y_mean
            x_sd = self.classes[x].x_sd
            y_sd = self.classes[x].y_sd
            x_num = np.exp(-((inx-x_mean)**2)/(2*(x_sd**2)))
            x_den = np.sqrt(2*np.pi*(x_sd**2))
            y_num = np.exp(-((iny-y_mean)**2)/(2*(y_sd**2)))
            y_den = np.sqrt(2*np.pi*(y_sd**2))
            x_prob = x_num/x_den
            y_prob = y_num/y_den
            #total_prob = (x_prob + y_prob)/2
            #total_prob = x_prob*y_prob
            #total_prob = np.convolve(x_prob,y_prob)
            #print(x,total_prob) # debug
            if total_prob > guess_prob:
                guess_prob = total_prob
                guess = x
        
        #print(guess,guess_prob) # debug
        if guess == inclass:
            return True
        else:
            return False
    
    def eval(self,newdata):
        total_correct = 0
        total_wrong = 0
        for x in newdata:
            if (self.return_probability(x[0],float(x[1]),float(x[2]))) == True:
                total_correct += 1
            else:
                total_wrong += 1
        accuracy_rate = total_correct/self.number_elements
        #print("Accuracy rate = ",accuracy_rate)
        return accuracy_rate

def main():
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    
    train = np.array(list(zip(train[:,0],train[:,1],train[:,2])))
    eval = np.array(list(zip(eval[:,0],eval[:,1],eval[:,2])))

    my_gauss = Custom_Gaussian(train)
    my_gauss.train()
    my_gauss.print_classes()
    #my_gauss.eval([["dogs",0.002100,-0.434914]])
    
    print("Evaluation data accuracy = ", 1-my_gauss.eval(eval))
    print("Training data accuracy = ", 1-my_gauss.eval(train))
    


main()