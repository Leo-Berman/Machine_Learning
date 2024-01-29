
import numpy as np
import pandas as pd
class Feature:
    def __init__(self,inname):
        self.class_name = inname
        self.number_elements = 0
        self.x_mean = None
        self.y_mean = None
    def print_self(self):
        print(self.class_name,"has",self.number_elements,"elements and the mean is",(self.x_mean,self.y_mean))

class Custom_Gaussian:
    def __init__(self,indata):
        self.data = indata
        self.classes = {}
    def train(self):
        value_totals = {}

        for i in range(len(self.data)):
            if self.data[i][0] not in self.classes:
                self.classes[self.data[i][0]]=Feature(self.data[i][0])
                value_totals[self.data[i][0]+"_x"] = self.data[i][1]
                value_totals[self.data[i][0]+"_y"] = self.data[i][2]

            else:
                self.classes[self.data[i][0]].number_elements+=1
                value_totals[self.data[i][0]+"_x"] += self.data[i][1]
                value_totals[self.data[i][0]+"_y"] += self.data[i][2]
            
        for x in self.classes:
            print(value_totals[x+"_x"])
            print(self.classes[x].number_elements)
            #features[x].x_mean = value_totals[x+"_x"]/features[x].number_elements
            #features[x].y_mean = value_totals[x+"_y"]/features[x].number_elements


    def print_classes(self):
        for x in self.classes:
            self.classes[x].print_self()


def main():
    print("Hello World")
    train = pd.read_csv("train.csv",comment = "#").to_numpy()
    eval = pd.read_csv("eval.csv", comment = "#").to_numpy()
    
    train = np.array(list(zip(train[:,0],train[:,1],train[:,2])))
    eval = np.array(list(zip(eval[:,0],eval[:,1],eval[:,2])))

    my_gauss = Custom_Gaussian(train)
    my_gauss.train()
    my_gauss.print_classes()
    



main()