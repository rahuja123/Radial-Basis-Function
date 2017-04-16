import math
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
import kmeans
from scipy import *
from scipy.linalg import pinv

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def euclidian(c, d):
    sum=0
    for i in range(len(c)):
        square_diff = (c[i]- d[i])**2
        sum = sum + square_diff

    return math.sqrt(sum)


def rand(a,b):
    return (b-a)*random.random() + a

def loadcsv(filename):
    with open ( filename, "r" ) as csvfile:
        lines = csv.reader ( csvfile )
        dataset = list ( lines )
        for i in range ( len ( dataset ) ):
            dataset[ i ] = [ float ( x ) for x in dataset[ i ] ]
        return dataset


class RBF:
    def __init__(self, ni, no, num_center):

        self.ni = ni
        self.nh = num_center
        self.no = no


        #weight matrix
        # self.wi= makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh+1, self.no)

        #activation input matrix
        self.phi = []


        # for i in range(self.ni):
        #     for j in range(self.nh):
        #         self.wi[i][j] = rand(-1.0, 1.0)
        for j in range(self.nh+1): #one extra weight layer for bias
            for k in range(self.no):
                self.wo[j][k] = rand(-1.0, 1.0)


    def _basisfunc(self, data_point, centroid_point, gamma):
        euc_prod = euclidian (data_point, centroid_point)
        return math.exp ( -gamma * euc_prod ** 2 )

    def calc_activation(self, data):
        centroids, gammas = kmeans.kmeans ( data, self.nh )  # calling k means to get a list of centroids
        self.phi= makeMatrix(len(data), self.nh+1)
        i=0
        for instance in data:
            self.phi[i][0]=1
            j=1
            for i,point in enumerate(centroids):
                gamma= gammas[i]
                self.phi[i][j]= self._basisfunc(instance, point,gamma)
                j=j+1
            i= i+1



    def calc_weight(self, train_label):
        pseudo_inverse= pinv(self.phi)
        self.wo = dot(pseudo_inverse, train_label)
        #look through this one
        #will calculate the wieght by multiplying

    def get_labels(self, patterns):
        data_len = len ( patterns )
        num_classes = self.no

        labels = makeMatrix ( data_len, num_classes )
        for i, instance in enumerate ( patterns ):
            label_value = int ( instance[ -1 ][ 0 ] )
            labels[ i ][ label_value - 1 ] = 1.0

        return labels

    def differentiate(self,data):
        train_feature = [ ]
        train_label = self.get_labels(data)
        for instance in data:
            feature = instance[0]
            train_feature.append(feature)

        return train_feature, train_label

    def train(self, train_data):

        train_features, train_label= self.differentiate(train_data)
        self.calc_activation(train_features)
        self.calc_weight(train_label)
        print(self.wo)

    def test(self, test_data):
        total=0
        correct=0
        test_features, test_label = self.differentiate(test_data)
        self.calc_activation(test_features)
        answers= dot(self.phi, self.wo)
        for i, answer in enumerate(answers):
            total=total+1
            print(answer)
            print(test_label[i])
            if np.argmax(answer)==np.argmax(test_label[i]):
                correct=correct+1

        print("Accuracy:")
        print(correct/total *100)










def demo():
    filename = 'iris.csv'
    dataset = loadcsv ( filename )
    train, test = train_test_split ( dataset, test_size=0.3 )
    print (len( train ))
    print (len( test ))
    print(train)
    print(test)

    pat_train= []
    for i in range(len(train)):
        label=[]
        label.append(train[i][-1])
        print(label)
        features= train[i][:-1]
        print(features)
        temp=[]
        temp.append(features)
        temp.append(label)
        pat_train.append(temp)
    print(pat_train)

    pat_test=[]
    for i in range(len(test)):
        label=[]
        label.append(test[i][-1])
        features= test[i][:-1]
        temp=[]
        temp.append(features)
        temp.append(label)
        pat_test.append(temp)
    print(pat_test)



    input_neuron= 31
    output_neuron= 2
    num_center= 30
    print("initialized RBF")
    rbf= RBF(input_neuron, output_neuron, num_center)
    print("calling training")
    rbf.train(pat_train)
    rbf.test(pat_test)


if __name__ == '__main__':
    demo()