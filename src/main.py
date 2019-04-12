# Program to train a perceptron.

# Imports:
from data_extractor import extract_sonar_data
from random import random
import numpy as np
import matplotlib.pyplot as plt

# Data extractor:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Weight initialization:
num_of_weights = len(train_features[0])

class FuzzyMatrix:
    def __init__(self):
        self.matrix = {}
    
    def add(self, predicted, expected):
        key = (predicted, expected)
        self.matrix[key] = self.matrix.get(key, 0) + 1
    
    def trace(self):
        matrixList = (self.matrix.items())
        return sum(map(lambda x: x[1], filter(lambda x: x[0][0] == x[0][1], matrixList)))
        
    def sum(self):
        return sum(self.matrix.values())
    
    def accuracy(self):
        return self.trace()/float(self.sum())
        
    def precision(self, label):
        predictedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][0] == label, self.matrix.items())))
        predictedCorrectly = self.matrix[(label, label)]
        return predictedCorrectly / float(predictedLabel)
        
    def sensitivity(self, label):
        expectedLabel = sum(map(lambda x: x[1], filter(lambda x: x[0][1] == label, self.matrix.items())))
        predictedCorrectly = self.matrix[(label, label)]
        return predictedCorrectly / float(expectedLabel)
        

class Perceptron:
    def __init__(self, num_of_weights, learn_constant=0.01):
        self.weights = np.random.rand(num_of_weights)*2 - 1
        self.learn_constant = learn_constant
        self.epochError = []
    
    def activation_function(self, value):
        return 1 if (value > 0) else 0

    def predict(self, feature):
        return self.activation_function(np.dot(feature, self.weights))

    def learn_feature(self, feature, expected):
        predicted = self.predict(feature)
        error = (expected - predicted)
        self.weights = self.weights + self.learn_constant * error * feature
        return error
        
    def online_epoch(self, features, labels):
        squaredError = 0
        for feature, label in zip(features, labels):
            squaredError += (self.learn_feature(feature, label))**2
        return squaredError
        
    def online_train(self, features, labels, nEpoch):
        self.epochError = []
        for i in range(1,nEpoch):
            self.epochError.append(self.online_epoch(features, labels))
            
    def plot_online_train_error(self):
        plt.plot(self.epochError)
        plt.show()
        
    def test(self, features, labels):
        fuzzyMatrix = FuzzyMatrix()
        for feature, label in zip(features, labels):
            predicted = self.predict(feature)
            fuzzyMatrix.add(predicted, label)
        return fuzzyMatrix
        
perceptron = Perceptron(num_of_weights, learn_constant = 0.001)
perceptron.online_train(train_features, train_labels, 10000)
perceptron.plot_online_train_error()

fuzzymatrix = perceptron.test(test_features, test_labels)

print("Acurácia: %.2f%%" % (fuzzymatrix.accuracy()*100))
print("Precisão classe 0: %.2f%%" % (fuzzymatrix.precision(0)*100))
print("Precisão classe 1: %.2f%%" % (fuzzymatrix.precision(1)*100))
print("Sensibilidade classe 0: %.2f%%" % (fuzzymatrix.sensitivity(0)*100))
print("Sensibilidade classe 1: %.2f%%" % (fuzzymatrix.sensitivity(1)*100))
