from fuzzymatrix import FuzzyMatrix
import numpy as np

class Perceptron:
    def __init__(self, num_of_weights = 0, learn_constant=0.01, weights=None):
        self.weights = (weights.copy()) if(weights is not None) else (np.random.rand(num_of_weights)*2 - 1)
        self.learn_constant = learn_constant

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
        hits = 0
        for feature, label in zip(features, labels):
            learnError = self.learn_feature(feature, label)
            squaredError += learnError**2
            hits += 1 if(learnError == 0) else 0
            
        return squaredError, hits

    def online_train(self, features, labels, nEpoch):
        epochErrorList = []
        epochHitsList = []

        for i in range(0,nEpoch):
            epochSquaredError, epochHits = self.online_epoch(features, labels)
            epochErrorList.append(epochSquaredError)
            epochHitsList.append(epochHits)
            
        return epochErrorList, epochHitsList

    def test(self, features, labels):
        fuzzyMatrix = FuzzyMatrix()
        for feature, label in zip(features, labels):
            predicted = self.predict(feature)
            fuzzyMatrix.add(predicted, label)
        return fuzzyMatrix
    
    # With this method after each epoch we make the test and get statistics about it
    def online_train_and_test(self, features, labels, nEpochMax, testFeatures, testLabels):
        epochErrorList = []
        epochHitsList = []
        fuzzymatrixList = []

        for i in range(0,nEpochMax):
            epochSquaredError, epochHits = self.online_epoch(features, labels)
            epochErrorList.append(epochSquaredError)
            epochHitsList.append(epochHits)
            fuzzymatrixList.append(self.test(testFeatures, testLabels))
            
        return epochErrorList, epochHitsList, fuzzymatrixList
            
