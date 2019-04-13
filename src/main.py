# Program to train a perceptron.

# Imports:
from data_extractor import extract_sonar_data
from random import random
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# Data extractor:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Weight initialization:
num_of_weights = len(train_features[0])

# Initialize the perceptron
perceptron = Perceptron(num_of_weights, learn_constant = 0.001)

# Online Train, will return statistics
error, hits = perceptron.online_train(train_features, train_labels, 5000)

# Plot statistics
plt.plot(error)
plt.title("Erro quadrático")
plt.xlabel("Número da época")
plt.show()

plt.title("Acurácia percentual")
plt.xlabel("Número da época")
plt.plot(list(map(lambda x: x/len(train_features)*100, hits)))
plt.show()

# Test perceptron, returns stats in a fuzzymatrix
fuzzymatrix = perceptron.test(test_features, test_labels)

# Print stats
print("Resultados do teste:")
print("Acurácia: %.2f%%" % (fuzzymatrix.accuracy()*100))
print("Precisão classe 0: %.2f%%" % (fuzzymatrix.precision(0)*100))
print("Precisão classe 1: %.2f%%" % (fuzzymatrix.precision(1)*100))
print("Sensibilidade classe 0: %.2f%%" % (fuzzymatrix.sensitivity(0)*100))
print("Sensibilidade classe 1: %.2f%%" % (fuzzymatrix.sensitivity(1)*100))
