# Program to train a perceptron.

# Imports:
from data_extractor import extract_sonar_data
from random import random
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import sys

# Program arguments:
if(len(sys.argv) != 3):
    print("[Error] Number of program arguments is wrong!")
    print("Usage: python src/main.py learn_constant num_of_epochs")
    sys.exit(1)

learn_constant = float(sys.argv[1])
num_of_epochs = int(sys.argv[2])

# Data extractor:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Weight initialization:
num_of_weights = len(train_features[0])

# Initialize the perceptron
perceptron = Perceptron(learn_constant = learn_constant, num_of_weights = num_of_weights)

# Online Train, will return statistics
error, hits = perceptron.online_train(train_features, train_labels,
                                      num_of_epochs)

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
