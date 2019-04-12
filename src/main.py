# Program to train a perceptron.

# Imports:
from data_extractor import extract_sonar_data
from random import random

# Data extractor:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Weight initialization:
num_of_weights = len(train_features[0])
weights = [(random() - 0.5) * 0.2 for i in range(num_of_weights)]

