# Program to train, test and generate plots

# Imports:
from data_extractor import extract_sonar_data
from random import random
from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Data extractor:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Constants and arguments
if(len(sys.argv) != 3):
    print("[Error] Number of program arguments is wrong!")
    print("Usage: python src/main.py learn_constant num_of_epochs")
    sys.exit(1)

learn_constant = float(sys.argv[1])
max_num_epochs = int(sys.argv[2])

num_of_weights = len(train_features[0])
weights = np.random.rand(num_of_weights)*2 - 1
windowSize = 100

# Initialize the perceptron
perceptron = Perceptron(learn_constant = learn_constant, weights = weights)

# Online Train, will return statistics
trainError, trainHits, testFuzzyMatrix = perceptron.online_train_and_test(train_features, train_labels, max_num_epochs, test_features, test_labels)

# Plot with moving average
def plot_moving_average(data, window, title):
    plt.plot(np.convolve(data, np.ones((window,))/window, mode='valid'), label = title)

# Initial configuration
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

if not os.path.exists("resultados/"):
    os.makedirs("resultados/")

# Test plot
f = plt.figure()
plt.title("Evolução do resultado dos testes ao longo do treinamento aplicando média móvel com janela %d" % windowSize)
plt.xlabel("Número da época")
plot_moving_average(list(map(lambda x: x.accuracy()*100, testFuzzyMatrix)), windowSize, "Acurácia")
plot_moving_average(list(map(lambda x: x.precision(0)*100, testFuzzyMatrix)), windowSize, "Precisão Rocha")
plot_moving_average(list(map(lambda x: x.precision(1)*100, testFuzzyMatrix)), windowSize, "Precisão Mina")
plot_moving_average(list(map(lambda x: x.sensitivity(0)*100, testFuzzyMatrix)), windowSize, "Sensibilidade Rocha")
plot_moving_average(list(map(lambda x: x.sensitivity(1)*100, testFuzzyMatrix)), windowSize, "Sensibilidade Mina")
plt.legend(loc='best')
plt.savefig("resultados/test.png", bbox_inches='tight')
plt.close(f)

# Train plot
f = plt.figure()
plt.title("Acurácia da época")
plt.xlabel("Número da época")
plot_moving_average(list(map(lambda x: x*100/len(train_features), trainHits)), windowSize, "Acurácia")
plt.legend(loc='best')
plt.savefig("resultados/train_accuracy.png", bbox_inches='tight')
plt.close(f)

f = plt.figure()
plt.title("Erro quadrático da época")
plt.xlabel("Número da época")
plot_moving_average(trainError, windowSize, "Erro quadrático")
plt.legend(loc='best')
plt.savefig("resultados/train_error.png", bbox_inches='tight')
plt.close(f)


# Print final stats
print("Resultados do teste:")
print("Acurácia: %.2f%%" % (testFuzzyMatrix[-1].accuracy()*100))
print("Precisão classe 0: %.2f%%" % (testFuzzyMatrix[-1].precision(0)*100))
print("Precisão classe 1: %.2f%%" % (testFuzzyMatrix[-1].precision(1)*100))
print("Sensibilidade classe 0: %.2f%%" % (testFuzzyMatrix[-1].sensitivity(0)*100))
print("Sensibilidade classe 1: %.2f%%" % (testFuzzyMatrix[-1].sensitivity(1)*100))
