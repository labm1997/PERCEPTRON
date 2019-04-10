# Programa para treinar um perceptron.

# Pacotes importados:
from random import random

# Variáveis de controle:
MINE = 1
ROCK = 0

# Extração dos dados:
with open("./data/sonar.train-data", "r") as train_file:

    # Primeiro, vamos ler cada linha, separando os elementos de uma linha por
    # vírgula em um array e retirando os '\n'.
    lines = list(map(lambda x: x.replace('\n', '').split(','), train_file.readlines()))

    # Depois, extrair os inputs (Tudo menos a última posição).
    train_features = list(map(lambda x: x[:-1], lines))

    # Finalmente, extrair o label conforme o valor da última posição.
    train_labels = list(map(lambda x: MINE if(x[-1] == 'M') else ROCK, lines))

with open("./data/sonar.test-data", "r") as test_file:

    # Primeiro, vamos ler cada linha, separando os elementos de uma linha por
    # vírgula em um array e retirando os '\n'.
    lines = list(map(lambda x: x.replace('\n', '').split(','), test_file.readlines()))

    # Depois, extrair os inputs (Tudo menos a última posição).
    test_features = list(map(lambda x: x[:-1], lines))

    # Finalmente, extrair o label conforme o valor da última posição.
    test_labels = list(map(lambda x: MINE if(x[-1] == 'M') else ROCK, lines))

# Inicialização dos pesos:
weights = [(random() - 0.5) * 0.2 for i in range(len(test_features) + 1)]

print(weights)
