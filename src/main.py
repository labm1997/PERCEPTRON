# Programa para treinar um perceptron.

# Pacotes importados:
from random import random

# Funções auxiliares:
def extract_sonar_data(file_path, mine_value=1, rock_value=0):

# TODO idea: Replace these lambdas to improve readability!

    with open(file_path, "r") as file:
        # Primeiro, vamos ler cada linha, separando os elementos de uma linha por
        # vírgula em um array e retirando os '\n'.
        lines = list(map(lambda x: x.replace('\n', '').split(','),
                         file.readlines()))

        # Depois, extrair os inputs como floats (Tudo menos a última posição).
        features = list(map(lambda x: [float(item) for item in x[:-1]], lines))

        # Append de bias para última posição de cada conjunto de features.
        for entry in features:
            entry.append(1.0)

        # Finalmente, extrair o label conforme o valor da última posição.
        labels = list(map(lambda x: mine_value if(x[-1] == 'M') else rock_value,
                          lines))

    return features, labels

# Extração dos dados:
train_features, train_labels = extract_sonar_data("./data/sonar.train-data")
test_features, test_labels = extract_sonar_data("./data/sonar.test-data")

# Inicialização dos pesos:
weights = [(random() - 0.5) * 0.2 for i in range(len(test_features) + 1)]

print(weights)
