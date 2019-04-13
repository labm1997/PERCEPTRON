import numpy as np

# Package to extract data from sonar files.

# Auxiliary functions:
def extract_features(sample):
    features = list(map(float, sample[:-1]))
    features.append(1.0)    # Add a constant input for bias weight.
    return np.array(features)

def extract_label(sample, label_values):
    label = sample[-1]

    if(label == 'M'):
        return label_values[0]
    elif(label == 'R'):
        return label_values[1]
    else:
        return -1   # This indicates an error!

def extract_sample(line):
    treated_line = line.replace('\n', '')   # Remove '\n' to avoid confusion.
    sample = treated_line.split(',')
    return sample

# Main function:
def extract_sonar_data(file_path, label_values=[1,0]):

    with open(file_path, "r") as file:
        samples = list(map(extract_sample, file.readlines()))
        features = list(map(extract_features, samples))
        labels = list(extract_label(sample, label_values) for sample in samples)

    return features, labels
