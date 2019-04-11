# Package to extract data from sonar files.

# Auxiliary functions:
def extract_features(line_items):
    features = list(map(float, line_items[:-1]))
    features.append(1.0)    # Add a constant input for bias weight.
    return features

def extract_label(line_items, label_values = [1, 0]):
    label = line_items[-1]

    if(label == 'M'):
        return label_values[0]
    elif(label == 'R'):
        return label_values[1]
    else:
        return -1   # This indicates an error!

def format_line(line):
    treated_line = line.replace('\n', '')   # Remove '\n' to avoid confusion.
    line_items = treated_line.split(',')
    return line_items

# Main function:
def extract_sonar_data(file_path, mine_value=1, rock_value=0):

    with open(file_path, "r") as file:
        input_items = list(map(format_line, file.readlines()))
        features = list(map(extract_features, input_items))
        labels = list(map(extract_label, input_items))

    return features, labels
