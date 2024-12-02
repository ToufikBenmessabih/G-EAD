import json

# Specify the path to your JSON file
json_file_path = 'gt.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract key names based on 'subset'
training_keys = [key for key, item in data['database'].items() if item['subset'] == 'training']
validation_keys = [key for key, item in data['database'].items() if item['subset'] == 'validation']

# Save the lists to separate text files
with open('train.split1.bundle.txt', 'w') as file:
    for key in training_keys:
        file.write(f"{key}.txt\n")

with open('test.split1.bundle.txt', 'w') as file:
    for key in validation_keys:
        file.write(f"{key}.txt\n")
