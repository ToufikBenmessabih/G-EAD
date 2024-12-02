import json

# Load the JSON data from the file
with open('gt.json') as f:
    data = json.load(f)

# Extract the durations from the annotations
durations = [entry['duration'] for entry in data['database'].values()]

# Find the maximum duration
max_duration = max(durations)
min_duration = min(durations)

print("max (min), min (sec) durations:", max_duration/60, min_duration)