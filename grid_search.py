import os
from itertools import product

# Define hyperparameter grids : one config to add in the end due to interruption: 1_8_x
Nombre_de_heads = [2, 4, 8, 16]
Hidden_dim = [1, 2, 4, 8] 
Bottleneck_dim = [1, 4, 8, 16, 32]

# Fixed parameters
dataset_name = "InHARD_13"  # Change as needed: InHARD/IKEA/HA4M
cuda_device = 0  # Specify your CUDA device number
base_dir = "./data/"  # Replace with your data directory path
split_number = 1  # Specify split number

# Create all combinations of hyperparameters
configurations = list(product(Nombre_de_heads, Hidden_dim, Bottleneck_dim))
print(f"Total configurations: {len(configurations)}")

# Directory to store results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Loop through each configuration
for i, (heads, hidden_dim, bottleneck_dim) in enumerate(configurations, 1):
    print(f"Running configuration {i}/{len(configurations)}: "
          f"Nombre_de_heads={heads}, Hidden_dim={hidden_dim}, Bottleneck_dim={bottleneck_dim}")
    
    # Build the command
    command = (
        f"python train_EAD_g.py "
        f"--dataset_name {dataset_name} "
        f"--cudad {cuda_device} "
        f"--base_dir {base_dir} "
        f"--split {split_number} "
        f"--num_heads {heads} "
        f"--hidden_dim {hidden_dim} "
        f"--bottleneck_dim {bottleneck_dim}"
    )
    
    # Log the command to a file for reference
    with open(f"{results_dir}/config_{i}.txt", "w") as f:
        f.write(command + "\n")
    
    # Run the command
    os.system(command)
