import matplotlib.pyplot as plt
import re

# Read the data from the text file
with open('training.txt', 'r') as file:
    data = file.readlines()

# Lists to hold the extracted values
epochs = []
accuracy = []
val_loss = []

# Regex patterns for extracting accuracy and validation loss
accuracy_pattern = re.compile(r'accuracy of the model after epoch (\d+),.*  =  (\d+\.\d+)%')
loss_pattern = re.compile(r'Validation loss =  (\d+\.\d+),.*')

# Extract data
for line in data:
    print(line)
    # Match accuracy
    acc_match = accuracy_pattern.search(line)
    print(acc_match)
    if acc_match:
        epoch = int(acc_match.group(1))
        acc = float(acc_match.group(2))
        print(epoch,' : ', acc)
        epochs.append(epoch)
        accuracy.append(acc)
    
    # Match validation loss
    loss_match = loss_pattern.search(line)
    if loss_match:
        loss = float(loss_match.group(1))
        val_loss.append(loss)

# Plotting Accuracy and Validation Loss
plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'o-', color='blue')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_loss) + 1), val_loss, 'o-', color='red')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
