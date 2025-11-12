# %%
# Read the file
import os
DATASET = os.path.dirname(os.path.abspath(__name__))
file_path = os.path.join(DATASET, '3class/BRSET_TL/ft_3class_DR_dinov2.sh-60060119.out')
train_loss = []
val_loss = []
f1_score = []
with open(file_path, 'r') as file:
    file_contents = file.readlines()
    for line in file_contents:
        if line.startswith('Epoch'):
            _, train_l, val_l, f1_s = line.split(',')
            train_loss.append(float(train_l.split(' ')[-1]))
            val_loss.append(float(val_l.split(' ')[-1]))
            f1_score.append(float(f1_s.split(' ')[-1]))

             
        
        

# %%
import matplotlib.pyplot as plt

# Plot train_loss, val_loss, and f1_score
epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.plot(epochs, f1_score, label='F1 Score', marker='o')

# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.ylim(0, 1.5)
plt.xticks(epochs)
plt.title('Training Loss, Validation Loss, and F1 Score Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


