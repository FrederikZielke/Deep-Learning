import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary
from helper import normalize, denormalize, train_model, get_normalized_data, evaluate_model
from model_examples import TinyCNN

DATA_PATH = "../data/4/"

# Hyperparameters
learning_rate = 2e-4
batch_size = 32
num_epochs = 100
patience = 10 # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation

# Call the function to get normalized data
spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)

# Convert numpy arrays to PyTorch tensors
spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Split the data into training, validation, and test sets
total_samples = len(spectra_tensor)
train_size = int(train_fraction * total_samples)
val_size = int(val_fraction * total_samples)
test_size = total_samples - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(TensorDataset(spectra_tensor, labels_tensor), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for batching and shuffling the data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model selection
model_choice = 'tiny_CNN'

if model_choice == 'tiny_CNN':
    model = TinyCNN(n_labels)
else:
    raise ValueError("Invalid model choice. Please select 'tiny_CNN'.")

# Print the model summary before moving it to the device
summary(model, input_size=(1, spectra_length))

# Detect and use Apple Silicon GPU (MPS) if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)



mse_calc = nn.MSELoss()  # Use MSE loss for regression

def loss_function(inputs, labels, model):
    """
    Computes the loss between the model predictions and the true labels using mean squared error (MSE).

    Parameters
    ----------
    inputs : array-like
        The input data to the model.
    labels : array-like
        The true labels corresponding to the input data.
    model : object
        The model used to make predictions.

    Returns
    -------
    float
        The computed mean squared error loss.
    """
    predictions = model(inputs)
    loss_result = mse_calc(predictions, labels)
    return loss_result

# Call the function
train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience, device)

# Final evaluation on the test dataset
model.load_state_dict(best_model)  # Load the best model
# save the best model to the "models" directory
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(best_model, f'models/{model_choice}_best.pth')
model.to(device)

all_predictions, all_true_labels,_,_ = evaluate_model(model, test_loader, loss_function, device)
all_predictions = denormalize(all_predictions, ranges)  # Denormalize predictions
all_true_labels = denormalize(all_true_labels, ranges)  # Denormalize true labels

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('plots/Vanilla_CNN_training_validation_loss.png')

# Scatter plots for predictions
plt.figure(figsize=(16,7.5))
for j in range(n_labels):
    plt.subplot(1,3,j+1)
    gt = all_true_labels
    plt.scatter(gt[:,j],y=all_predictions[:,j],s=6,alpha=0.2)
    plt.plot([gt[:,j].min().item(),gt[:,j].max().item()],[gt[:,j].min().item(),gt[:,j].max().item()],c="black",linestyle="dashed",label="Perfect prediction")
    plt.xlabel("true "+labelNames[j])
    plt.ylabel("predicted "+labelNames[j])
    plt.legend()
plt.tight_layout()
plt.savefig('plots/Vanilla_CNN_scatter.png')
plt.show()
