import awkward
import torch


import os
import sys
import time
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import torch.nn as nn

from scipy.stats import binned_statistic
from scipy.optimize import curve_fit


def normalize_dataset(dataset, stats=None):
    """
    Normalizes the 'data', 'xpos', and 'ypos' fields of an Awkward array dataset.
    
    Args:
        dataset: The Awkward array dataset (or dict of Awkward arrays) to normalize.
        stats (dict, optional): Pre-computed means and stds. If None, computes them.
        
    Returns:
        tuple: (normalized_dataset, stats_dictionary)
    """
    # 1. Extract the coordinates
    times = dataset["data"][:, 0:1, :]
    x = dataset["data"][:, 1:2, :]
    y = dataset["data"][:, 2:3, :]
    
    # 2. Compute statistics if they aren't provided
    if stats is None:
        stats = {
            "t_mean": awkward.mean(times),
            "t_std": awkward.std(times),
            "x_mean": awkward.mean(x),
            "x_std": awkward.std(x),
            "y_mean": awkward.mean(y),
            "y_std": awkward.std(y)
        }
        
    # 3. Normalize coordinates using the stats dictionary
    norm_times = (times - stats["t_mean"]) / stats["t_std"]
    norm_x = (x - stats["x_mean"]) / stats["x_std"]
    norm_y = (y - stats["y_mean"]) / stats["y_std"]
    
    # 4. Concatenate and assign back to the dataset
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)
    
    # 5. Normalize the labels using the x and y statistics
    dataset["xpos"] = (dataset["xpos"] - stats["x_mean"]) / stats["x_std"]
    dataset["ypos"] = (dataset["ypos"] - stats["y_mean"]) / stats["y_std"]
    
    return dataset, stats

def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the features are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensor_data = tensor_data.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensor_data)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels

def train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience,
                device, plot_fn=None, plot_interval=10, plot_kwargs=None, model_name=None):
    """
    Trains a given model using the provided training and validation data loaders, loss function, and optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Start the timer for this epoch

        # Training phase
        model.train()
        total_train_loss = 0.0
        for step, (batch_spectra, batch_labels) in enumerate(train_loader):
            
            # ---> FIX: Removed .unsqueeze(1) for PyG compatibility <---
            batch_spectra[0] = batch_spectra[0].to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()

            predictions = model(batch_spectra)

            loss = loss_function(predictions, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                sys.stdout.flush()

        sys.stdout.write("\n")  # Move to the next line after the epoch

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_spectra, batch_labels in val_loader:
                
                # ---> FIX: Removed .unsqueeze(1) for PyG compatibility <---
                batch_spectra[0] = batch_spectra[0].to(device)
                batch_labels = batch_labels.to(device)

                predictions = model(batch_spectra)
                val_loss = loss_function(predictions, batch_labels)

                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds, Patience Counter: {patience_counter}/{patience}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.copy(model.state_dict())
            # Save the best model to the "models" directory
            if not os.path.exists('models'):
                os.makedirs('models')
            if model_name is not None:
                torch.save(best_model, f"models/{model_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if (epoch % plot_interval == 0):
            if plot_fn is not None:

                assert(plot_kwargs is not None)
                assert("test_loader" in plot_kwargs.keys())
                assert("ranges" in plot_kwargs.keys())
                assert("plot_folder" in plot_kwargs.keys())

                plot_fn(model,
                        plot_kwargs["test_loader"],
                        loss_function,
                        device,
                        plot_kwargs["ranges"],
                        train_losses,
                        val_losses,
                        plot_folder=plot_kwargs["plot_folder"],
                        suffix="epoch_%.5d" % epoch)

    return train_losses, val_losses, best_model

def plot_loss(train_losses, val_losses, model_name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses[0:], label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses[0:], label='Validation Loss', color='orange', linewidth=2)
    
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if model_name is not None:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.title(f'{model_name} Training Convergence')
        plt.savefig(f"./plots/{model_name}_training_convergence.png")
    else:
        plt.title('Model Training Convergence')
    plt.show()


# ==========================================
# 1. INDIVIDUAL PLOTTING FUNCTIONS
# ==========================================

def gaussian(x, amp, mu, std):
    """Gaussian function used for fitting the residual histograms."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * std ** 2))

# ==========================================
# 2. The Plotting Functions
# ==========================================
def plot_residuals(truths, preds, model_name=None):
    residual_1 = truths[:, 0] - preds[:, 0]
    residual_2 = truths[:, 1] - preds[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. True vs Predicted Scatter
    axes[0,0].scatter(truths[:, 0], preds[:, 0], alpha=0.5, s=5, color='purple')
    axes[0,0].plot([truths[:, 0].min(), truths[:, 0].max()], [truths[:, 0].min(), truths[:, 0].max()], 'k--', lw=2)
    axes[0,0].set_title('X-Position: True vs Predicted')
    axes[0,0].set_xlabel('True X')
    axes[0,0].set_ylabel('Predicted X')
    
    axes[0,1].scatter(truths[:, 1], preds[:, 1], alpha=0.5, s=5, color='teal')
    axes[0,1].plot([truths[:, 1].min(), truths[:, 1].max()], [truths[:, 1].min(), truths[:, 1].max()], 'k--', lw=2)
    axes[0,1].set_title('Y-Position: True vs Predicted')
    axes[0,1].set_xlabel('True Y')
    axes[0,1].set_ylabel('Predicted Y')

    # 2. Residual Histograms
    # X Residuals
    counts_1, bins_1, _ = axes[1,0].hist(residual_1, bins=50, density=True, color='purple', alpha=0.7, edgecolor='black')
    axes[1,0].axvline(x=0, color='red', linestyle='dashed', linewidth=2)
    
    bin_centers_1 = (bins_1[:-1] + bins_1[1:]) / 2
    initial_guess_1 = [max(counts_1), 0, np.std(residual_1)]
    
    # Use try-except in case the fit fails on noisy data
    try:
        popt_1, _ = curve_fit(gaussian, bin_centers_1, counts_1, p0=initial_guess_1)
        amp_1, mu_1, std_1 = popt_1
        xmin_1, xmax_1 = axes[1,0].get_xlim()
        x_1 = np.linspace(xmin_1, xmax_1, 200)
        axes[1,0].plot(x_1, gaussian(x_1, *popt_1), 'k', linewidth=2, label=f'Core Gaussian\n$\\mu={mu_1:.2f}$, $\\sigma={std_1:.2f}$')
        axes[1,0].legend()
    except RuntimeError:
        print("Warning: Gaussian fit for X residuals failed.")

    axes[1,0].set_title('X: Residuals')
    axes[1,0].set_xlabel('Error (True - Predicted)')
    axes[1,0].set_ylabel('Probability Density')

    # Y Residuals
    counts_2, bins_2, _ = axes[1,1].hist(residual_2, bins=50, density=True, color='teal', alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=0, color='red', linestyle='dashed', linewidth=2)
    
    bin_centers_2 = (bins_2[:-1] + bins_2[1:]) / 2
    initial_guess_2 = [max(counts_2), 0, np.std(residual_2)]
    
    try:
        popt_2, _ = curve_fit(gaussian, bin_centers_2, counts_2, p0=initial_guess_2)
        amp_2, mu_2, std_2 = popt_2
        xmin_2, xmax_2 = axes[1,1].get_xlim()
        x_2 = np.linspace(xmin_2, xmax_2, 200)
        axes[1,1].plot(x_2, gaussian(x_2, *popt_2), 'k', linewidth=2, label=f'Core Gaussian\n$\\mu={mu_2:.2f}$, $\\sigma={std_2:.2f}$')    
        axes[1,1].legend()
    except RuntimeError:
        print("Warning: Gaussian fit for Y residuals failed.")

    axes[1,1].set_title('Y: Residuals')
    axes[1,1].set_xlabel('Error (True - Predicted)')

    plt.suptitle(f'{model_name if model_name else "Model"} Residual Analysis', fontsize=16)
    plt.tight_layout()
    
    if model_name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"./plots/{model_name}_residuals.png", bbox_inches='tight')
    plt.show()

def plot_spatial_resolution(truths, preds, model_name=None):
    dx = truths[:, 0] - preds[:, 0]
    dy = truths[:, 1] - preds[:, 1]
    distance_errors = np.sqrt(dx**2 + dy**2)
    median_resolution = np.median(distance_errors)
    
    plt.figure(figsize=(8, 6))
    plt.hist(distance_errors, bins=50, color='teal', edgecolor='black')
    plt.axvline(median_resolution, color='red', linestyle='dashed', label=f'Median ({median_resolution:.2f})')
    plt.xlabel('Distance Error [m] (True vs Predicted)')
    plt.ylabel('Number of Events')
    plt.title('Vertex Reconstruction Spatial Resolution')
    plt.legend()
    
    if model_name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"./plots/{model_name}_resolution.png", bbox_inches='tight')
    plt.show()

def plot_nhits_performance(truths, preds, nhits, model_name=None):
    dx = truths[:, 0] - preds[:, 0]
    dy = truths[:, 1] - preds[:, 1]
    distance_errors = np.sqrt(dx**2 + dy**2)
    
    max_hits = int(np.max(nhits))
    bins = np.arange(0, max_hits + 10, 10) 
    
    median_errors, bin_edges, _ = binned_statistic(nhits, distance_errors, statistic='median', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, median_errors, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
    plt.scatter(nhits, distance_errors, alpha=0.1, color='gray', s=10, zorder=0, label='Individual Events')
    
    plt.title('Vertex Reconstruction: Error vs. Number of Sensor Hits')
    plt.xlabel('Number of Sensor Hits ($N_{hits}$)')
    plt.ylabel('Median Distance Error ($\\Delta d$)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.ylim(0, np.percentile(distance_errors, 95)) 
    plt.xlim(0, bins[-1])
    
    if model_name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"./plots/{model_name}_nhits.png", bbox_inches='tight')
    plt.show()

def plot_error_vector_map(truths, preds, model_name=None, max_events=300):
    if len(truths) > max_events:
        indices = np.random.choice(len(truths), max_events, replace=False)
        T = truths[indices]
        P = preds[indices]
    else:
        T, P = truths, preds

    X, Y = T[:, 0], T[:, 1]
    U, V = P[:, 0] - X, P[:, 1] - Y
    magnitudes = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, color='black', s=15, alpha=0.5, label='True Position')
    
    quiver = plt.quiver(X, Y, U, V, magnitudes, angles='xy', scale_units='xy', scale=1, cmap='coolwarm', alpha=0.8, width=0.003)
    cbar = plt.colorbar(quiver)
    cbar.set_label('Distance Error Magnitude ($\\Delta d$)')

    plt.title(f'Vertex Reconstruction Error Map (Sampled {len(X)} events)')
    plt.xlabel('Detector X Position')
    plt.ylabel('Detector Y Position')
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    
    if model_name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"./plots/{model_name}_vector_map.png", bbox_inches='tight')
    plt.show()

def plot_spatial_bias(truths, preds, model_name=None):
    X_true = truths[:, 0]
    Y_true = truths[:, 1]
    
    dx = X_true - preds[:, 0]
    dy = Y_true - preds[:, 1]
    distance_errors = np.sqrt(dx**2 + dy**2)
    
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(X_true, Y_true, C=distance_errors, reduce_C_function=np.median, 
                        gridsize=20, cmap='inferno', mincnt=1, alpha=0.9)
    
    cbar = plt.colorbar(hb)
    cbar.set_label('Median Distance Error ($\\Delta d$)')
    
    plt.title('Top-Down Spatial Error Heatmap')
    plt.xlabel('True X Position')
    plt.ylabel('True Y Position')
    plt.axis('equal')
    
    plt.suptitle(f'{model_name if model_name else "Model"} Spatial Bias Analysis', fontsize=16)
    plt.tight_layout()
    
    if model_name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"./plots/{model_name}_spatial_bias.png", bbox_inches='tight')
    plt.show()

# ==========================================
# 3. Master Evaluation Function
# ==========================================
def evaluate_model(model, dataloader, device, model_name="Transformer_Model"):
    model.eval()
    total_loss = 0.0
    
    all_distances = []
    all_predictions = []
    all_labels = []
    all_nhits = []  # Added to track hits per event
    
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            
            packed_data, lengths = batch_data
            
            packed_data = packed_data.to(device)
            batch_labels = batch_labels.to(device)
            
            model_input = [packed_data, lengths]
            predictions = model(model_input)
            
            loss = criterion(predictions, batch_labels)
            total_loss += loss.item() * len(lengths)
            
            diff = predictions - batch_labels
            distances = torch.sqrt(torch.sum(diff**2, dim=1))
            
            all_distances.extend(distances.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
            
            # Record the number of hits directly from the 'lengths' list
            all_nhits.extend(lengths)
            
    avg_loss = total_loss / len(dataloader.dataset)
    mean_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_nhits = np.array(all_nhits)
    
    print(f"Validation MSE Loss: {avg_loss:.4f}")
    print(f"Mean Error Distance: {mean_distance:.2f} meters")
    print(f"Median Error Distance: {median_distance:.2f} meters")
    
    # ---------------------------------------------------------
    # Call all plotting functions automatically
    # ---------------------------------------------------------
    print("\nGenerating evaluation plots...")
    plot_residuals(all_labels, all_predictions, model_name)
    plot_spatial_resolution(all_labels, all_predictions, model_name)
    plot_nhits_performance(all_labels, all_predictions, all_nhits, model_name)
    plot_error_vector_map(all_labels, all_predictions, model_name)
    plot_spatial_bias(all_labels, all_predictions, model_name)
    
    return avg_loss, all_distances, all_predictions, all_labels


def load_best_model(model, model_path, device):
    """
    Loads the best model from the specified path and moves it to the given device.
    
    Args:
        model: The model architecture to load the weights into.
        model_path: The file path where the best model is saved.
        device: The device to move the model to (e.g., 'cpu' or 'cuda').
    Returns:
        The model with loaded weights, moved to the specified device.
    """
    best_model_state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(best_model_state)
    model.to(device)
    return model

