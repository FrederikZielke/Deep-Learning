"""
This script demonstrates how to deconstruct a PDF layer by layer, both in forward and backward passes.
The script generates data from a mixture of Gaussians and fits a mixture model to the data using a
concatenation of Gaussianisation flows and one affine flow.
The script then deconstructs the PDF layer by layer and plots the fitted density at each layer.
This script shows how a complex PDF can be constructed by the application of successive Gaussianization flows.
"""
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import jammy_flows
import pylab


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ----- Data Generation -----
# we generate data from a mixture of Gaussians
N = 1000000  # total samples
data = torch.zeros(N)  # Initialize data tensor with zeros

# Define the parameters of the mixture model
nn = np.array([0.2, 0.3, 0.2, 0.1, 0.2]) * N  # Relative number of samples for each component
nn = nn.astype(int)  # Convert to integer
mus = torch.tensor([-3, -2, -1, 1, 3], dtype=torch.float32)  # Means of the components
sigmas = torch.tensor([0.5, 0.2, 0.4, 1, 0.4], dtype=torch.float32)  # Standard deviations of the components

# Generate data for each component
for i in range(len(nn)):
    start_idx = nn[:i].sum()  # Start index for the current component
    end_idx = nn[:i + 1].sum()  # End index for the current component
    data[start_idx:end_idx] = torch.normal(mus[i], sigmas[i], size=(nn[i],))  # Generate data

# Shuffle the data
data = data[torch.randperm(N)]

# Plot histogram of the generated data
plt.hist(data.numpy(), bins=500, density=True)
plt.show()


def deconstruct_pdf_layer_by_layer(model_pdf):
    """
    Deconstructs a model's PDF layer by layer, both in forward and backward passes.

    Parameters
    ----------
    model_pdf : object
        The model object containing the PDF definitions and layer information.
    Returns
    -------
    new_pdfs_forward : list of tuples
        A list of tuples where each tuple contains a new PDF and its corresponding PDF definition list
        created during the forward pass.
    new_pdfs_backward : list of tuples
        A list of tuples where each tuple contains a new PDF and its corresponding PDF definition list
        created during the backward pass.
    """

    # Ensure the model has only one PDF definition
    assert len(model_pdf.pdf_defs_list) == 1

    new_pdfs_forward = []  # List to store forward PDFs
    num_layers = len(model_pdf.layer_list[0])  # Number of layers in the model
    max_layers = 1  # Initialize max_layers to 1
    flow_def = model_pdf.pdf_defs_list[0]  # Get the PDF definition

    # Forward pass: create PDFs layer by layer
    while max_layers <= num_layers:
        this_pdf_def_list = model_pdf.flow_defs_list[0][:max_layers]  # Get current PDF definition list
        new_pdf = jammy_flows.pdf(flow_def, this_pdf_def_list)  # Create new PDF
        for layer_ind, layer in enumerate(model_pdf.layer_list[0][:max_layers]):
            # Copy over parameters
            cur_new_layer = new_pdf.layer_list[0][layer_ind]
            for n, p in layer.named_parameters():
                getattr(cur_new_layer, n).data = p.data
        max_layers += 1
        new_pdfs_forward.append((new_pdf, this_pdf_def_list))  # Append new PDF to the list

    # Backward pass: create PDFs layer by layer
    new_pdfs_backward = []
    while max_layers >= 1:
        this_pdf_def_list = model_pdf.flow_defs_list[0][-max_layers + 1:]  # Get current PDF definition list
        new_pdf = jammy_flows.pdf(flow_def, this_pdf_def_list)  # Create new PDF
        for layer_ind, layer in enumerate(model_pdf.layer_list[0][-max_layers + 1:]):
            # Copy over parameters
            cur_new_layer = new_pdf.layer_list[0][layer_ind]
            for n, p in layer.named_parameters():
                getattr(cur_new_layer, n).data = p.data
        max_layers -= 1
        new_pdfs_backward.append((new_pdf, this_pdf_def_list))  # Append new PDF to the list

    return new_pdfs_forward, new_pdfs_backward  # Return the lists of forward and backward PDFs


# ----- Prepare DataLoader for Mini-batch Training -----
batch_size = 200  # Define batch size
dataset = TensorDataset(data)  # Create dataset from the generated data
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Create DataLoader


def plot_global(forward_list, backward_list, iteration, loss):
    """
    Plots the forward and backward subflows and saves the plots and data.

    Parameters
    ----------
    forward_list : list
        A list of tuples where each tuple contains a PDF and its corresponding definitions for the forward subflows.
    backward_list : list
        A list of tuples where each tuple contains a PDF and its corresponding definitions for the backward subflows.
    iteration : int
        The current iteration number.
    loss : float
        The current loss value.
    Returns
    -------
    None
    """
    plotfolder = "./plots"  # Define folder for saving plots
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)  # Create folder if it doesn't exist

    num_subflows = len(forward_list)  # Number of subflows
    # Global plot in one thing (backward/forward)
    fig, ax_dict = pylab.subplots(2, num_subflows, figsize=(num_subflows * 2.3, 8))

    xs = np.linspace(-12, 12, 1000)  # Define x-axis values
    output = np.zeros((8, len(xs)))  # Initialize output array
    output[0] = xs  # Set first row to x-axis values
    for row in [0, 1]:
        for col in range(len(forward_list)):
            if row == 0:
                # Forward plots
                fw_item = forward_list[col]
                this_pdf = fw_item[0]
                this_defs = fw_item[1]
            else:
                bw_item = backward_list[num_subflows - 1 - col]
                this_pdf = bw_item[0]
                this_defs = bw_item[1]

            ax = ax_dict[row, col]
            fitted_pdf = np.zeros_like(xs)
            fitted_pdf, _, _ = this_pdf(torch.from_numpy(xs)[:, None])
            fitted_pdf = fitted_pdf.exp().numpy()
            if row == 0:
                output[col + 1] = copy.copy(fitted_pdf)

            # Plot histogram of the data and the fitted density
            ax.hist(data.numpy(), bins=100, density=True, alpha=0.3, label="Data histogram")
            ax.plot(xs, fitted_pdf, color="red", lw=2, label="Fitted mixture")
            ax.set_title(this_defs)
            ax.set_xlabel("x")
            ax.set_ylabel("Density")

    fig.suptitle("iter %.5d - loss %.3f" % (iteration, loss))
    fig.tight_layout()

    np.savez(os.path.join(plotfolder, "global_iter_%.5d.npz" % iteration), output)
    pylab.savefig(os.path.join(plotfolder, "global_iter_%.5d.png" % iteration))
    pylab.close(fig)


# ----- Training Loop -----
num_epochs = 5  # You can increase the number of epochs for better convergence
iteration = 0
plt.ion()  # Turn on interactive mode for plotting

model_pdf = jammy_flows.pdf("e1", "ggggggt")  # Initialize model
model_pdf.double()  # Convert model parameters to double precision

# ----- Optimizer -----
optimizer = optim.Adam(model_pdf.parameters(), lr=0.001)  # Initialize optimizer

for epoch in range(num_epochs):
    for batch in dataloader:
        x_batch = batch[0].unsqueeze(-1)  # Get batch data and add a dimension
        x_batch = x_batch.to(dtype=torch.float64)  # Convert to double precision
        optimizer.zero_grad()  # Zero the gradients
        res, _, _ = model_pdf(x_batch)  # Forward pass
        loss = (-res).mean()  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        # Every 1000 batch iterations, plot the current fit
        if iteration % 1000 == 0:
            current_loss = loss.item()
            print(f"Epoch {epoch} Iteration {iteration}, Loss = {current_loss:.4f}")
            with torch.no_grad():
                fw_list, bw_list = deconstruct_pdf_layer_by_layer(model_pdf)
                plot_global(fw_list, bw_list, iteration, loss.cpu().detach().numpy())

        iteration += 1

with torch.no_grad():
    fw_list, bw_list = deconstruct_pdf_layer_by_layer(model_pdf)
    plot_global(fw_list, bw_list, iteration, loss.cpu().detach().numpy())

