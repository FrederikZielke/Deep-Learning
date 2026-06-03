import sys, os
from typing import Tuple, NamedTuple

import torch
from torch import Tensor, nn, optim  # For Optimizer
from torch.utils.data import DataLoader  # For Data Loader
from torch.utils.tensorboard import SummaryWriter  # For Tensor Board Visualisation
from torchvision import transforms, datasets, utils  # For Data Set, Image Transforms

# Hyperparameters
learning_rate = 3e-4
batch_size = 32  # Batch size
num_epochs = 100
log_step = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128  # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784

# Select device to train on, we use GPU if available, otherwise we use CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# we define a transform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
# let's create a dataloader to load the data in batches
loader = DataLoader(datasets.MNIST(root="dataset/", transform=myTransforms, download=True), batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    """
    Generator Model
    """

    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential( # pyright: ignore
            nn.Linear(latent_dimension, ...) #TODO define a suitable hidden dimension # pyright: ignore[reportArgumentType]
            ... #TODO define a suitable network architecture for the generator
            nn.Tanh(),  # It is helpful to use the tanh activation function to force the output into the [-1,1] range that our normalized images have.
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    """
    Discriminator Model
    """

    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential( # pyright: ignore
            nn.Linear(image_dimension, ...) #TODO define a suitable hidden dimension # pyright: ignore[reportArgumentType]
            ... #TODO define a suitable network architecture for the discriminator,
        )

    def forward(self, x):
        return self.disc(x)


# initialize networks and optimizers
discriminator = Discriminator().to(device)
generator = Generator().to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate)

# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Training Loop
step = 0
print("Started Training and visualization...")
for epoch in range(num_epochs):
    # loop over batches
    print()
    for batch_idx, (real, _) in enumerate(loader):
        # First we train the discriminator on real images vs. generated images

        # Get the real images and flatten them
        # for simplicity, we flatten the image to a vector and to use simple MLP networks
        # 28 * 28 * 1 flattens to 784
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Step 1) generate fake images
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = generator(noise)

        # Step 2) Train Discriminator:
        # - predict the discriminator output for real images
        # - real images are labeled as 1
        # - calculate the loss for real images

        # - predict the discriminator output for fake images
        # -fake images are labeled as 0
        # -calculate the loss for fake images

        # -average the loss for real and fake images

        # - now update the weights of the discriminator by backpropagating the loss through the discriminator
        # the generator is not updated in this step
        # HINT: call the `backward` method of the discriminator with the argument `retain_graph=True` to keep the computational graph
        # this is necessary because we will use the same discriminator to train the generator

        # Train Generator:
        # Now train the generator by generating fake images and passing them through the discriminator
        # You can do a little trick and modify the original objective function of
        # "minimizing the probability of the discriminator predicting the fake images as fake"
        # to "maximizing the probability of the discriminator predicting the fake images as real"
        # this leads to a faster training of the generator when it does not represent the real data well
        # this is a common trick in GANs
        # for more information see section 17.1.2 of the book Deep Learning by Bishop and Bishop

        # Todo:
        # - pass the fake images through the discriminator
        # - calculate the loss (by passing the output of the discriminator through the criterion with labels set to 1 (real images)
        # - update the weights of the generator

        # print the progress
        print(
            f"\rEpoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)}, Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}",  # pyright: ignore[reportUndefinedVariable]
            end="",
        )

        # Log the losses and example images to tensorboard
        if batch_idx % log_step == 0:
            with torch.no_grad():
                # Generate noise via Generator, we always use the same noise to see the progression
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)  # pyright: ignore[reportUndefinedVariable]
                # Get real data
                data = real.reshape(-1, 1, 28, 28)
                # make grid of pictures and add to tensorboard
                img_grid_fake = utils.make_grid(fake, normalize=True)
                img_grid_real = utils.make_grid(data, normalize=True)

                # TODO: add the images and losses to tensorboard
                # HINT: use the SummaryWriter to add the images and scalars to tensorboard
                # HINT: use the `add_image` method to add the images to tensorboard
                # HINT: use the `add_scalar` method to add the losses to tensorboard

                # increment step
                step += 1
