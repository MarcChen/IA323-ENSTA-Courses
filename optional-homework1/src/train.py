import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from data_loading import get_transform
import os
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime

## Rename with G(Z) and D(X) and  D(G(Z)) the generator and discriminator functions it's more readable





def plot_loss_curves(d_loss_list, g_loss_list, batch_size, latent_dim, epochs, lr, Times_train_discriminator, save_path, dropout_prob_discriminator, dropout_prob_generator, timestamp=None):
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_list, label="Discriminator Loss")
    plt.plot(g_loss_list, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


    # Add training information to the right side of the plot (outside the box)
    info_text = (
        f"Batch Size: {batch_size}\n"
        f"Latent Dim: {latent_dim}\n"
        f"Epochs: {epochs}\n"
        f"Learning rate: {lr}\n"
        f"Dropout prob discriminator: {dropout_prob_discriminator}\n"
        f"Dropout prob generator: {dropout_prob_generator}\n"
        f"Times Train Discriminator: {Times_train_discriminator}"
    )
    # Dynamically position the text outside the plot on the right
    plt.gca().set_xlim(left=0, right=len(d_loss_list))
    ax = plt.gca()
    x_offset = ax.get_xlim()[1] * 1.02  # Place it 10% to the right of the plot area
    y_offset = ax.get_ylim()[1] * 0.5  # Center vertically

    plt.text(
        x_offset,
        y_offset,
        info_text,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round", alpha=0.5, facecolor='lightblue')
    )
    
    # Save and show the updated plot
    plt.savefig(os.path.join(save_path, f"loss_curves_{timestamp}.png"), bbox_inches="tight")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train_discriminator2(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels):
#     optimizer_D.zero_grad()
#     # train with real images
#     real_validity = discriminator(real_images, labels)
#     real_loss = criterion(real_validity, torch.ones(batch_size).to(device))
#     # train with fake images
#     z = torch.randn(batch_size, 100).to(device)
#     fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
#     fake_images = generator(z, fake_labels)
#     fake_validity = discriminator(fake_images, fake_labels)
#     fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))
    
#     d_loss = real_loss + fake_loss
#     d_loss.backward()
#     optimizer_D.step()
#     return d_loss.item()

def train_discriminator(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels):
    optimizer_D.zero_grad()
    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones_like(real_validity).to(device))
    # train with fake images
    z = torch.randn(batch_size, 100).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()
    return d_loss.item()

def train_generator(optimizer_G, generator, discriminator, criterion, batch_size, device):
    optimizer_G.zero_grad()
    z =torch.randn(batch_size, 100).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, torch.ones(batch_size).to(device))
    g_loss.backward()
    optimizer_G.step()
    return g_loss.item()

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.
    
    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset for which to compute the mean and std.

    Returns:
    - mean (float): Mean value of the dataset.
    - std (float): Standard deviation of the dataset.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)
        mean += data.mean(dim=(0, 2, 3)) * batch_samples
        std += data.std(dim=(0, 2, 3)) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()

def train(batch_size, lr=0.0002, epochs=10, latent_dim=100, save_path="./checkpoints", dropout_prob_discriminator=0.0, dropout_prob_generator=0.3,Times_train_discrimnizator=5):
    """Train the conditional GAN."""
    device = get_device()
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = os.path.join(save_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Dataset and DataLoader
    transform = get_transform()
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print("Dataset loaded!")

    # Initialize generator and discriminator
    # generator = Generator(dropout_prob = dropout_prob_generator).to(device)
    generator = Generator().to(device)
    discriminator = Discriminator(dropout_prob = dropout_prob_discriminator).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    print("Starting training...")
    d_loss_list = []
    g_loss_list = []
    best_g_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (images, labels) in enumerate(train_loader):
            real_images = images.to(device)
            labels = labels.to(device)
            generator.train()

            # Train the discriminator multiple times
            d_loss = 0
            for _ in range(Times_train_discrimnizator):
                d_loss += train_discriminator(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels)
            d_loss /= Times_train_discrimnizator  # Average discriminator loss
            # Train the generator
            g_loss = train_generator(optimizer_G, generator, discriminator, criterion, batch_size, device)
            

        print(f"EPOCH: {epoch} | D_Loss: {d_loss:.5f} | G_Loss: {g_loss:.5f}")
        d_loss_list.append(d_loss)  # Append discriminator loss per batch
        g_loss_list.append(g_loss)  # Append generator loss per batch
    
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_{timestamp}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_{timestamp}.pth"))
            print(f"Improved G_Loss: {g_loss:.5f}. Model saved with timestamp {timestamp}.")

    
    # Save the loss curves plot in the new folder
    plot_loss_curves(d_loss_list, g_loss_list, batch_size, latent_dim, epochs, lr, Times_train_discrimnizator, save_dir, dropout_prob_discriminator, dropout_prob_generator, timestamp=timestamp)
    
    
    return

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--save-name",default="", help="Name of the saved model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    parser.add_argument("--times_train_discrimnizator", type=int, default=5, help="Dimension of the latent noise vector.")
    parser.add_argument("--dropout_prob_discriminator", type=float, default=0.3, help="Dropout probability for the discriminator.")
    parser.add_argument("--dropout_prob_generator", type=float, default=0.0, help="Dropout probability for the generator.")
    args = parser.parse_args()

    train(batch_size=args.batch_size,
          lr=args.lr,
          epochs=args.epochs,
          latent_dim=args.latent_dim,
          Times_train_discrimnizator=args.times_train_discrimnizator,
          dropout_prob_discriminator=args.dropout_prob_discriminator,
          dropout_prob_generator=args.dropout_prob_generator
          )

    return

if __name__ == "__main__":
    main()
