import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from data_loading import get_transform
import os
from model import Generator, Discriminator, GeneratorV3, DiscriminatorV3, GeneratorV2, DiscriminatorV2
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torchvision import transforms
from model import DiscriminatorV4 as DiscModel, GeneratorV4 as GenModel
from datetime import datetime
from generate import (
    latent_space_analysis,
    analyze_diversity,
    reconstruct_test_data,
    generate_sample,
    prepare_test_data_loader,
    load_models,
)

image_size = 28 * 28


def plot_loss_curves(d_loss_list, g_loss_list, batch_size, latent_dim, epochs, lr, Times_train_discriminator, save_path, dropout_prob_discriminator, dropout_prob_generator, timestamp=None):
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_list, label="Discriminator Loss")
    plt.plot(g_loss_list, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    info_text = (
        f"Batch Size: {batch_size}\n"
        f"Latent Dim: {latent_dim}\n"
        f"Epochs: {epochs}\n"
        f"Learning rate: {lr}\n"
        f"Dropout prob discriminator: {dropout_prob_discriminator}\n"
        f"Dropout prob generator: {dropout_prob_generator}\n"
        f"Times Train Discriminator: {Times_train_discriminator}"
    )
    plt.gca().set_xlim(left=0, right=len(d_loss_list))
    ax = plt.gca()
    x_offset = ax.get_xlim()[1] * 1.02
    y_offset = ax.get_ylim()[1] * 0.5

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

def label_smoothing(labels, smoothing=0.1):
    """Apply label smoothing to real labels."""
    return labels * (1.0 - smoothing) + 0.5 * smoothing

def compute_gradient_penalty(discriminator, real_images, fake_images, labels, device, lambda_gp=10):
    """
    Compute the gradient penalty for the discriminator.
    """
    # Random weight term for interpolation between real and fake images
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_images)
    
    # Interpolated images
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Compute validity for interpolated images
    interpolated_validity = discriminator(interpolated, labels)
    
    # Compute gradients of validity with respect to interpolated images
    gradients = torch.autograd.grad(
        outputs=interpolated_validity,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_validity).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty term
    gradients = gradients.view(gradients.size(0), -1)  # Flatten the gradients
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


# def train_discriminator(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels, latent_dim, smoothing=0.1):
#     optimizer_D.zero_grad()
#     # Train with real images
#     real_validity = discriminator(real_images, labels)
#     real_labels = torch.ones_like(real_validity).to(device)
#     real_labels = label_smoothing(real_labels, smoothing)
#     real_loss = criterion(real_validity, real_labels)

#     # Train with fake images
#     z = torch.randn(batch_size, latent_dim).to(device)
#     fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
#     fake_images = generator(z, fake_labels)
#     fake_validity = discriminator(fake_images, fake_labels)
#     fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity).to(device))

#     d_loss = real_loss + fake_loss
#     d_loss.backward()
#     optimizer_D.step()
#     return d_loss.item()

# def train_generator(optimizer_G, generator, discriminator, criterion, batch_size, device, latent_dim):
#     optimizer_G.zero_grad()
#     z = torch.randn(batch_size, latent_dim).to(device)
#     fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
#     fake_images = generator(z, fake_labels)
#     validity = discriminator(fake_images, fake_labels)
#     g_loss = criterion(validity, torch.ones_like(validity).to(device))
#     g_loss.backward()
#     optimizer_G.step()
#     return g_loss.item()

def train_discriminator(real_imgs, real_labels, disc_model, gen_model, loss_function, disc_optimizer, batch_size, device, latent_dim, num_classes):
    disc_optimizer.zero_grad()
    batch_size = real_imgs.size(0)
    # Train with real images
    real_validity = disc_model(real_imgs, real_labels)
    real_loss = loss_function(real_validity, torch.ones(batch_size).to(device))
    # Train with fake images
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size)).to(device)
    fake_imgs = gen_model(noise, fake_labels)
    fake_validity = disc_model(fake_imgs, fake_labels)
    fake_loss = loss_function(fake_validity, torch.zeros(batch_size).to(device))

    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_optimizer.step()
    return disc_loss.item()

def train_generator(gen_model, disc_model, loss_function, gen_optimizer, batch_size, device, latent_dim, num_classes):
    gen_optimizer.zero_grad()
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size)).to(device)
    fake_imgs = gen_model(noise, fake_labels)
    validity = disc_model(fake_imgs, fake_labels)
    gen_loss = loss_function(validity, torch.ones(batch_size).to(device))
    gen_loss.backward()
    gen_optimizer.step()
    return gen_loss.item()


def train(batch_size, lr=0.0002, epochs=10, latent_dim=100, save_path="./checkpoints", dropout_prob_discriminator=0.0, dropout_prob_generator=0.3, Times_train_discriminator=5):

    device = get_device()
    print(f"Device: {device}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = os.path.join(save_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # transform = get_transform(train=False)
    # train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_data = datasets.FashionMNIST(root='./data/', train=True, transform=data_transform, download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # generator = Generator(dropout_prob=dropout_prob_generator, latent_dim=latent_dim).to(device)
    # discriminator = Discriminator(dropout_prob=dropout_prob_discriminator).to(device)

    generator = nn.DataParallel(GenModel(img_dim=latent_dim, class_label_size=10, Image_size=image_size).to(device))
    discriminator = nn.DataParallel(DiscModel(class_label_size=10, Image_size=image_size).to(device))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    criterion = nn.BCELoss()

    d_loss_list = []
    g_loss_list = []
    best_g_loss = float("inf")
    g_loss = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (images, labels) in enumerate(train_loader):
            real_images = images.to(device)
            labels = labels.to(device)

            d_loss = 0
            # for _ in range(Times_train_discriminator):
            #     d_loss = train_discriminator(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels, latent_dim)
                
            # for _ in range(3):
            #     g_loss = train_generator(optimizer_G, generator, discriminator, criterion, batch_size, device, latent_dim)
            for _ in range(Times_train_discriminator):
                # d_loss = train_discriminator(discriminator, generator, criterion, optimizer_D, batch_size, device, real_images, labels, latent_dim)
                # g_loss = train_generator(optimizer_G, generator, discriminator, criterion, batch_size, device, latent_dim)
                d_loss = train_discriminator(real_images, labels, discriminator, generator, criterion, optimizer_D, batch_size, device, latent_dim, num_classes=10)
            g_loss = train_generator(generator, discriminator, criterion, optimizer_G, batch_size, device, latent_dim, num_classes=10)
            
            # g_loss = train_generator2(optimizer_G, generator, discriminator, criterion, batch_size, device)
            # optimizer_G.zero_grad()
            # z =torch.randn(batch_size, 100).to(device)
            # fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
            # fake_images = generator(z, fake_labels)
            # validity = discriminator(fake_images, fake_labels)
            # g_loss_tmp = criterion(validity, torch.ones(batch_size).to(device))
            # g_loss_tmp.backward()
            # optimizer_G.step()
            # g_loss = g_loss_tmp.item()

        # scheduler_G.step()
        # scheduler_D.step()

        print(f"EPOCH: {epoch} | D_Loss: {d_loss:.5f} | G_Loss: {g_loss:.5f}")
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)

        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_{timestamp}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_{timestamp}.pth"))
            with open(os.path.join(save_dir, f"generator_structure_{timestamp}.txt"), "w") as f:
                f.write(str(generator))
            with open(os.path.join(save_dir, f"discriminator_structure_{timestamp}.txt"), "w") as f:
                f.write(str(discriminator))
            print(f"Improved G_Loss: {g_loss:.5f}. Model saved with timestamp {timestamp}.")

    plot_loss_curves(d_loss_list, g_loss_list, batch_size, latent_dim, epochs, lr, Times_train_discriminator, save_dir, dropout_prob_discriminator, dropout_prob_generator, timestamp=timestamp)
    return timestamp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    parser.add_argument("--times_train_discriminator", type=int, default=5, help="Times to train discriminator.")
    parser.add_argument("--dropout_prob_discriminator", type=float, default=0.3, help="Dropout probability for the discriminator.")
    parser.add_argument("--dropout_prob_generator", type=float, default=0.0, help="Dropout probability for the generator.")
    args = parser.parse_args()

    ts = train(batch_size=args.batch_size,
          lr=args.lr,
          epochs=args.epochs,
          latent_dim=args.latent_dim,
          Times_train_discriminator=args.times_train_discriminator,
          dropout_prob_discriminator=args.dropout_prob_discriminator,
          dropout_prob_generator=args.dropout_prob_generator
          )

    test_loader = prepare_test_data_loader()

    model_path = f"./checkpoints/{ts}/generator_{ts}.pth"
    save_dir = f"./samples/{ts}"
    generator = Generator(latent_dim=args.latent_dim, dropout_prob=args.dropout_prob_generator)
    generator = load_models(generator, model_path=model_path)
    discriminator = Discriminator()
    discriminator = load_models(discriminator, model_path=f"./checkpoints/{ts}/discriminator_{ts}.pth")

    # Generate samples for visualization
    generate_sample(model_path=model_path, img_dim=args.latent_dim, 
                    dropout_prob_generator=args.dropout_prob_generator, save_dir=save_dir)

    # Reconstruct test data and compare with real images
    reconstruct_test_data(generator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)

    # Analyze diversity of generated images
    analyze_diversity(generator, img_dim=args.latent_dim, save_dir=save_dir)
    
    # Detect outliers using the discriminator
    # outlier_detection(discriminator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)

    # Perform latent space analysis
    latent_space_analysis(generator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)

    return

if __name__ == "__main__":
    main()
