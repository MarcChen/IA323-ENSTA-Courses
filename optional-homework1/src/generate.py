import argparse
import dis
import os
from sympy import im
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator as G , Discriminator as D, Generator_For_LatentSpaceAnalysis as G_LSA
from data_loading import get_transform
import re
import numpy as np
from sklearn.decomposition import PCA


def test_discriminator(discriminator, generator, test_loader, save_dir, img_dim=100, device='cpu'):
    """
    Evaluate the Discriminator's performance on real and fake samples, and save a figure with three pie charts.
    
    Args:
        discriminator (nn.Module): The trained Discriminator model.
        generator (nn.Module): The trained Generator model.
        test_loader (DataLoader): DataLoader for real images.
        img_dim (int): Dimensionality of the latent vector.
        save_dir (str): Directory to save the accuracy pie charts.
        device (str): 'cpu' or 'cuda'.
    """
    discriminator.eval()
    generator.eval()
    total_real = 0
    total_fake = 0
    correct_real = 0
    correct_fake = 0

    with torch.no_grad():
        # Evaluate on real samples
        for real_imgs, labels in test_loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            outputs_real = discriminator(real_imgs, labels)
            predictions_real = (outputs_real > 0.5).float()
            correct_real += predictions_real.sum().item()
            total_real += real_imgs.size(0)

        # Generate and evaluate fake samples
        for _ in range(len(test_loader)):
            z = torch.randn(real_imgs.size(0), img_dim).to(device)
            fake_labels = torch.randint(0, 10, (real_imgs.size(0),)).to(device)
            fake_imgs = generator(z, fake_labels)
            outputs_fake = discriminator(fake_imgs, fake_labels)
            predictions_fake = (outputs_fake < 0.5).float()
            correct_fake += predictions_fake.sum().item()
            total_fake += fake_imgs.size(0)

    # Calculate accuracies
    real_acc = correct_real / total_real * 100
    fake_acc = correct_fake / total_fake * 100
    overall_acc = (correct_real + correct_fake) / (total_real + total_fake) * 100

    print(f"Real Accuracy: {real_acc:.2f}%")
    print(f"Fake Accuracy: {fake_acc:.2f}%")
    print(f"Overall Accuracy: {overall_acc:.2f}%")

    # Prepare data for pie charts
    correct_overall = correct_real + correct_fake
    incorrect_overall = (total_real + total_fake) - correct_overall

    correct_real_pct = correct_real
    incorrect_real_pct = total_real - correct_real

    correct_fake_pct = correct_fake
    incorrect_fake_pct = total_fake - correct_fake

    # Plot three pie charts in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Overall Accuracy Pie Chart
    axes[0].pie([correct_overall, incorrect_overall],
                labels=["Correct", "Incorrect"], autopct="%1.1f%%", 
                colors=["#4CAF50", "#F44336"], startangle=140)
    axes[0].set_title("Overall Accuracy")

    # Real Accuracy Pie Chart
    axes[1].pie([correct_real_pct, incorrect_real_pct],
                labels=["Correct", "Incorrect"], autopct="%1.1f%%",
                colors=["#2196F3", "#FF9800"], startangle=140)
    axes[1].set_title("Real Accuracy")

    # Fake Accuracy Pie Chart
    axes[2].pie([correct_fake_pct, incorrect_fake_pct],
                labels=["Correct", "Incorrect"], autopct="%1.1f%%",
                colors=["#9C27B0", "#E91E63"], startangle=140)
    axes[2].set_title("Fake Accuracy")

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    pie_chart_path = os.path.join(save_dir, "discriminator_accuracies_over_testset.png")
    plt.tight_layout()
    plt.savefig(pie_chart_path, bbox_inches="tight")
    plt.close()

    print(f"Accuracy pie charts saved to '{pie_chart_path}'")



import random
import matplotlib.pyplot as plt
import os
import torch

def random_interpolate_labels(generator, save_dir, num_rows=5, img_dim=100, device='cpu'):
    """
    Randomly interpolate between two labels for multiple rows of images.

    Args:
        generator (nn.Module): The trained Generator model.
        save_dir (str): Directory to save the interpolation plot.
        num_rows (int): Number of rows (interpolations) to generate.
        img_dim (int): Dimensionality of the latent vector.
        device (str): 'cpu' or 'cuda' for computation.
    """
    # Define a dictionary of labels for easy selection
    label_dict = {
        'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4,
        'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9
    }
    label_names = list(label_dict.keys())  # List of label names

    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    fig, axs = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    
    for i in range(num_rows):
        # Randomly select two different labels
        label1_name, label2_name = random.sample(label_names, 2)
        label1 = label_dict[label1_name]
        label2 = label_dict[label2_name]

        # Fix a single random latent vector
        z = torch.randn(1, img_dim).to(device)
        label_tensor1 = torch.LongTensor([label1]).to(device)
        label_tensor2 = torch.LongTensor([label2]).to(device)
        
        # Linearly interpolate between labels
        label_interp = torch.tensor((label1 + label2) / 2).round().long().to(device)

        # Generate images
        with torch.no_grad():
            img1 = generator(z, label_tensor1).cpu().squeeze()
            img_interp = generator(z, label_interp.unsqueeze(0)).cpu().squeeze()
            img2 = generator(z, label_tensor2).cpu().squeeze()

        # Plot
        axs[i, 0].imshow(img1, cmap='gray')
        axs[i, 0].set_title(f"{label1_name}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(img_interp, cmap='gray')
        axs[i, 1].set_title(f"Interpolated")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(img2, cmap='gray')
        axs[i, 2].set_title(f"{label2_name}")
        axs[i, 2].axis('off')

    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, "random_label_interpolation.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Random label interpolations saved to '{save_path}'")

def load_and_transfer_generator_layers(pth_path, latent_analysis_generator_class, device='cpu'):
    """
    Load a Generator model trained with nn.DataParallel from a .pth file and
    transfer the first two layers to a Generator_For_LatentSpaceAnalysis.

    Args:
        pth_path (str): Path to the .pth file containing the saved Generator model.
        latent_analysis_generator_class (type): The class definition for Generator_For_LatentSpaceAnalysis.
        device (str): Device to load the model ('cpu' or 'cuda').

    Returns:
        latent_analysis_generator (torch.nn.Module): Initialized Generator_For_LatentSpaceAnalysis
                                                      with transferred layers.
    """
    # Load the saved state_dict
    state_dict = torch.load(pth_path, map_location=device)

    # Remove the 'module.' prefix if the model was trained with DataParallel
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    # Initialize the Generator class and load the weights
    generator = G().to(device)
    generator.load_state_dict(state_dict)

    # Initialize the Generator_For_LatentSpaceAnalysis
    latent_analysis_generator = latent_analysis_generator_class()

    # Transfer the embedding layer
    latent_analysis_generator.label_emb.weight.data = generator.label_emb.weight.data.clone()

    # Transfer the weights and biases of the first linear layer
    latent_analysis_generator.fc1.weight.data = generator.model[0].weight.data.clone()
    latent_analysis_generator.fc1.bias.data = generator.model[0].bias.data.clone()

    print("Layers successfully transferred to the latent analysis generator.")
    return latent_analysis_generator


def load_models(model, model_path="./checkpoints/generator.pth", was_data_parallel=False):
    """Load the model weights."""
    try:
        checkpoint = torch.load(model_path)
        
        # Print the keys for debugging
        # print("Original checkpoint keys:", checkpoint.keys())
        
        if was_data_parallel:
            # Adjust keys by removing "module."
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            # print("Modified state_dict keys:", state_dict.keys())
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        if 'checkpoint' in locals():
            print("Checkpoint keys:", checkpoint.keys())
        print("Model state_dict keys:", model.state_dict().keys())
    
    return model



def outlier_detection(discriminator, test_loader, img_dim, save_dir):
    """Identify outliers using the discriminator."""
    discriminator.eval()
    outliers = []
    scores = []

    os.makedirs(save_dir, exist_ok=True)

    for idx, (real_img, label) in enumerate(test_loader):
        if idx >= 10:  # Limit to 10 samples for visualization
            break
        with torch.inference_mode():
            real_img = real_img.to(torch.device('cpu'))
            score = discriminator(real_img.unsqueeze(0)).item()
            scores.append(score)
            if score < 0.5:  # Arbitrary threshold for outlier detection
                outliers.append((real_img.squeeze().numpy(), label))

    # Visualization: Outliers
    if outliers:
        f, axes = plt.subplots(1, len(outliers), figsize=(16, 8))
        f.suptitle("Detected Outliers", fontsize=16)
        for i, (outlier_img, label) in enumerate(outliers):
            axes[i].imshow(outlier_img, cmap='gray')
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        plt.savefig(f"{save_dir}/outliers.png", bbox_inches="tight")
        print(f"Outliers saved to '{save_dir}/outliers.png'.")
    else:
        print("No outliers detected.")



def latent_space_analysis(generator, train_loader, img_dim, save_dir):
    """Analyze learned latent space representations, center them, and save plots per class with consistent colors."""
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Class names for FashionMNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    latent_vectors = []
    labels_list = []

    # Extract latent representations from the Generator
    for idx, (real_img, label) in enumerate(train_loader):
        if idx >= 1000:  # Limit to 1000 samples for visualization
            break
        noise = torch.randn(1, img_dim)  # Random noise
        with torch.no_grad():
            latent_rep = generator(noise, label)  # Extract latent representations
            latent_vectors.append(latent_rep.squeeze().cpu().numpy())
            labels_list.append(label.item())

    # Convert to numpy arrays
    latent_vectors = np.array(latent_vectors)
    labels_list = np.array(labels_list)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    reduced_latent_vectors = pca.fit_transform(latent_vectors)

    # Center the data
    reduced_latent_vectors -= reduced_latent_vectors.mean(axis=0)

    # Define a consistent colormap
    colormap = plt.cm.get_cmap("tab10")

    # General Visualization: All Classes Together
    plt.figure(figsize=(10, 10))
    plt.title("Learned Latent Space Representations  with Class Names", fontsize=16)
    scatter = plt.scatter(reduced_latent_vectors[:, 0], reduced_latent_vectors[:, 1],
                          c=labels_list, cmap=colormap, s=20, alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels(class_names)
    cbar.set_label("Class Names")
    plt.xlabel("Principal Component 1 ")
    plt.ylabel("Principal Component 2 ")
    plt.grid(True)
    plt.savefig(f"{save_dir}/latent_space_with_class_names.png", bbox_inches="tight")
    plt.close()
    print(f"General latent space analysis saved to '{save_dir}/latent_space_with_class_names.png'.")

    # Save Individual Class Plots with Consistent Colors
    class_folder = os.path.join(save_dir, "individual_classes")
    os.makedirs(class_folder, exist_ok=True)

    for label_idx, class_name in enumerate(class_names):
        # Filter points for the current label
        mask = (labels_list == label_idx)
        class_vectors = reduced_latent_vectors[mask]

        # Sanitize the class name for safe filenames
        safe_class_name = re.sub(r'[^\w\-_.]', '_', class_name)

        # Plot for the current class with consistent colors
        plt.figure(figsize=(8, 8))
        plt.title(f"Latent Space for Class: {class_name} ", fontsize=14)
        plt.scatter(class_vectors[:, 0], class_vectors[:, 1], color=colormap(label_idx), s=20, alpha=0.8)
        plt.xlabel("Principal Component 1 ")
        plt.ylabel("Principal Component 2 ")
        plt.grid(True)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        # Save the plot
        plt.savefig(os.path.join(class_folder, f"latent_space_{safe_class_name}.png"), bbox_inches="tight")
        plt.close()
        print(f"Saved latent space plot for class '{class_name}' to '{class_folder}/latent_space_{safe_class_name}.png'.")






def extract_timestamp(model_path):
    """Extract timestamp from the model path."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', model_path)
    if match:
        return match.group(1)
    return "unknown_timestamp"

def reconstruct_test_data(generator, test_loader, img_dim, save_dir):
    """Reconstruct test data and compare with real images."""
    generator.eval()
    reconstructed_images = []
    real_images = []

    os.makedirs(save_dir, exist_ok=True)

    for idx, (real_img, label) in enumerate(test_loader):
        if idx >= 10:  # Limit to 10 samples for visualization
            break
        noise = torch.randn(1, img_dim)
        label_tensor = torch.LongTensor([label])
        with torch.inference_mode():
            gen_img = generator(noise, label_tensor)
            gen_img = gen_img.squeeze().reshape(28, 28).to('cpu').detach().numpy()
            reconstructed_images.append(gen_img)
            real_images.append(real_img.squeeze().numpy())

    # Visualization: Real vs Reconstructed
    f, axes = plt.subplots(2, len(reconstructed_images), figsize=(16, 8))
    for i in range(len(reconstructed_images)):
        axes[0, i].imshow(real_images[i], cmap='gray')
        axes[0, i].set_title("Real")
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.savefig(f"{save_dir}/reconstructed_images.png", bbox_inches="tight")
    print(f"Reconstructed images saved to '{save_dir}/reconstructed_images.png'.")

def analyze_diversity(generator, img_dim, save_dir):
    """Analyze diversity of generated images for each label."""
    generator.eval()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    f = plt.figure(figsize=(16, 16))

    os.makedirs(save_dir, exist_ok=True)

    for label in range(10):
        for i in range(10):  # Generate 10 images per label
            noise = torch.randn(1, img_dim)
            label_tensor = torch.LongTensor([label])
            with torch.inference_mode():
                gen_img = generator(noise, label_tensor)
                gen_img = gen_img.squeeze().reshape(28, 28).to('cpu').detach().numpy()
                ax = f.add_subplot(10, 10, label * 10 + i + 1)
                ax.imshow(gen_img, cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_title(class_names[label])
    plt.savefig(f"{save_dir}/diversity_images.png", bbox_inches="tight")
    print(f"Diversity images saved to '{save_dir}/diversity_images.png'.")

def prepare_test_data_loader(batch_size=1, training = False):
    """Prepare the test dataset loader."""
    # transform = get_transform()
    if not training:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return loader
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        test_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return loader

def generate_sample(model_path, img_dim=100, save_dir="./samples", was_data_parallel=False):
    """Generate samples for visualization."""
    generator_loaded = G(latent_dim=img_dim)
    generator_loaded = load_models(generator_loaded, model_path=model_path, was_data_parallel=was_data_parallel)

    noise_test = torch.randn(1, img_dim)  # Adjusted for correct shape
    label_list = [torch.LongTensor([x]) for x in range(10)]

    generator_loaded.eval()
    gan_img_test_list = []

    for x in range(10):
        with torch.inference_mode():
            gan_img_test = generator_loaded(noise_test, label_list[x])
            gan_img_test = gan_img_test.squeeze().reshape(28, 28).to('cpu').detach().numpy()
            gan_img_test_list.append(gan_img_test)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    f = plt.figure(figsize=(16, 16))

    for x in range(10):
        ax = f.add_subplot(1, 10, x + 1)
        ax.imshow(gan_img_test_list[x], cmap='gray')
        ax.set_title(f"{class_names[x]}")
        ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/generated_samples.png", bbox_inches="tight")
    print(f"Generated samples saved to '{save_dir}/generated_samples.png'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    parser.add_argument("--ts", type=str,required=True ,default="2024-12-11_16:59:05", help="Timestamp identifier for the model.")
    parser.add_argument("--data_parallel", type=bool, default=True, help="Whether the model was trained using nn.DataParallel.")
    args = parser.parse_args()

    print(f"Using latent dimension: {args.latent_dim}")
    test_loader = prepare_test_data_loader()
    train_loader = prepare_test_data_loader(training=True)

    model_path = f"./checkpoints/{args.ts}/generator_{args.ts}.pth"
    save_dir = f"./samples/{args.ts}"
    generator = G(latent_dim=args.latent_dim)
    generator = load_models(generator, model_path=model_path, was_data_parallel=args.data_parallel)
    discriminator = D()
    discriminator = load_models(discriminator, model_path=f"./checkpoints/{args.ts}/discriminator_{args.ts}.pth", was_data_parallel=args.data_parallel)

    # Generate samples for visualization
    generate_sample(model_path=model_path, img_dim=args.latent_dim, 
                    save_dir=save_dir, was_data_parallel=args.data_parallel)

    # Reconstruct test data and compare with real images
    reconstruct_test_data(generator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)

    # Analyze diversity of generated images
    analyze_diversity(generator, img_dim=args.latent_dim, save_dir=save_dir)
    
    Latent_Analysis_G = G_LSA(latent_dim=args.latent_dim)
    Latent_Analysis_G = load_and_transfer_generator_layers(model_path, G_LSA, device=torch.device('cpu'))

    # Perform latent space analysis with the training set
    latent_space_analysis(Latent_Analysis_G, train_loader, img_dim=args.latent_dim, save_dir=save_dir)

    # Perform latent interpolation
    print("Latent Space Interpolation:")
    random_interpolate_labels(generator, save_dir=save_dir, num_rows=6, img_dim=100, device='cpu')

    # Evaluate Discriminator performance
    print("Evaluating Discriminator Performance:")
    test_discriminator(discriminator, generator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)



if __name__ == "__main__":
    main()
