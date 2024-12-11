import argparse
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator
import re

def load_models(model, model_path="./checkpoints/generator.pth"):
    """Load the model weights."""
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Checkpoint keys:", checkpoint.keys())
        print("Model state_dict keys:", model.state_dict().keys())
    return model

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

def prepare_test_data_loader(batch_size=1):
    """Prepare the test dataset loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def generate_sample(model_path, img_dim=100, dropout_prob_generator=0.3, save_dir="./samples"):
    """Generate samples for visualization."""
    generator_loaded = Generator(latent_dim=img_dim, dropout_prob=dropout_prob_generator)
    generator_loaded = load_models(generator_loaded, model_path=model_path)

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
    parser.add_argument("--ts", type=str, default="2024-12-11_16:59:05", help="Timestamp identifier for the model.")
    parser.add_argument("--dropout_prob_generator", type=float, default=0.0, help="Dropout probability for the generator.")
    args = parser.parse_args()

    print(f"Using latent dimension: {args.latent_dim}")
    test_loader = prepare_test_data_loader()

    model_path = f"./checkpoints/{args.ts}/generator_{args.ts}.pth"
    save_dir = f"./samples/{args.ts}"
    generator = Generator(latent_dim=args.latent_dim, dropout_prob=args.dropout_prob_generator)
    generator = load_models(generator, model_path=model_path)

    # Generate samples for visualization
    generate_sample(model_path=model_path, img_dim=args.latent_dim, 
                    dropout_prob_generator=args.dropout_prob_generator, save_dir=save_dir)

    # Reconstruct test data and compare with real images
    reconstruct_test_data(generator, test_loader, img_dim=args.latent_dim, save_dir=save_dir)

    # Analyze diversity of generated images
    analyze_diversity(generator, img_dim=args.latent_dim, save_dir=save_dir)

if __name__ == "__main__":
    main()
