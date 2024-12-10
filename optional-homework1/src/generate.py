import argparse
import os
import torch
from model import Generator
import matplotlib.pyplot as plt
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

def generate_sample(model_path, img_dim=100, dropout_prob_generator=0.3):
    # Load the generator model
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
    
    os.makedirs("./samples", exist_ok=True)
    timestamp = extract_timestamp(model_path)
    print(model_path)
    plt.savefig(f"./samples/generated_samples_{timestamp}.png", bbox_inches="tight")
    print(f"Generated samples saved to './samples/generated_samples_{timestamp}.png'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    parser.add_argument("--model-name", type=str, default="generator.pth", help="Name of the generator to load (e.g name + timestamp).")
    parser.add_argument("--dropout_prob_generator", type=float, default=0.0, help="Dropout probability for the generator.")
    args = parser.parse_args()

    print(f"Using latent dimension: {args.latent_dim}")
    generate_sample(img_dim=args.latent_dim, 
                    model_path=f"./checkpoints/{args.model_name}",
                    dropout_prob_generator=args.dropout_prob_generator)
    

if __name__ == "__main__":
    main()
