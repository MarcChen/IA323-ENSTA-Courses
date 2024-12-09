# Fashion MNIST Conditional GAN

This repository implements a **Conditional Generative Adversarial Network (GAN)** trained on the Fashion MNIST dataset. The purpose of this project is twofold:

1. **Modeling & Generation**: Build a Conditional GAN that learns to produce high-quality images of fashion items conditioned on their class labels (e.g., shirts, shoes, bags).  
2. **Data Analysis & Insights**: Use data science techniques to analyze the generated images, uncovering patterns, biases, or trends that emerge when conditioning on different classes.

## Key Features

- **Conditional GAN Architecture**: Implements a generator and discriminator designed to incorporate class label information, ensuring that the generated images correspond to specific fashion categories.
- **Comprehensive Training Pipeline**: Includes data loading, model definition, training scripts, and evaluation metrics.
- **Visualization & Analysis**: Offers notebooks and scripts to visualize generated samples, track training progress, and derive insights from the model’s outputs.
- **Reproducible Setup**: Provides clear instructions and environment configurations for easy reproduction of results.

## Goals

1. **Understand Conditional Image Generation**: Explore how conditioning on class labels influences the diversity and realism of generated images.
2. **Evaluate Model Performance**: Employ metrics like FID or qualitative assessments to measure how closely generated samples resemble real data.
3. **Data-Driven Insights**: Apply clustering, dimensionality reduction, and other data analysis techniques to the generated dataset to find interesting patterns and relationships.

## Getting Started

1. **Setup**: Make the setup script executable and run it to install dependencies and download the dataset:
    ```sh
    chmod +x ./setup.sh
    ./setup.sh
    ```
2. **Train the Model**: Run the training scripts to fit the Conditional GAN on the Fashion MNIST dataset:
    ```sh
    python3 src/train.py
    ```
   parameters:
   - `--epochs`: Number of training epochs (default: 100).
   - `--batch_size`: Batch size for training (default: 64).
   - `--latent_dim`: Dimensionality of the latent space (default: 100).
   - `--lr`: Learning rate for the optimizer (default: 0.0002).
   - `--data_dir`: Directory to store the Fashion MNIST dataset (default: './data').
   - `--save_dir`: Directory to save model checkpoints and generated samples (default: './outputs').
   - `--save-name`: Name of the model checkpoint file.

3. **Generate Samples**: Use the trained model to generate samples for each class label:
    ```sh
    python3 src/generate.py
    ```
    parameters:
    - `--num_samples`: Number of samples to generate for each class (default: 10).
    - `--save_path`: Directory to save the generated samples (default: './outputs').
    - `--generator_path`: Path to the trained generator model.
    - `--discriminator_path`: Path to the trained discriminator model.
    - `--latent_dim`: Dimensionality of the latent space (default: 100).

By engaging with this repository, you will gain practical experience with conditional generative modeling and learn how to leverage data science methods to deepen your understanding of model-generated outputs.
