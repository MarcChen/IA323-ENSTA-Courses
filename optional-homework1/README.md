# GAN Framework for FashionMNIST Dataset

This repository provides a framework for training and analyzing Generative Adversarial Networks (GANs) on the FashionMNIST dataset. It includes functionality for training a GAN, evaluating its performance, and performing advanced latent space analyses.

---

## Features

### 1. **Generator and Discriminator Models**
- **Generator**: Generates FashionMNIST-like images conditioned on class labels.
- **Discriminator**: Evaluates the authenticity of the images (real or fake).

### 2. **Latent Space Analysis**
- Extract and analyze the latent space representations.
- Perform latent space interpolation between two classes.
- Visualize latent space clustering using PCA.

### 3. **Evaluation Tools**
- Test discriminator accuracy on real and fake samples.
- Visualize accuracy using pie charts.
- Reconstruct real images from latent vectors.
- Analyze diversity of generated images.

### 4. **Training Utilities**
- Implements custom training loops for the generator and discriminator.
- Supports adjustable hyperparameters such as batch size, learning rate, and latent dimensions.
- Saves best-performing models automatically.

### 5. **Visualization**
- Generate and save plots for latent space representations.
- Visualize real vs. generated images.
- Analyze diversity and reconstruction results.

---

# Getting Started

## 1. **Setup**

Make the setup script executable and run it to install dependencies and download the dataset:

```sh
chmod +x ./setup.sh
./setup.sh
```
The setup script will:
1. Download the dataset to the `./data` directory.
2. Install the required Python packages using `pip` as specified in the `requirements.txt` file.

---

## 2. **Training the GAN**

To train the GAN, use the following command:

```bash
python main.py \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0001 \
    --latent-dim 100 \
    --times_train_discriminator 5 \
    --dropout_prob_discriminator 0.3 \
    --data_augmentation False
```

### **Arguments:**
- `--batch-size`: Number of samples per training batch (default: `32`).
- `--epochs`: Number of epochs for training (default: `100`).
- `--lr`: Learning rate for both generator and discriminator optimizers (default: `0.0001`).
- `--latent-dim`: Dimensionality of the latent noise vector (default: `100`).
- `--times_train_discriminator`: Number of iterations for discriminator training per generator iteration (default: `5`).
- `--dropout_prob_discriminator`: Dropout probability for the discriminator (default: `0.3`).
- `--data_augmentation`: Boolean flag to enable/disable data augmentation (default: `False`).

---

## 3. **Generating Samples**

After training, generate samples for visualization:

```bash
python main.py \
    --ts <timestamp> \
    --latent-dim 100 \
    --data_parallel True
```

### **Arguments:**
- `--ts`: Timestamp identifier of the saved model (e.g., `2024-12-11_16:59:05`).
- `--data_parallel`: Boolean flag to specify if the model was trained with `nn.DataParallel` (default: `True`).

---

# Repository Organization

The repository is structured as follows:

- **`main.py`**: Entry point for training, evaluation, and analysis.
- **`model.py`**: Contains the implementations of the Generator and Discriminator models.
- **`generate.py`**: Includes functions for visualization, latent space analysis, and diversity analysis.
- **`data_loading.py`**: Utilities for loading the FashionMNIST dataset.
- **`setup.sh`**: Script to install dependencies and download the dataset.
- **`checkpoints/`**: Directory for saving trained models.
- **`samples/`**: Directory for saving generated images and analysis results.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: Documentation and usage instructions for the repository.

# Further Analysis

For a detailed analysis of the results, refer to the [Results Analysis](./results_analysis.md) document.

