import os
from requests import get
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

def load_mean_std(file_path):
    """
    Load the mean and standard deviation from a file.

    Parameters:
    - file_path (str): The file path to load the computed values from.

    Returns:
    - mean (float): The loaded mean value.
    - std (float): The loaded standard deviation value.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Please compute and save mean/std first.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data["mean"], data["std"]


def compute_and_save_mean_std(dataset, file_path):
    """
    Compute the mean and standard deviation of a dataset and save them to a file.

    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset for which to compute mean and std.
    - file_path (str): The file path to save the computed values.
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

    # Save the computed mean and std to a file
    with open(file_path, 'w') as f:
        json.dump({"mean": mean.item(), "std": std.item()}, f)

    print(f"Mean and std saved to {file_path}")
    return mean.item(), std.item()


def get_transform(train=False):
    """
    Create a transformation pipeline with optional data augmentation and normalization.

    Parameters:
    - train (bool): If True, include data augmentation suitable for training.

    Returns:
    - transform (torchvision.transforms.Compose): Composed transformation pipeline.
    """
    try:
        mean, std = load_mean_std(os.path.join(os.path.dirname(__file__), '..', 'data', 'mean_std.json'))
    except FileNotFoundError:
        mean, std = 0.5, 0.5
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    return transform



def load_dataset(path=None):
    """
    This function loads the Fashion MNIST dataset, checks if it is already downloaded in the specified path,
    and if not, downloads it. The dataset is normalized and stored in the given path.

    Parameters:
    - path (str): The directory where the dataset should be stored. Default is "../data".

    """
    if path is None:
        # Get the absolute path of the project's root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path = os.path.join(project_root, 'data')

    # # Check if the dataset is already downloaded
    # if os.path.exists(os.path.join(path, 'FashionMNIST', 'raw')):
    #     print("Dataset already downloaded in data folder !")
    #     return

    # Download the dataset and store it in the path
    train_dataset = datasets.FashionMNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    datasets.FashionMNIST(root=path, train=False, download=True)

    # Compute and save mean and std
    mean, std = compute_and_save_mean_std(train_dataset, os.path.join(path, 'mean_std.json'))

    # Print the computed mean and std
    print(f"Computed mean: {mean}, std: {std}")

    return

if __name__ == "__main__":
    load_dataset()