import os
from torchvision import datasets, transforms

def get_transform():
    # Normalization of dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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

    # Check if the dataset is already downloaded
    if os.path.exists(os.path.join(path, 'FashionMNIST', 'raw')):
        print("Dataset already downloaded in data folder !")
        return

    # Normalization of dataset
    transform = get_transform()

    # Download the dataset and store it in the path
    datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
    datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)

    return

if __name__ == "__main__":
    load_dataset()