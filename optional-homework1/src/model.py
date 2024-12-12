import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_size=28 * 28, class_label_size=10, latent_dim=100, dropout_prob=0.0):
        super().__init__()

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        # Define the layer function for modularity
        def layer(input_dim, output_dim, dropout_prob=0.0):
            layers = [nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2, inplace=True)]
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            return layers

        # Build the model using the layer function
        self.model = nn.Sequential(
            *layer(latent_dim + class_label_size, 256, dropout_prob),
            *layer(256, 512, dropout_prob),
            *layer(512, 1024, dropout_prob),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)  # Assuming latent space dimension is 100
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)  # Assuming output image size is 28x28

class Discriminator(nn.Module):
    def __init__(self, image_size=28 * 28, class_label_size=10, dropout_prob=0.3):
        super().__init__()

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        # Define the layer function for modularity
        def layer(input_dim, output_dim, dropout_prob=0.0, activation=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if activation:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            return layers

        # Build the model using the layer function
        self.model = nn.Sequential(
            *layer(image_size + class_label_size, 1024, dropout_prob),
            *layer(1024, 512, dropout_prob),
            *layer(512, 256, dropout_prob),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a single probability value
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)  # Flatten the input image
        c = self.label_emb(labels)  # Embed the labels
        x = torch.cat([x, c], 1)  # Concatenate input image and label embeddings
        out = self.model(x)  # Pass through the model
        return out.squeeze()  # Remove extra dimensions for compatibility

class GeneratorV2(nn.Module):
    def __init__(self, image_size=28, class_label_size=10, latent_dim=100):
        super().__init__()
        self.label_emb = nn.Embedding(class_label_size, latent_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + class_label_size, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1).unsqueeze(2).unsqueeze(3)  # Add spatial dimensions
        out = self.model(x)
        return out

class DiscriminatorV2(nn.Module):
    def __init__(self, image_size=28, class_label_size=10):
        super().__init__()
        self.label_emb = nn.Embedding(class_label_size, image_size * image_size)

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(labels.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)  # Concatenate image and label embeddings
        out = self.model(x)
        return out.squeeze()

img_dim = 100
BATCH_SIZE = 32
Image_size=28*28
class_label_size=10

class GeneratorV3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(class_label_size, class_label_size)
        
        self.model = nn.Sequential(
            nn.Linear(img_dim+class_label_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, Image_size),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
    
class DiscriminatorV3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(class_label_size, class_label_size)
        
        self.model = nn.Sequential(
            nn.Linear(Image_size+class_label_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()