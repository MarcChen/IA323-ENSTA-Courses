import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, class_label_size=10, Image_size=784):
        super().__init__()

        self.latent_dim = latent_dim

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + class_label_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, Image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), self.latent_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

class Discriminator(nn.Module):
    def __init__(self, class_label_size=10, Image_size=784, dropout_prob=0.3):
        super().__init__()

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        self.model = nn.Sequential(
            nn.Linear(Image_size + class_label_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
    

class Generator_For_LatentSpaceAnalysis(nn.Module):
    def __init__(self, latent_dim=100, class_label_size=10, Image_size=784):
        super().__init__()

        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        self.fc1 = nn.Linear(self.latent_dim + class_label_size, 256)  # First layer
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, Image_size),
            nn.Tanh()
        )

    def forward(self, z, labels, return_latent=False):
        z = z.view(z.size(0), self.latent_dim)  # Reshape noise
        c = self.label_emb(labels)  # Label embedding
        x = torch.cat([z, c], 1)  # Concatenate noise and labels

        # Compute latent representation after first layer
        latent_rep = self.fc1(x)
        latent_rep = self.leaky_relu1(latent_rep)

        if return_latent:
            return latent_rep  # Return latent representation
        
        # Pass through the rest of the model
        out = self.model(latent_rep)
        return out.view(x.size(0), 28, 28)
