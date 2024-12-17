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
    def __init__(self, latent_dim=100, class_label_size=10):
        super().__init__()

        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        # First layer
        self.fc1 = nn.Linear(self.latent_dim + class_label_size, 256)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z, labels, output_before_relu=False):
        z = z.view(z.size(0), self.latent_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)

        # Raw output of fc1
        latent_raw = self.fc1(x)

        if output_before_relu:
            return latent_raw  # Return before activation
        
        # Activated output
        latent_activated = self.leaky_relu1(latent_raw)
        return latent_activated  # Return after activation

        
    