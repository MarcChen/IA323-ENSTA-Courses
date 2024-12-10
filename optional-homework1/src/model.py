import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, image_size = 28 * 28, class_label_size = 10, latent_dim = 100, dropout_prob=0.0):
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

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_size = 28 * 28, class_label_size = 10, dropout_prob=0.3):
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
    

    
