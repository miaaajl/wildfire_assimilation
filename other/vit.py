import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from tqdm import tqdm

class WildfireDatasetViT(Dataset):
    """
    Wildfire dataset for training Vision Transformer (ViT) model.
    """
    def __init__(self, image_array, transform=None):
        """
        Args:
            image_array (numpy array): Shape (num_images, height, width)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_array = image_array.astype(np.float32)  # Ensure the array is of type float32
        self.transform = transform

        # Ensure the array has at least 2 images to form pairs
        if len(self.image_array) < 2:
            raise ValueError("image_array must contain at least 2 images to form pairs")

    def __len__(self):
        # The number of pairs is one less than the number of images
        return len(self.image_array) - 1

    def __getitem__(self, idx):
        image1 = self.image_array[idx]
        image2 = self.image_array[idx + 1]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        else:
            image1 = torch.from_numpy(image1).unsqueeze(0)  # Add channel dimension
            image2 = torch.from_numpy(image2).unsqueeze(0)  # Add channel dimension
        return image1, image2
    

class PatchEmbedding(nn.Module):
    """
    Patch Embedding module for Vision Transformer.
    """
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=256):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, emb_size, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, emb_size)
        x += self.pos_embed
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module for Vision Transformer.
    """
    def __init__(self, emb_size=768, num_heads=12, depth=12, mlp_dim=3072, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size, 
                nhead=num_heads, 
                dim_feedforward=mlp_dim, 
                dropout=dropout
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TemporalViT(nn.Module):
    """
    Vision Transformer for temporal data.
    """
    def __init__(self, img_size=256, patch_size=16, emb_size=768, num_heads=12, depth=12, mlp_dim=3072, dropout=0.1):
        super(TemporalViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels=1, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        self.transformer_encoder = TransformerEncoder(emb_size=emb_size, num_heads=num_heads, depth=depth, mlp_dim=mlp_dim, dropout=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, img_size * img_size)  # Output size for single channel
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]  # Extracting CLS token
        x = self.mlp_head(x)
        x = x.view(x.size(0), 1, 256, 256)  # Assuming output image size 256x256
        return x

def train_vit(model, dataloader, optimizer, criterion, num_epochs=10, patience=5, model_save_path='best_vit_model.pth'):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (image1, image2) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(image1)
            loss = criterion(output, image2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
