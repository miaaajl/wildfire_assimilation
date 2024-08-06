import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class VectorQuantizerEMA(nn.Module):
    
    def __init__(self, embedding_dim, num_embeddings, beta=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        """ 
        Class for creating a vector quantiser with exponential moving averages.
        
        Parameters
        ----------
        
        embedding_dim: int
            Dimensionality of the latent space 
             
        num_embeddings: int
            Number of embeddings 
             
        beta: float
            Weighting factor for the commitment loss
            
        decay: float
            Decay rate for exponential moving averages
            
        epsilon: float
            Small value added to avoid division by zero
            
            
        Attributes 
        --------
            
        embedding: torch.nn.Parameter
            Learnable parameter representing the codebook
            
        _ema_cluster_size: torch.tensor
            Buffer to store cluster size for exponential moving average
            
        _ema_w: torch.nn.Parameter
            Exponential moving average of the codebook
            
        """
            
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Parameter(torch.randn(embedding_dim, num_embeddings))
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(self.embedding.clone(), requires_grad=False)

    def forward(self, x):
        
        """ 
        Runs a forward pass of the Vector Quantizer.
        
        Parameters
        ----------
        
        x: torch.tensor
            The input tensor of shape (batch_size, channels, height, width)
            
        
        Returns
        -------
        
        quantized: torch.tensor
            Quantized latent space representation
            
            
        loss: torch.tensor
            Total loss composed of reconstruction and commitment loss
            
        perplexity: torch.tensor
            Perplexity of the encoding distribution 
            
        encodings: torch.tensor
            One-hot encodings of the nearest embeddings 
            
        encoding_indices (torch.Tensor): Indices of the nearest embeddings
        
        
        """
        
        flatten = x.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flatten ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=0)
            - 2 * torch.matmul(flatten, self.embedding)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        quantized = torch.matmul(encodings, self.embedding.t()).view(x.shape)
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            dw = torch.matmul(flatten.t(), encodings)
            self._ema_w.data = self._ema_w.data * self.decay + (1 - self.decay) * dw

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(0)

        loss = self.beta * e_latent_loss
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encodings, encoding_indices

class Residual1D(nn.Module):
    
    """ 1D Residual Block for VQ-VAE """
    
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Residual1D, self).__init__()
        
        """        
        
        Parameters
        ----------
    
        in_channels: int
        Number of input channels
        
        hidden_channels: int
        Number of hidden channels
        
        out_channels: int
        Number of output channels 
        
        
        Attributes 
        --------
        
        conv1: torch.nn.Conv2d
            First convolutional layer 
            
        conv2: torch.nn.Conv2d
            Second convolutional layer 
        
        """
        
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        
        """
        Runs a forward pass of the 1D Residual Block
        
        Parameters
        ----------
        
        x: torch.tensor
            The input tensor.
            
        Returns
        -------
        
        out:
             The output tensor run through the Residual1D framework
        
        """
        
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

class Encoder(nn.Module):
    
    """The Encoder part of the VQ-VAE network."""
    
    def __init__(self, out_channel):
        super(Encoder, self).__init__()
        
        """
        Parameters
        ----------
        
        out_channel: int
            Number of output channels 
            
        """
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        
        """
        Runs a forward pass of the encoder network.
        
        Parameters
        ----------
        
        x: torch.tensor
            Input tensor
            
        Returns
        -------
        
        x: torch.tensor
            Output tensor run through the encoder network
            
        """
  
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    
    """The Decoder part of the VQ-VAE network."""
    
    def __init__(self, out_channel):
        
        """
        Parameters
        ----------
        
        out_channel: int
            The number of output channels at the end of the decompression
            
        """
        
        super(Decoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(out_channel, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        
        """
        Runs a forward pass of the Decoder network.
        
        Parameters
        ----------
        
        x: torch.tensor
            Input tensor 
            
        Returns 
        -------
        x: torch.tensor
            The output tensor
        
        """
        
        x = torch.relu(self.conv_trans1(x))
        x = torch.relu(self.conv_trans2(x))
        x = torch.tanh(self.conv_trans3(x)) * 0.5
        return x

class VQVAE(nn.Module):
    
    """The full Variational Quantized Variational Autoencoder (VQ-VAE) framework."""
    
    def __init__(self, embedding_dim=8, num_embeddings=32):
        super(VQVAE, self).__init__()

        
        """
        
        Parameters
        ----------
        
        embedding_dim: int
            The dimensionality of the latent space
            
        num_embeddings: int
            Number of embeddings (codebook size)
            
        """
        
        self.encoder = Encoder(embedding_dim)
        self.vector_quantizer = VectorQuantizerEMA(embedding_dim, num_embeddings)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        
        """
        Runs a forward pass of the VQ-VAE
        
        Parameters
        ----------
        
        x: torch.tensor
            Input tensor.
            
        Returns
        -------
        
        x_recon: torch.tensor
            Reconstructed input
            
        loss: torch.tensor
            Total loss
            
        perplexity: torch.tensor 
            Perplexity of the encoding distribution
        
        encodings: torch.tensor
            One-hot encodings of the nearest embeddings 
            
        encoding_indices: torch.tensor
            Indices of the nearest embeddings 
            
        """
        
        z = self.encoder(x)
        quantized, loss, perplexity, encodings, encoding_indices = self.vector_quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, loss, perplexity, encodings, encoding_indices



def train_vq_vae(model, train_dataset, optimizer, criterion, data_variance, n_epoch=2, model_filepath='vqvae_model.pth'):
    
    """
    Trains the VQ-VAE model
    
    Parameters
    ----------
    
    model: VQ-VAE
        The VQ-VAE model to be trained.
        
    train_dataset: torch.utils.data.Dataset:
        The training dataset
        
    optimizer: torch.optim.Optimizer
        The optimizer for training
     
    criterion: torch.nn.Module
        The loss function for training
        
    data_variance: float
        Variance of the data
        
    n_epoch: int
        Number of epochs for training 
        
    model_filepath: str
        Filepath to save the trained model
        
    """
    
    # Training
    n_epoch = 10
    train_res_recon_error = []
    train_res_perplexity = []
    vqloss = []

    for epoch in range(n_epoch):
        for i, (Xbatch,) in enumerate(train_dataset):
            optimizer.zero_grad()
            x_recon, loss, perplexity, encodings, encoding_indices = model(Xbatch)
            recon_error = criterion(x_recon, Xbatch) / data_variance
            total_loss = recon_error + loss
            total_loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
            vqloss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{n_epoch}, Step {i + 1}, Recon Error: {recon_error.item()}, VQ Loss: {loss.item()}, Perplexity: {perplexity.item()}')

    # Save model
    torch.save(model.state_dict(), model_filepath)

def predict(model, num_samples, device='cpu'):
    
    """
    Generates samples from the trained VQ-VAE model.
    
    Parameters
    ----------
    
    model: VQ-VAE
        The trained VQ-VAE model
    
    num_samples: int
        Number of samples to generate
        
    device: str
        Device to run inference on ('cpu' or 'cuda')
        
        
    Returns
    -------
    
    generated_samples: torch.tensor
        Generated samples 
        
    """
    
    model.eval()  # Set the model to evaluation mode

    # Generate random latent vectors
    embedding_dim = model.vector_quantizer.embedding_dim
    num_embeddings = model.vector_quantizer.num_embeddings
    latent_samples = torch.randn(num_samples, embedding_dim, 1, 1).to(device)
    
    # Find the nearest embeddings for the latent vectors
    flatten = latent_samples.view(-1, embedding_dim)
    distances = (
        torch.sum(flatten ** 2, dim=1, keepdim=True)
        + torch.sum(model.vector_quantizer.embedding ** 2, dim=0)
        - 2 * torch.matmul(flatten, model.vector_quantizer.embedding)
    )
    encoding_indices = torch.argmin(distances, dim=1)
    encodings = torch.zeros(encoding_indices.shape[0], num_embeddings, device=device)
    encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

    quantized = torch.matmul(encodings, model.vector_quantizer.embedding.t()).view(latent_samples.shape)

    # Decode the quantized latent vectors to generate samples
    with torch.no_grad():
        generated_samples = model.decoder(quantized)

    return generated_samples

def plot_samples(samples, num_samples=10):
    
    """ 
    Plot generated samples.
    
    Parameters
    ----------
    
    samples: torch.tensor
        Generated samples.
        
    num_samples: int
        Number of samples to plot.
        
    """
    
    # Convert samples to numpy
    samples_np = samples.cpu().numpy()
    
    # Determine the grid size for plotting
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create a figure for the samples
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        if i < samples_np.shape[0]:
            axes[i].imshow(samples_np[i, 0], cmap='gray')  # Assuming single-channel (grayscale) images
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

