import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class Convolutional_VAE_Encoder(nn.Module):
    
  def __init__(self, device, input_dim, hidden_dims, compressed_dim, activation="RELU"):
    super(Convolutional_VAE_Encoder, self).__init__()

    """

    Parameters
    ----------

    device: str
        The device (CPU/GPU) on which the VAE architecture should be stored. 

    input_dim: int
        The size of the input to the VAE (should correspond to the size of a flattened set of images

    hidden_dims: list
        The sizes of the hidden dimensions of the VAE

    compressed_dim: int
        The size of the latent space

    activation: str, optional 
        The activation function that should be applied to each layer of the VAE. 
    
    """
    
    if activation == "RELU":
      self.activate = nn.ReLU()

    if activation == "ELU":
      self.activate = nn.ELU()

    if activation == "GELU":
      self.activate = nn.GELU()

    self.layer0 = nn.Sequential(
        nn.Conv2d(1, hidden_dims[0], 3, padding=3),
        self.activate,
        nn.MaxPool2d(2)
        )

    self.layer1 = nn.Sequential(
        nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=3),
        self.activate,
        nn.MaxPool2d(2)
        )

    self.layerMu = nn.Conv2d(hidden_dims[1], compressed_dim, 4)

    self.layerSigma = nn.Conv2d(hidden_dims[1], compressed_dim, 4)

  def forward(self, x):

    """
    Encodes the input images into the latent space.

    Parameters 
    ----------

    x: ___
        The batch of images to be compressed 

    Returns 
    -------
    self.fc41(h3), self.fc42(h3): ____
         The 'mu' and 'sigma' of the compressed latent space. 
    
    """
    
    x = self.layer0(x)
    x = self.layer1(x)
    mu = self.layerMu(x)
    sigma = self.layerSigma(x)

    return mu, sigma

class Convolutional_VAE_Decoder(nn.Module):
    
  def __init__(self, device, compressed_dim, hidden_dims, input_dim, activation="RELU"):
    super(Convolutional_VAE_Decoder, self).__init__()

    """

    Parameters
    ----------

    device: str
        The device (CPU/GPU) on which the VAE architecture should be stored. 

    input_dim: int
        The size of the input to the VAE (should correspond to the size of a flattened set of images

    hidden_dims: list
        The sizes of the hidden dimensions of the VAE

    compressed_dim: int
        The size of the latent space

    activation: str, optional 
        The activation function that should be applied to each layer of the VAE. 
    
    """

    if activation == "RELU":
      self.activate = nn.ReLU()

    if activation == "ELU":
      self.activate = nn.ELU()

    if activation == "GELU":
      self.activate = nn.GELU()

    self.layerLatent = nn.Sequential(
        nn.ConvTranspose2d(compressed_dim, hidden_dims[1], 2),
        self.activate,
    )

    self.layer1 = nn.Sequential(
        nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 2, padding=1),
        self.activate,
        nn.Upsample(scale_factor=2, mode='nearest')
    )

    self.layerOut = nn.Sequential(
        nn.Conv2d(hidden_dims[0], 1, 3, padding=1),
        nn.Upsample(scale_factor = 2, mode='nearest'),
        nn.Conv2d(1,1,3,padding=1),
        nn.Sigmoid()
    )

  def forward(self, z):

    """
    Decompresses the latent space into a reconstructed image.

    Parameters
    ----------

    z = ___
        The random sample obtained from the latent space. 

    Returns 
    -------

    torch.sigmoid(self.fc8(h7): ____
        The reconstructed batch of images

    """
      
    z = self.layerLatent(z)
    z = self.layer1(z)
    z = self.layerOut(z)
    return z
      

class Convolutional_VAE(nn.Module):
  def __init__(self, device, input_dim, hidden_dims, compressed_dim, activation="RELU"):

    """

    Parameters
    ----------

    device: str
        The device (CPU/GPU) on which the VAE architecture should be stored. 

    input_dim: int
        The size of the input to the VAE (should correspond to the size of a flattened set of images

    hidden_dims: list
        The sizes of the hidden dimensions of the VAE

    compressed_dim: int
        The size of the latent space

    activation: str, optional 
        The activation function that should be applied to each layer of the VAE. 
    
    """    

    super(Convolutional_VAE, self).__init__()
    self.device = device
    self.encoder = Convolutional_VAE_Encoder(device, input_dim, hidden_dims, compressed_dim, activation=activation)
    self.decoder = Convolutional_VAE_Decoder(device, compressed_dim, hidden_dims, input_dim, activation=activation)
    self.distribution = torch.distributions.Normal(0,1)

  def sample_latent_space(self, mu, sigma):

    """
    Takes a random sample from the encoded latent space. 

    Parameters
    ----------
    mu, sigma: ___
        The 'mean' and 'standard deviation' of the latent space, computed from the encoder


    Returns
    --------

    mu + eps * std: ____
        The random sample obtained from the latent space
    
    """
      
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(sigma)
    return mu + eps * sigma

  def forward(self, x):

    """
    Runs the full convolutional autoencoder architecture i.e. encoding the input images to a latent space,
    taking a random sample from that latent space and reconstructing a set of images from that sample.

    Parameters
    ----------

    x: ___
        The set of input images


    Returns
    --------
    self.decode(z): ___
        The reconstructed set of images

    mu, log_var:
        The 'mean' and 'standard deviation' from the latent space, used to compute the loss function.

    """

    mu, sigma = self.encoder(x)
    z = self.sample_latent_space(mu, sigma)
    x_hat = self.decoder(z)

    return x_hat, mu, sigma
  
def loss_function(reconstructed_x, x, mean, log_var, epsilon=1e-10):

    """
    Calculates the loss function to assess the performance of a model at each iteration.

    Parameters
    ----------
    reconstructed_x: torch.tensor
        The images reconstructed by the decoder part of the model architecture.

    x: torch.tensor 
        The original images passed into the autoencoder architecture.

    mean:__
        The mean from which the the latent space is sampled. 

    log_var: ___
        The standard deviation from which the latent space is sampled 
        
    epsilon: float, optional
        The minimum value to which the reconstructed image pixel values are clamped.


    Returns 
    -------

    BCE + KLD: ___

    Examples
    --------

    """
    
    # Clamp the reconstructed outputs to avoid log(0) in binary cross entropy
    reconstructed_x = torch.clamp(reconstructed_x, min=epsilon, max=1-epsilon)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD


def evaluate_conv_vae_results(vae, model_path, test_data, batch_size, device='cpu'): 

    """
    Produce a set of reconstructed images using the trained autoencoder and the test set

    Parameters
    ----------

    vae: VAE
        The trained autoencoder used to produce new images.

    model_path: str
        The folder directory in which the vae's model state is located 

    test_data: WildFireDataset
        The testing dataset that should be passed into a dataloader to generate new images

    batch_size: int
        The size of the batches passed into the trained vae. 

    Returns
    -------

    generated_images: tuple
        A set of batches of reconstructed images, with dimensions [batches, images, x, y]

    Examples
    --------
    
    """
    
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    vae.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("Loaded data")
    total_loss = 0.0
    results = []
    i = 0
    with torch.no_grad():
        for data in test_loader:
            while i<5:
                print("In dataloader")
                inputs = data[0].to(device).float()
                print("Input loaded")
                reconstructed_x, mean, log_var = vae(inputs)
                print("Passed through VAE")
                if len(reconstructed_x) != batch_size:
                    continue
                results.append(np.array(reconstructed_x.cpu()))
                loss_val = loss_function(reconstructed_x, inputs, mean, log_var)
                total_loss += loss_val.item()
                print(total_loss)
                i+=1
    return np.array(results)