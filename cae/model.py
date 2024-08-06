import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """
    A Convolutional Autoencoder for image reconstruction.

    The Autoencoder consists of two main parts:
    1. Encoder: Compresses the input image into a lower-dimensional representation.
    2. Decoder: Reconstructs the image from the encoded representation.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 128x128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 64x64
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 128x128
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 256x256
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.
        
        Args:
            x (torch.Tensor): Input tensor, expected shape is (N, 1, 256, 256) where N is the batch size.
            
        Returns:
            torch.Tensor: Reconstructed tensor with the same shape as input.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses data from a .npy file for use in a PyTorch model.
    
    The function performs the following steps:
    1. Loads the data from a specified .npy file.
    2. Converts the data type to float32.
    3. Expands the dimensions of the data to match the input requirements of the model.
    4. Converts the data to a PyTorch tensor and moves it to the specified device (CPU or GPU).
    
    Args:
        file_path (str): Path to the .npy file containing the data.
        
    Returns:
        torch.Tensor: The preprocessed data as a PyTorch tensor, ready for input into a model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(file_path)
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=1)
    tensor_data = torch.tensor(data).to(device)
    return tensor_data


def reshape_and_compute_cov(data):
    """
    Reshapes the input data and computes the covariance matrix.
    
    This function performs the following steps:
    1. Retrieves the number of samples and features from the input data.
    2. Reshapes the data to ensure it is in the correct format for covariance computation.
    3. Moves the data to the CPU (if it's on a GPU) and converts it to a NumPy array.
    4. Computes the covariance matrix of the reshaped data.
    
    Args:
        data (torch.Tensor): The input data tensor with shape (n_samples, n_features).
        
    Returns:
        np.ndarray: The computed covariance matrix.
    """
    n_samples, n_features = data.shape
    reshaped_data = data.view(n_samples, n_features).cpu().numpy()
    cov_matrix = np.cov(reshaped_data, rowvar=False)
    return cov_matrix


def compute_covariances(encoded_background, encoded_observation):
    """
    Computes the covariance matrices for encoded background and encoded observation data.
    
    This function performs the following steps:
    1. Computes the covariance matrix of the encoded background data using the `reshape_and_compute_cov` function.
    2. Computes the covariance matrix of the encoded observation data using the `reshape_and_compute_cov` function.
    
    Args:
        encoded_background (torch.Tensor): The encoded background data tensor.
        encoded_observation (torch.Tensor): The encoded observation data tensor.
        
    Returns:
        tuple: A tuple containing two covariance matrices:
            - B (np.ndarray): The covariance matrix of the encoded background data.
            - R (np.ndarray): The covariance matrix of the encoded observation data.
    """
    B = reshape_and_compute_cov(encoded_background)
    R = reshape_and_compute_cov(encoded_observation)
    
    return B, R


def kalman_filter(x, P, measurement, R, Q, F, H):
    """
    Perform a single update step of the Kalman filter.

    The Kalman filter is a recursive algorithm used to estimate the state of a linear dynamic system from a series
    of noisy measurements. It predicts the state of the system and then updates this prediction based on new measurement data.

    Parameters:
    - x (numpy.ndarray): The state estimate at the previous time step.
    - P (numpy.ndarray): The covariance matrix of the state estimate.
    - measurement (numpy.ndarray): The new measurement data.
    - R (numpy.ndarray): The covariance matrix of the measurement noise.
    - Q (numpy.ndarray): The covariance matrix of the process noise.
    - F (numpy.ndarray): The state transition model which is applied to the previous state.
    - H (numpy.ndarray): The observation model which maps the true state space into the observed space.

    Returns:
    - x (numpy.ndarray): The updated state estimate.
    - P (numpy.ndarray): The updated covariance matrix of the state estimate.

    The function operates in two steps: prediction and update. In the prediction step, the state estimate and
    its uncertainty are propagated through the system dynamics. In the update step, the new measurement is
    incorporated to refine the state estimate, utilizing the Kalman Gain to balance the estimates with the uncertainties.
    """
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    P = P.numpy() if isinstance(P, torch.Tensor) else P
    measurement = measurement.numpy() if isinstance(measurement, torch.Tensor) else measurement
    R = R.numpy() if isinstance(R, torch.Tensor) else R
    Q = Q.numpy() if isinstance(Q, torch.Tensor) else Q
    F = F.numpy() if isinstance(F, torch.Tensor) else F
    H = H.numpy() if isinstance(H, torch.Tensor) else H
    
    # Predict the state and covariance
    x = F @ x  # State prediction
    P = F @ P @ F.T + Q  # Covariance prediction

    # Compute the Kalman Gain
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman Gain

    # Update the estimate via z
    y = measurement - H @ x  # Innovation or measurement pre-fit residual
    x = x + K @ y  # Updated state estimate
    P = P - K @ H @ P  # Updated estimate covariance

    return x, P

def run_data_assimilation(encoded_background, encoded_observation, B, R, alpha=1.0, beta=1.0):
    """
    Runs the data assimilation process using the Kalman filter.

    This function performs the following steps:
    1. Adjusts the covariance matrices B and R by the scaling factors alpha and beta, respectively.
    2. Initializes the state vector x and covariance matrix P.
    3. Defines the state transition matrix F, observation matrix H, and process noise covariance matrix Q.
    4. Flattens the encoded observation data for processing.
    5. Iteratively applies the Kalman filter to each observation to update the state estimate.
    6. Reshapes the updated state estimates to match the input shape.

    Args:
        encoded_background (torch.Tensor): The encoded background data tensor.
        encoded_observation (torch.Tensor): The encoded observation data tensor.
        B (np.ndarray): The covariance matrix of the encoded background data.
        R (np.ndarray): The covariance matrix of the encoded observation data.
        alpha (float, optional): Scaling factor for the background covariance matrix B. Default is 1.0.
        beta (float, optional): Scaling factor for the observation covariance matrix R. Default is 1.0.
        
    Returns:
        np.ndarray: The updated state estimates after data assimilation.
    """
    n_samples, n_features = encoded_background.shape

    # Adjust weights
    B = alpha * B
    R = beta * R

    x = np.zeros(n_features)
    P = B
    F = np.eye(n_features)
    H = np.eye(n_features)
    Q = np.eye(n_features) * 0.01

    encoded_observation_flat = encoded_observation.reshape(n_samples, n_features)
    updated_states = []

    for obs in encoded_observation_flat:
        x, P = kalman_filter(x, P, obs, R, Q, F, H)
        updated_states.append(x)
    
    updated_states = np.array(updated_states)
    updated_states = updated_states.reshape(n_samples, -1)
    return updated_states


def binarize_image(image, threshold=0.5):
    """
    Convert an image to a binary (black and white) format based on a specified threshold.

    This function processes a grayscale image where pixel values are assumed to be normalized
    between 0 and 1. It binarizes the image by setting all pixels with values greater than the 
    threshold to 1 (white) and all other pixels to 0 (black).

    Parameters:
    - image (numpy.ndarray): A single-channel (grayscale) image where pixel values are expected
      to be in the range [0, 1].
    - threshold (float, optional): The cutoff value for binarizing the image. Default is 0.5.
      Pixels with values above this threshold will be set to 1, and those below or equal to it will be set to 0.

    Returns:
    - binary_image (numpy.ndarray): The binarized image as a numpy array of the same shape as the input,
      with pixel values of 0 or 1.

    The function can be used for image processing tasks where binary images are required, such as 
    contour detection, background subtraction, or other morphological operations.
    """
    
    # Apply the threshold to binarize the image
    binary_image = (image >= threshold).astype(np.float32)

    return binary_image
