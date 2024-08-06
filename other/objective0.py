import numpy as np
import torch
from torch.utils.data import Dataset

class DataPreparation:
    """
    A class to prepare training and testing data by separating and filtering wildfires data.

    Attributes:
    -----------
    train : numpy.ndarray
        The training data loaded from the provided path.
    test : numpy.ndarray
        The testing data loaded from the provided path.

    Methods:
    --------
    separate_and_filter_datas():
        Separates and filters the training and testing wildfires data based on the threshold.
    """
    
    def __init__(self, train_path, test_path, dtype="int8", flatten=False):
        """
        Initializes the DataPreparation class with train and test paths and a threshold value.

        Parameters:
        -----------
        train_path : str
            The file path to the training data.
        test_path : str
            The file path to the testing data.
        """
        self.train = np.load(train_path).astype(dtype)
        self.test = np.load(test_path).astype(dtype)
    
    def separate_and_filter_datas(self, threshold=0):
        """
        Separates and filters the training and testing wildfires data.
        
        Parameters:
        -----------
        threshold : int, optional
            The threshold value for filtering the data (default is 100).

        Returns:
        --------
        tuple: A tuple containing two lists of numpy arrays:
            - filter_wildfires_train: The filtered training wildfires data.
            - filter_wildfires_test: The filtered testing wildfires data.
        """
        separate_wildfires_train = [
            self.train[100 * (i - 1):100 * i] 
            for i in range(1, len(self.train) // 100 + 1)
        ]
        separate_wildfires_test = [
            self.test[100 * (i - 1):100 * i] 
            for i in range(1, len(self.test) // 100 + 1)
        ]
        filter_wildfires_train = [
            wildfire for wildfire in separate_wildfires_train 
            if wildfire.sum(axis=1).sum(axis=1)[-1] > threshold
        ]
        filter_wildfires_test = [
            wildfire for wildfire in separate_wildfires_test 
            if wildfire.sum(axis=1).sum(axis=1)[-1] > threshold
        ]
        return filter_wildfires_train, filter_wildfires_test


class WildFireDataset(Dataset):
    """
    A PyTorch Dataset class to handle wildfires data for training models.

    Attributes:
    -----------
    wildfires : list of numpy.ndarray
        The list containing wildfire data arrays.
    n_previous : int
        The number of previous frames to consider as input.
    skipped_frame : int
        The number of frames to skip between the inputs.
    transform : callable, optional
        A function/transform to apply to the data.

    Methods:
    --------
    __len__():
        Returns the length of the dataset.
    __getitem__(idx):
        Retrieves the input and target data for a given index.
    """
    
    def __init__(self, wildfires, sequence_length=4, timestep=10, flatten=False):
        """
        Initializes the WildFireDataset class.

        Parameters:
        -----------
        wildfires : list of numpy.ndarray
            The list containing wildfire data arrays.
        n_previous : int, optional
            The number of previous frames to consider as input (default is 4).
        skipped_frame : int, optional
            The number of frames to skip between the inputs (default is 10).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        """
        self.flatten = flatten
        self.wildfires = wildfires
        self.number_of_wildfires = len(wildfires)
        self.images_in_wildfire = len(wildfires[0])
        self.sequence_length = sequence_length
        self.timestep = timestep
        self.max_idx = (
            self.images_in_wildfire - self.sequence_length * self.timestep
        ) * self.number_of_wildfires

    def __len__(self):
        """Returns the length of the dataset."""
        return self.max_idx

    def __getitem__(self, idx):
        """
        Retrieves the input and target data for a given index.

        Parameters:
        -----------
        idx : int
            The index of the data to retrieve.

        Returns:
        --------
        tuple: A tuple containing two elements:
            - previous_wildfires: A numpy array of previous wildfire frames.
            - next_wildfire: The next wildfire frame reshaped for compatibility.
        """
        i = idx % (self.images_in_wildfire - self.sequence_length * self.timestep)
        j = idx // (self.images_in_wildfire - self.sequence_length * self.timestep)
        wildfire = self.wildfires[j]
        previous_wildfires = np.array([
            wildfire[i + self.timestep * k] for k in range(self.sequence_length)
        ])
        next_wildfire = wildfire[i + self.timestep * self.sequence_length]
        next_wildfire = next_wildfire[np.newaxis, ...]  

        if self.flatten:
            previous_wildfires = previous_wildfires.reshape(previous_wildfires.shape[0], -1)
            next_wildfire = next_wildfire.reshape(1, -1)

        return previous_wildfires, next_wildfire
