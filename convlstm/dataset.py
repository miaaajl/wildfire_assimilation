import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class WildfireDatasetImage(Dataset):
    """
    Custom dataset for wildfire images.

    Attributes
    ----------
    data_path: str
        Path to the data file.
    seq_len: int
        Sequence length.
    time_step: int
        Time step.
    window_size: int
        Window size.
    img_size: int
        Image size to transform.
    transform: torchvision.transforms.Compose
        Transformations to apply to the image.
    
    Examples
    --------
    >>> data_path = 'data.npy'
    >>> seq_len = 2
    >>> time_step = 10
    >>> window_size = 1
    >>> img_size = 128
    >>> transform = None
    >>> dataset = WildfireDatasetImage(data_path, seq_len, time_step,
    window_size, img_size, transform)
    """
    def __init__(self,
                 data_path,
                 seq_len=2,
                 time_step=10,
                 window_size=1,
                 img_size=128,
                 transform=None):
        """
        Initialize the WildfireDatasetImage.

        Parameters
        ----------
        data_path: str
            Path to the data file.
        seq_len: int
            Sequence length. Default is 2.
        time_step: int
            Time step. Default is 10.
        window_size: int
            Window size. Default is 1.
        img_size: int
            Image size to transform. Default is 128.
        transform: torchvision.transforms.Compose
            Transformations to apply to the image.
        """
        self.data = self._load_data(data_path)
        self.seq_len = seq_len
        self.time_step = time_step
        self.window_size = window_size
        self.img_size = img_size
        self.transform = self._default_transform() if transform is None else transform

        fire_idx_ls = self._divide_wildfires(self.data)
        wildfire_ls = self._create_wildfire_ls(self.data, fire_idx_ls)

        self.input_sequences, self.output_sequences = self._create_sequences(wildfire_ls)

    def _load_data(self, data_path):
        """
        Load the data from the data path.

        Check if the data path exists and the file format is .npy.

        Parameters
        ----------
        data_path: str
            Path to the data file.
        
        Returns
        -------
        data: numpy.ndarray
            Data array.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        elif not data_path.endswith('.npy'):
            raise ValueError("Data file must be in .npy format.")
        else:
            data = np.load(data_path).astype(np.float32)
            return data

    def _default_transform(self):
        """
        Default transformation to apply to the image.
        Resize the image to the specified image size.

        Returns
        -------
        transform: torchvision.transforms.Compose
            Transformations to apply to the image.
        """
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])
        return transform

    def _divide_wildfires(self, data):
        """
        Divide the wildfires in the data.
        
        Check the difference between the mean of the current and next data.
        If the difference is positive, it is considered two kinds of wildfire.
        
        Parameters
        ----------
        data: numpy.ndarray
            Data array.
        
        Returns
        -------
        fire_idx_ls: list
            List of wildfire indexes.
        """
        fire_idx_ls = []
        for i in range(len(data) - 1):
            if np.mean(data[i]) - np.mean(data[i+1]) > 0:
                fire_idx_ls.append(i)
        return fire_idx_ls

    def _create_wildfire_ls(self, data, idx_ls):
        """
        Create a list of wildfires. Use the wildfire indexes to divide the data.
        
        Parameters
        ----------
        data: numpy.ndarray
            Data array.
        idx_ls: list
            List of wildfire indexes.
        
        Returns
        -------
        data_ls: list
            List of wildfires.
        """
        idx_ls = [-1] + idx_ls + [len(data)-1]
        data_ls = [data[idx_ls[i]+1:idx_ls[i+1]+1] for i in range(len(idx_ls)-1)]
        return data_ls

    def _create_sequences(self, data_ls):
        """
        Create input and output sequences from the wildfire list.

        Parameters
        ----------
        data_ls: list
            List of wildfires.
        
        Returns
        -------
        input_sequences: torch.Tensor
            Input sequences.
        output_sequences: torch.Tensor
            Output sequences.
        """
        input_sequences = []
        output_sequences = []

        for data in data_ls:
            for i in range(0, len(data) - self.seq_len * self.time_step, self.window_size):
                input_seq = [data[i+j*self.time_step] for j in range(self.seq_len)]
                unique_arrays = set(tuple(arr.flatten()) for arr in input_seq)
                # if the input sequence is not the same
                if len(unique_arrays) != 1:
                    input_sequences.append(torch.stack([self.transform(seq) for seq in input_seq]))
                    output_sequences.append(self.transform(data[i+self.seq_len*self.time_step]))

        if not input_sequences:
            raise ValueError("No input sequences found. \
                             Try changing the window size or sequence length.")
        img_size = input_sequences[0][0].size(-1)

        input_sequences = torch.stack(input_sequences)
        # resize to (batch_size, 1, seq_len, img_size, img_size)
        input_sequences = input_sequences.view(input_sequences.size(0), 1,
                                               self.seq_len, img_size, img_size)

        output_sequences = torch.stack(output_sequences)
        output_sequences = output_sequences.view(output_sequences.size(0), 1,
                                                 img_size, img_size)

        return input_sequences, output_sequences

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.input_sequences)

    def __getitem__(self, idx):
        """
        Get the item of the dataset.
        """
        input_seq = self.input_sequences[idx]
        output_seq = self.output_sequences[idx]
        return input_seq, output_seq
