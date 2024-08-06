import sys
sys.path.append('..')

import unittest
import torch
import numpy as np
import os
from torchvision import transforms
from convlstm import WildfireDatasetImage

class TestWildfireDatasetImage(unittest.TestCase):
    """
    Test the WildfireDatasetImage class.
    """
    def setUp(self):
        # Create dummy data
        self.data_path = './dummy_data.npy'
        self.seq_len = 2
        self.time_step = 1
        self.window_size = 1
        self.img_size = 128

        # Create a dummy numpy array and save it
        self.dummy_data = np.random.rand(20, 64, 64).astype(np.float32)
        self.dummy_data = np.nan_to_num(self.dummy_data,
                                        nan=0.0, posinf=1.0, neginf=0.0)  # Ensure no NaNs or infs
        print("Initial dummy data range:", self.dummy_data.min(), self.dummy_data.max())
        np.save(self.data_path, self.dummy_data)

    def tearDown(self):
        # Remove the dummy data file
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        self.assertEqual(len(dataset), len(dataset.input_sequences))

    def test_default_transform(self):
        """
        Test the default transformation.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        transform = dataset._default_transform()
        dummy_image = self.dummy_data[0]
        dummy_image = np.nan_to_num(dummy_image, nan=0.0,
                                    posinf=1.0, neginf=0.0)  # Ensure no NaNs or infs
        transformed_image = transform(dummy_image)
        print("Transformed image range:", transformed_image.min().item(),
              transformed_image.max().item())
        print("Transformed image dtype:", transformed_image.dtype)
        self.assertEqual(transformed_image.shape, torch.Size([1, self.img_size, self.img_size]))

    def test_divide_wildfires(self):
        """
        Test the divide_wildfires method.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        fire_idx_ls = dataset._divide_wildfires(self.dummy_data)
        self.assertIsInstance(fire_idx_ls, list)

    def test_create_wildfire_ls(self):
        """
        Test the create_wildfire_ls method.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        fire_idx_ls = dataset._divide_wildfires(self.dummy_data)
        wildfire_ls = dataset._create_wildfire_ls(self.dummy_data, fire_idx_ls)
        self.assertIsInstance(wildfire_ls, list)

    def test_create_sequences(self):
        """
        Test the create_sequences method.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        fire_idx_ls = dataset._divide_wildfires(self.dummy_data)
        wildfire_ls = dataset._create_wildfire_ls(self.dummy_data, fire_idx_ls)
        input_sequences, output_sequences = dataset._create_sequences(wildfire_ls)
        print("Input sequences range:", input_sequences.min().item(), input_sequences.max().item())
        print("Input sequences dtype:", input_sequences.dtype)
        self.assertIsInstance(input_sequences, torch.Tensor)
        self.assertIsInstance(output_sequences, torch.Tensor)

    def test_dataset_interface(self):
        """
        Test the dataset interface.
        """
        dataset = WildfireDatasetImage(data_path=self.data_path, seq_len=self.seq_len,
                                       time_step=self.time_step, window_size=self.window_size,
                                       img_size=self.img_size)
        self.assertEqual(len(dataset), len(dataset.input_sequences))
        input_seq, output_seq = dataset[0]
        self.assertIsInstance(input_seq, torch.Tensor)
        self.assertIsInstance(output_seq, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
