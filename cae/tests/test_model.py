import sys
sys.path.append('..')

import unittest
import numpy as np
import torch
from cae import Autoencoder, load_and_preprocess_data, reshape_and_compute_cov
from cae import compute_covariances, kalman_filter, run_data_assimilation, binarize_image

class TestAutoencoder(unittest.TestCase):
    """
    Functions:
    ----------
    setUp(self):
        Initialize the Autoencoder object.
    test_forward(self):
        Test the forward pass of the Autoencoder.
    """
    def setUp(self):
        self.autoencoder = Autoencoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(self.device)

    def test_forward(self):
        input_tensor = torch.randn(1, 1, 256, 256).to(self.device)
        output_tensor = self.autoencoder(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)

class TestDataProcessing(unittest.TestCase):
    """
    Functions:
    ----------
    test_load_and_preprocess_data(self):
        Test the load_and_preprocess_data function.
    test_reshape_and_compute_cov(self):
        Test the reshape_and_compute_cov function.
    test_compute_covariances(self):
        Test the compute_covariances function.
    """
    def test_load_and_preprocess_data(self):
        """
        Test the load_and_preprocess_data function.
        """
        test_data = np.random.rand(10, 256, 256)
        np.save('test_data.npy', test_data)
        tensor_data = load_and_preprocess_data('test_data.npy')
        self.assertEqual(tensor_data.shape, (10, 1, 256, 256))
        self.assertTrue(tensor_data.dtype, torch.float32)

    def test_reshape_and_compute_cov(self):
        """
        Test the reshape_and_compute_cov function.
        """
        test_tensor = torch.randn(10, 256)
        cov_matrix = reshape_and_compute_cov(test_tensor)
        self.assertEqual(cov_matrix.shape, (256, 256))

    def test_compute_covariances(self):
        """
        Test the compute_covariances function.
        """
        encoded_background = torch.randn(10, 256)
        encoded_observation = torch.randn(10, 256)
        B, R = compute_covariances(encoded_background, encoded_observation)
        self.assertEqual(B.shape, (256, 256))
        self.assertEqual(R.shape, (256, 256))

class TestKalmanFilter(unittest.TestCase):
    """
    Functions:
    ----------
    test_kalman_filter(self):
        Test the kalman_filter function.
    test_run_data_assimilation(self):
        Test the run_data_assimilation function.
    """
    def test_kalman_filter(self):
        """
        Test the kalman_filter function.
        """
        x = np.zeros(4)
        P = np.eye(4)
        measurement = np.array([1, 1, 1, 1])
        R = np.eye(4)
        Q = np.eye(4) * 0.1
        F = np.eye(4)
        H = np.eye(4)

        x_updated, P_updated = kalman_filter(x, P, measurement, R, Q, F, H)
        self.assertEqual(x_updated.shape, (4,))
        self.assertEqual(P_updated.shape, (4, 4))

    def test_run_data_assimilation(self):
        """
        Test the run_data_assimilation function.
        """
        encoded_background = torch.randn(10, 256)
        encoded_observation = torch.randn(10, 256)
        B, R = compute_covariances(encoded_background, encoded_observation)
        updated_states = run_data_assimilation(encoded_background, encoded_observation, B, R)
        self.assertEqual(updated_states.shape, (10, 256))

class TestImageProcessing(unittest.TestCase):
    """
    Functions:
    ----------
    test_binarize_image(self):
        Test the binarize_image function.
    """
    def test_binarize_image(self):
        test_image = np.random.rand(256, 256)
        binary_image = binarize_image(test_image, threshold=0.5)
        self.assertEqual(binary_image.shape, (256, 256))
        self.assertTrue(np.array_equal(binary_image, binary_image.astype(bool)))

if __name__ == '__main__':
    unittest.main()
