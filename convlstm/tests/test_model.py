import sys
sys.path.append('..')

import unittest
import torch
import torch.nn as nn
from convlstm import ConvLSTMModel  
from convlstm import set_device  

device = set_device()

class TestConvLSTMModel(unittest.TestCase):
    """
    Functions:
    ----------
    setUp(self):
        Initialize the ConvLSTMModel object.
    test_model_instantiation(self):
        Test if the model is an instance of nn.Module.
    test_forward_pass(self):
        Test the forward pass of the model.
    test_weight_initialization(self):
        Test if the weights are initialized.
    test_train_eval_mode(self):
        Test if the model is able to turn in training or evaluation mode.
    """

    def setUp(self):
        self.input_channels = 1
        self.hidden_channels = 16
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.activation = 'tanh'
        self.frame_size = (64, 64)
        self.num_layers = 2
        self.batch_size = 2
        self.seq_len = 5

        self.model = ConvLSTMModel(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            self.padding,
            self.activation,
            self.frame_size,
            self.num_layers
        ).to(device)

    def test_model_instantiation(self):
        """
        Test if the model is an instance of nn.Module.
        """
        self.assertIsInstance(self.model, nn.Module)

    def test_forward_pass(self):
        """
        Test the forward pass of the model.
        """
        input_tensor = torch.randn(self.batch_size, self.input_channels,
                                   self.seq_len, *self.frame_size).to(device)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.input_channels, *self.frame_size))

    def test_weight_initialization(self):
        """
        Test if the weights are initialized.
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.assertTrue(torch.any(param != 0))

    def test_train_eval_mode(self):
        """
        Test if the model is able to turn in training or evaluation mode.
        """
        self.model.train()
        self.assertTrue(self.model.training)

        self.model.eval()
        self.assertFalse(self.model.training)

if __name__ == '__main__':
    unittest.main()
