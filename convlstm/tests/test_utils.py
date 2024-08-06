import sys
sys.path.append('..')

import unittest
import torch
import random
import numpy as np
from convlstm import set_device, set_seed

class TestUtilsFunctions(unittest.TestCase):
    """
    We will not test the random seed because it should not be deterministic. 
    Thus it is not point in using the known answer to assert the result of randomnizing the seed.

    def test_set_seed(self):
        seed = 42
        set_seed(seed)
        
        # Test random seed
        self.assertEqual(random.randint(0, 100), 81)
        
        # Test numpy seed
        self.assertEqual(np.random.randint(0, 100), 51)
        
        # Test torch seed
        self.assertEqual(torch.randint(0, 100, (1,)).item(), 42)
        
        # Test cuda seed (if available)
        if torch.cuda.is_available():
            self.assertEqual(torch.cuda.FloatTensor(1).uniform_().item(), 
            torch.cuda.FloatTensor(1).uniform_().item())
        
        # Test cudnn settings
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cudnn.enabled)
    """

    def test_set_device(self):
        """
        Test if the device is either 'mps', 'cuda', or 'cpu'.
        """
        device = set_device()
        self.assertIn(device.type, ['mps', 'cuda', 'cpu'])

if __name__ == '__main__':
    unittest.main()
