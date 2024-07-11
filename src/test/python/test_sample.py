import unittest
import numpy as np
from src.main.python.data.datasets import Videodataset

class TestVideodataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Videodataset(data_dir="test_data")

    def test_len(self):
        self.assertEqual(len(self.dataset), 10)

    def test_getitem(self):
        video, label = self.dataset[0]
        self.assertEqual(video.shape, (16, 224, 224, 3))
        self.assertEqual(label, 0)

if __name__ == '__main__':
    unittest.main()
