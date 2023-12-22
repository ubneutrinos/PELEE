import unittest

import numpy as np
from microfit.histogram import MultiChannelBinning, Binning

class TestMultiChannelBinning(unittest.TestCase):
    def make_test_binning(
        self, multichannel=False, with_query=False, second_query="matching"
    ):
        bin_edges = np.array([0, 1, 2, 3])
        first_channel_binning = Binning("x", bin_edges, "x-axis label")
        if with_query:
            first_channel_binning.selection_query = "bdt > 0.5"
        if not multichannel:
            return first_channel_binning

        second_channel_binning = Binning("y", bin_edges, "y-axis label")
        second_query_string = {
            "matching": "bdt > 0.5",
            "non_matching": "bdt < 0.5",
            "overlapping": "bdt < 0.8",
        }[second_query]
        if with_query:
            second_channel_binning.selection_query = second_query_string
        binning = MultiChannelBinning([first_channel_binning, second_channel_binning])
        return binning
    
    def test_binning_to_from_dict(self):
        bin_edges = np.array([0, 1, 2, 3])
        binning = Binning("x", bin_edges, "x-axis label")
        
        binning_dict = binning.to_dict()
        new_binning = Binning.from_dict(binning_dict)
        
        self.assertEqual(binning, new_binning)
    
    def test_equality(self):
        binning = self.make_test_binning()
        binning2 = self.make_test_binning()
        assert isinstance(binning, Binning)
        assert isinstance(binning2, Binning)
        self.assertEqual(binning, binning2)
        # Test that equality fails when the bin edges are different
        binning2.bin_edges = np.array([1, 2, 3, 4])
        self.assertNotEqual(binning, binning2)
        binning2.bin_edges = binning.bin_edges
        # The label should normally affect equality, except when it is None
        binning2.label = "asdf"
        self.assertNotEqual(binning, binning2)
        binning2.label = None
        self.assertEqual(binning, binning2)
    
    def test_multi_channel_binning(self):
        binning = self.make_test_binning(multichannel=True)
        binning2 = self.make_test_binning(multichannel=True)
        assert isinstance(binning, MultiChannelBinning)
        assert isinstance(binning2, MultiChannelBinning)
        self.assertEqual(binning, binning2)
        binning_from_dict = MultiChannelBinning(**binning.__dict__)
        self.assertEqual(binning, binning_from_dict)

        # test __getitem__
        self.assertEqual(binning[0], binning["x-axis label"])
        # Test __len__
        self.assertEqual(len(binning), 2)
        # Test iteration
        for channel in binning:
            self.assertTrue(channel in binning)
    
    def test_copy(self):
        binning = self.make_test_binning(multichannel=True)
        binning2 = binning.copy()
        self.assertEqual(binning, binning2)
        binning2[0].bin_edges = np.array([1, 2, 3, 4])
        self.assertNotEqual(binning, binning2)


if __name__ == '__main__':
    unittest.main()