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
        third_channel_binning = Binning("z", bin_edges, "z-axis label")
        third_channel_binning.selection_query = "bdt > 0.5"

        binning = MultiChannelBinning([first_channel_binning, second_channel_binning, third_channel_binning])
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
        binning_from_dict = MultiChannelBinning.from_dict(binning.to_dict())
        self.assertEqual(binning, binning_from_dict)

        # test __getitem__
        self.assertEqual(binning[0], binning["x-axis label"])
        # Test __len__
        self.assertEqual(len(binning), 3)
        # Test iteration
        for channel in binning:
            self.assertTrue(channel in binning)
    
    def test_copy(self):
        binning = self.make_test_binning(multichannel=True)
        binning2 = binning.copy()
        self.assertEqual(binning, binning2)
        binning2[0].bin_edges = np.array([1, 2, 3, 4])
        self.assertNotEqual(binning, binning2)

    def test_roll_channels(self):
        binning = self.make_test_binning(multichannel=True)
        original_order = binning.binnings.copy()
        binning.roll_channels(1)
        new_order = binning.binnings
        self.assertEqual(original_order[-1], new_order[0])
        for i in range(len(original_order) - 1):
            self.assertEqual(original_order[i], new_order[i + 1])

    def test_roll_to_first(self):
        binning = self.make_test_binning(multichannel=True)
        original_order = binning.binnings.copy()
        binning.roll_to_first("y-axis label")
        new_order = binning.binnings
        self.assertEqual(original_order[1], new_order[0])
        self.assertEqual(original_order[0], new_order[2])
    
    def test_delete_channel(self):
        binning = self.make_test_binning(multichannel=True)
        original_order = binning.binnings.copy()
        binning.delete_channel("y-axis label")
        new_order = binning.binnings
        self.assertEqual(len(original_order) - 1, len(new_order))
        self.assertEqual(original_order[0], new_order[0])
        self.assertNotEqual(original_order[1], new_order[1])

    
    def test_deep_copy(self):
        binning = self.make_test_binning(multichannel=True)
        binning_copy = binning.copy()

        # Check if the copy is a MultiChannelBinning object
        self.assertIsInstance(binning_copy, MultiChannelBinning)

        # Check if the copy is a distinct object from the original
        self.assertIsNot(binning, binning_copy)

        # Check if each binning in the copy is a distinct object from the original
        for original_binning, copied_binning in zip(binning.binnings, binning_copy.binnings):
            self.assertIsNot(original_binning, copied_binning)


if __name__ == '__main__':
    unittest.main()