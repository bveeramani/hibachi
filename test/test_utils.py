"""Tests for fspy.utils"""
import unittest

from torch.utils.data import Dataset

from fspy import utils


class UtilityFunctionTest(unittest.TestCase):

    def test_remove_unselected_features(self):
        dataset = DummyDataset()
        dataset = utils.eliminate(dataset, set([1]))

        self.assertEqual([x for x, y in dataset], [[0], [0], [1], [1]])
        self.assertEqual([y for x, y in dataset], [0, 1, 0, 1])

    def test_create_folds(self):
        dataset = DummyDataset()
        folds = utils.fold(dataset, 3)

        self.assertEqual(len(folds), 3)

        # Three folds of size 1 x 1 x 2
        for fold_number, (training_fold, testing_fold) in enumerate(folds):
            if fold_number < 3 - 1:
                self.assertEqual(len(training_fold), 3)
                self.assertEqual(len(testing_fold), 1)
            else:
                self.assertEqual(len(folds[2][0]), 2)
                self.assertEqual(len(folds[2][1]), 2)


class DummyDataset(Dataset):

    def __init__(self):
        self.xy = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.z = [0, 1, 0, 1]  # f(x, y) = y

    def __getitem__(self, index):
        return self.xy[index], self.z[index]

    def __len__(self):
        return 4
