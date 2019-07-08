"""Tests for datasets.py"""
import os
import unittest

import torch
from torch.utils.data import Dataset, TensorDataset

from fspy import datasets

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class P450Test(unittest.TestCase):

    def test_P450_content(self):
        dataset_path = os.path.join(PROJECT_PATH,
                                    "test.csv")
        dataset = datasets.P450(dataset_path)

        actual = dataset[0]
        expected = ([
            0.3636005765, -0.1180073836, -0.3341110865, 0.06111906874,
            -0.1532670732, 0.1469780266, -0.09911487805, 0.0540478714,
            -0.1234771175, 0.004010443459, -6.336845033, 1.366588625,
            -1.996578359, 0.01043966741, 0.2307686696, 0.6712427051,
            3.825980377, -0.2106333925, -0.115734235, -0.1537076718,
            -0.590584235, 0, -0.1657192683, 0.03242172949, -0.1923523725,
            0.1851440576, 0.03018465632, 0.1749733925, 0.0006231263858,
            0.001270110865, 0.0009713747228, 0.001719778271, 0.001044101996,
            0.002105543237, 0.3048780488, 0.01441241685, -3.233787162,
            4.409356984, 4.297090817, 2.580609042, 0.02494882483, 0.1162843942,
            0.05829276433, 23.42270603, 43.44917863, 20.02647271, 0.855312892,
            1130.394948, 1275.936343, 6.239191279, 7.042503964
        ], 1)

        self.assertEqual(actual, expected)

    def test_P450_size(self):
        dataset_path = os.path.join(PROJECT_PATH,
                                    "data/P450/raw/P450_chimeric_library.csv")
        dataset = datasets.P450(dataset_path)

        size = len(dataset)

        self.assertEqual(size, 988)


class LactamaseTest(unittest.TestCase):

    def test_lactamase_content(self):
        dataset_path = os.path.join(
            PROJECT_PATH, "data/lactamase/raw/lactamase_chimeric_library.csv")
        dataset = datasets.P450(dataset_path)

        actual = dataset[0]
        expected = ([
            0.3563395038, -0.211545, -0.2198880534, -0.01764950382,
            -0.1528637786, 0.202701374, -0.221229542, 0.04743209924,
            -0.2863560687, 0.0009678244275, -6.11018042, 1.128021641,
            -1.890459847, 0.009322061069, 0.1860464504, 0.6380646947,
            3.73549458, -0.2226339313, -0.257766145, -0.1237607252,
            -0.4780089695, 0, -0.1703918321, 0.01377801527, -0.211341374,
            0.1511583588, 0.00706129771, 0.3167061069, 0.00011, 0.002649656489,
            0.001576793893, 0.003111145038, 0.002209656489, 0.003802977099,
            0.25, -0.00572519084, -3.27878084, 3.71110687, 4.058386506,
            3.117082142, 0.0452620229, 0.1580859766, 0.1002191762, 22.68271656,
            42.58527851, 19.90256218, 0.877742033, 960.2157277, 1081.637438,
            6.14160518, 6.918226708
        ], 0)
        self.assertEqual(actual, expected)

    def test_lactamase_size(self):
        dataset_path = os.path.join(
            PROJECT_PATH, "data/lactamase/raw/lactamase_chimeric_library.csv")
        dataset = datasets.P450(dataset_path)

        size = len(dataset)

        self.assertEqual(size, 553)
