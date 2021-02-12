#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np

TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

class CMU(Dataset):
    def __init__(self, data_path, use_hg=True):
        """
        :param data_path: path to dataset
        """

        self.data_path = data_path
        # loading data
        data = torch.load(self.data_path)
        self.input_data = data['src']
        self.output_data = data['tgt']
        
    def __getitem__(self, index):
        inputs = torch.tensor(self.input_data[index]).float()
        outputs = torch.tensor(self.output_data[index]).float()
        data={'joint2d': inputs, 'truth': outputs}
        return data

    def __len__(self):
        return len(self.input_data)