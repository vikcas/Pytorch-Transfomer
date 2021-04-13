from torch.utils.data import Dataset
from torch import tensor
import pandas as pd
import numpy as np

class LoadAggregatedAzureDataset(Dataset):
    """
    Returns a batch_size times a sequence of len seq_len
    It has path hardcoded to prepared files from data_preprocess.py
    :param
    is_train: Default True. If False it uses test data
    bs: Batch size for training
    w_step: Sliding window step
    prediction_steps: Lookahead training, default 1
    """

    def __init__(self, mode="train", seq_len=1, prediction_step=1):
        self.mode = mode
        self.seq_len = int(seq_len)
        self.p_steps = int(prediction_step)
        if self.mode=="train":
            path = '/data/cloud_data/AzurePublicDataset2019/processed_data/univariate_data/depl_ANu_all/train_data.csv'
        elif self.mode=="validation":
            path = '/data/cloud_data/AzurePublicDataset2019/processed_data/univariate_data/depl_ANu_all/val_data.csv'
        elif self.mode == "test":
            path = '/data/cloud_data/AzurePublicDataset2019/processed_data/univariate_data/depl_ANu_all/test_data.csv'
        self.data_set = pd.read_csv(path)
        if self.seq_len > len(self.data_set) - self.p_steps:
            self.seq_len = len(self.data_set) - self.p_steps - 1
        # self.data_set_len = int(np.ceil((len(self.data_set) - self.p_steps - self.batch_size) / self.step) + 1)

    def __len__(self):
        return len(self.data_set)-(self.seq_len+self.p_steps) + 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            ind = np.random.randint(0,len(self.data_set)-(self.seq_len+self.p_steps), 1, dtype=int)
            start = int(ind)
            end = int(start + self.seq_len)
            input_idx = range(start, end)
            target_idx = range(end, end + self.p_steps)
        elif self.mode == 'validation':
            ind = np.random.randint(0, len(self.data_set) - (self.seq_len + self.p_steps), 1, dtype=int)
            start = int(ind)
            end = int(start + self.seq_len)
            input_idx = range(start, end)
            target_idx = range(end, end + self.p_steps)
        else:
            start = idx
            end = idx + self.seq_len
            input_idx = range(start, end)
            target_idx = range(end, end + self.p_steps)

        return tensor(self.data_set.iloc[input_idx, :-1].values), tensor(self.data_set.iloc[input_idx, -1].values), \
               tensor(self.data_set.iloc[target_idx, -1].values)