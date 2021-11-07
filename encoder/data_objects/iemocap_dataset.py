from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


emo_categories = ['Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
                  'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise']
emo_map = {k: i for i, k in enumerate(emo_categories)}


class IemocapDataset(Dataset):
    def __init__(self, meta_data: Path):
        self.meta_data_path = meta_data
        self.meta_data = pd.read_csv(meta_data).dropna().reset_index()

    def __len__(self):
        return len(self.meta_data)
        
    def __getitem__(self, index):
        row = self.meta_data.iloc[index]
        mel = np.load(row['m_file'], allow_pickle=True)
        label = emo_map[row['labels']]
        return row['uttid'], torch.from_numpy(mel), label, row['text']
