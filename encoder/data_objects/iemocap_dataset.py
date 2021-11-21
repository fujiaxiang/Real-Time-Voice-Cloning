from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


emo_categories = ['Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
                  'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise']
emo_map = {k: i for i, k in enumerate(emo_categories)}


class IemocapDataset(Dataset):
    def __init__(self, meta_data: Path, triplet: bool = False):
        self.meta_data_path = meta_data
        self.meta_data = pd.read_csv(meta_data).dropna().reset_index()
        self.triplet = triplet

        self.meta_data['labels'] = self.meta_data['labels'].map(emo_map)

        if self.triplet:
            self.index = self.meta_data.index.values
            self.labels = self.meta_data['labels']

    def __len__(self):
        return len(self.meta_data)
        
    def __getitem__(self, index):
        if self.triplet:
            anchor = self.meta_data.iloc[index]
            anchor_mel = np.load(anchor['m_file'], allow_pickle=True)

            pos_mask = (self.index != index) & (self.labels == anchor['labels'])
            pos = np.random.choice(self.meta_data['m_file'][pos_mask])
            pos = np.load(pos, allow_pickle=True)

            neg_mask = self.labels != anchor['labels']
            neg = np.random.choice(self.meta_data['m_file'][neg_mask])
            neg = np.load(neg, allow_pickle=True)
            return torch.from_numpy(anchor_mel), torch.from_numpy(pos), torch.from_numpy(neg)
        else:
            row = self.meta_data.iloc[index]
            mel = np.load(row['m_file'], allow_pickle=True)
            return row['uttid'], torch.from_numpy(mel), row['labels'], row['text']
