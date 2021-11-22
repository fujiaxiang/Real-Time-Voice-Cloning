import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence

from encoder.data_objects.iemocap_dataset import emo_map


class IemocapSynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, speaker_embed_path, emo_embed_path: Path):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, speaker_embed_path, emo_embed_path))

        self.meta_data = pd.read_csv(metadata_fpath).dropna().reset_index()
        self.meta_data['labels'] = self.meta_data['labels'].map(emo_map)

        self.speaker_embeds = np.load(speaker_embed_path, allow_pickle=True)
        self.emotion_embeds = np.load(speaker_embed_path, allow_pickle=True)

    def __getitem__(self, index):
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        meta = self.meta_data.iloc[index]
        mel = np.load(meta['m_file'], allow_pickle=True).astype(np.float32)
        
        # Load the embed
        speaker_embed = self.speaker_embeds[index].astype(np.float32)
        emotion_embed = self.emotion_embeds[index].astype(np.float32)

        # Get the text and clean it
        text = text_to_sequence(meta['text'], ["english_cleaners"])
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        return torch.from_numpy(text), torch.from_numpy(mel), torch.from_numpy(speaker_embed), torch.from_numpy(emotion_embed)

    def __len__(self):
        return len(self.meta_data)


def collate_fn(batch):
    """Collates a batch of padded variable length inputs"""

    texts, mels, speaker_embeds, emotion_embeds = zip(*batch)

    # get sequence lengths
    text_lengths = torch.tensor([x.shape[0] for x in texts])
    mel_lengths = torch.tensor([x.shape[0] for x in mels])

    # padding
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)

    # collate the embeddings
    speaker_embeds = torch.cat(speaker_embeds, dim=0)
    emotion_embeds = torch.cat(emotion_embeds, dim=0)

    return texts, text_lengths, mels, mel_lengths, speaker_embeds, emotion_embeds


# from synthesizer.iemocap_dataset import IemocapSynthesizerDataset
# ds = IemocapSynthesizerDataset('iemocap_meta_dev.csv', 'data/iemocap/synthesizer/speaker_enc_dev.npy', 'data/iemocap/synthesizer/emotion_enc_dev.npy')
