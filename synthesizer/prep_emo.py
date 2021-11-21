import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from encoder.model import SpeakerEncoder
from encoder.emo_models import EmoEncoder
from encoder.train_emo import collate_fn
from encoder.data_objects.iemocap_dataset import IemocapDataset


def create_embeddings(model, loader, enc_type='speaker'):
    results = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            uttid, features, labels, texts, lengths = batch
            features = features.to(device)
            lengths = lengths.cpu()

            packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            if enc_type == 'speaker':
                embeds = model(packed_features)
            else:
                embeds, _ = model(packed_features)
            embeds = embeds.cpu().detach().numpy()
            results.append(embeds)
            break
    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speaker_enc_path = Path("encoder/saved_models/pretrained.pt")
emotion_enc_path = Path("encoder/saved_models/test2_backups/test2_bak_180000.pt")


speaker_enc = SpeakerEncoder(device, torch.device("cpu"))
checkpoint = torch.load(speaker_enc_path, device)
speaker_enc.load_state_dict(checkpoint["model_state"])

emotion_enc = EmoEncoder(device)
checkpoint = torch.load(emotion_enc_path, device)
emotion_enc.load_state_dict(checkpoint["model_state"])


output_dir = Path("data/iemocap/synthesizer")
output_dir.mkdir(parents=True, exist_ok=True)

data = {
    'train': "iemocap_meta_train.csv",
    'dev': "iemocap_meta_dev.csv",
    'test': "iemocap_meta_test.csv",
}

for env, meta in data.items():
    print("Env: ", env)
    dataset = IemocapDataset(Path(meta))
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=os.cpu_count() - 1 if sys.platform.startswith('linux') else 0,
        collate_fn=collate_fn
    )

    print("Creating speaker embeddings...")
    speaker_embeds = create_embeddings(speaker_enc, loader)
    speaker_embeds = np.concatenate(speaker_embeds)
    out_fpath = output_dir.joinpath(f'speaker_enc_{env}' + '.npy')
    np.save(out_fpath, speaker_embeds)

    print("Creating emotion embeddings...")
    emotion_embeds = create_embeddings(emotion_enc, loader, 'emotion')
    emotion_embeds = np.concatenate(emotion_embeds)
    out_fpath = output_dir.joinpath(f'emotion_enc_{env}' + '.npy')
    np.save(out_fpath, emotion_embeds)


# python -m synthesizer.prep_emo
