from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from encoder import audio
from encoder.params_data import *


def proc_t_file(fp):
    res = []

    with fp.open("r") as f:
        for line in f:
            if not line.startswith('Ses'):
                continue

            meta, text = line.split(": ", 1)
            uttid, _ = meta.split(" ", 1)
            res.append({
                'uttid': uttid,
                'text': text.strip()
            })

    return pd.DataFrame(res)


def proc_cat_file(fp, labels):
    with fp.open("r") as f:
        for line in f:
            uttid, cats = line.split(" ", 1)
            cat = cats.split(";")[0]  # only takes primary category
            cat = cat[1:]  # removing ":" at the beginning
            cat = cat.split(" ")[0]  # only takes first descriptive word
            if uttid in labels:
                labels[uttid].append(cat)
            else:
                labels[uttid] = [cat]


def wav_to_mel(fp):

    wav = audio.preprocess_wav(fp)

    if len(wav) == 0:
        return None

    # Create the mel spectrogram, discard those that are too short
    frames = audio.wav_to_mel_spectrogram(wav)
    if len(frames) < partials_n_frames:
        return None

    return frames


data_root = Path('data/downloaded/IEMOCAP_full_release')
sessions = data_root.glob("Session*")

res = []
w_files = {}
labels = {}
for sess in sessions:
    trans = sess.joinpath('dialog').joinpath('transcriptions')

    for t_file in trans.glob("*.txt"):
        df = proc_t_file(t_file)
        df['session'] = str(sess)
        df['t_file'] = str(t_file)
        res.append(df)

    wavs_dir = sess.joinpath('sentences').joinpath('wav')
    for conv in wavs_dir.glob('Ses*'):
        for wav_file in conv.glob("*.wav"):
            uttid = wav_file.name.split(".")[0]
            w_file = str(wav_file)
            w_files[uttid] = w_file

    labels_dir = sess.joinpath('dialog').joinpath('EmoEvaluation').joinpath('Categorical')
    for cat_file in labels_dir.glob('*_cat.txt'):
        proc_cat_file(cat_file, labels)


df = pd.concat(res)
df['w_file'] = df['uttid'].map(w_files)
df['labels'] = df['uttid'].map(labels)
df = df.dropna()


output_dir = Path("data/iemocap/encoder/mel")
output_dir.mkdir(parents=True, exist_ok=True)


m_files = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    w_file = row['w_file']
    mel = wav_to_mel(w_file)

    if mel is None:
        m_files.append(np.nan)
    else:
        out_fpath = output_dir.joinpath(row['uttid'] + '.npy')
        np.save(out_fpath, mel)
        m_files.append(out_fpath)

df['m_file'] = m_files


df = df.explode('labels')
print(df.count())

df.to_csv("iemocap_meta.csv", index=False)


# python -m encoder.prep_iemocap
