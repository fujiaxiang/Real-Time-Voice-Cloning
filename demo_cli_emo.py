from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

from functools import partial

from playsound import playsound
import pandas as pd

from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.emo_models import MultispeakerEmotionalSynthesizer
from synthesizer.utils.symbols import symbols
from synthesizer.iemocap_dataset import IemocapSynthesizerDataset, collate_fn
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.train_emo import np_now


def main(idx: int, env: str, seed: int, synthesizer_method: str, speaker_idx: int = None, emo_idx: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(f"iemocap_meta_{env}.csv").dropna()

    m_file = df['synthesizer_m_file'].iat[idx]
    mel = np.load(m_file)

    print('playing source audio...')
    # playsound(df['w_file'].iat[idx])

    torch.manual_seed(seed)
    # weights_fpath = Path("synthesizer/saved_models/synthesizer_3/synthesizer_3_15k.pt")
    # weights_fpath = Path("synthesizer/saved_models//synthesizer_2_140k.pt")
    weights_fpath = Path("synthesizer/saved_models//synthesizer_4_16k.pt")
    synthesizer = Synthesizer(weights_fpath)

    if synthesizer_method == 'raw':
        data_paths = {
            'train_meta': "iemocap_meta_train.csv",
            'dev_meta': "iemocap_meta_dev.csv",
            'test_meta': "iemocap_meta_test.csv",
            'train_speaker_embeds': "data/iemocap/synthesizer/speaker_enc_train.npy",
            'dev_speaker_embeds': "data/iemocap/synthesizer/speaker_enc_dev.npy",
            'test_speaker_embeds': "data/iemocap/synthesizer/speaker_enc_test.npy",
            'train_emotion_embeds': "data/iemocap/synthesizer/emotion_enc_train.npy",
            'dev_emotion_embeds': "data/iemocap/synthesizer/emotion_enc_dev.npy",
            'test_emotion_embeds': "data/iemocap/synthesizer/emotion_enc_test.npy"
        }
        dataset = IemocapSynthesizerDataset(data_paths[f'{env}_meta'], data_paths[f'{env}_speaker_embeds'], data_paths[f'{env}_emotion_embeds'])
        data_loader = DataLoader(dataset, collate_fn=partial(collate_fn, r=2), batch_size=1, num_workers=0, shuffle=False)
        for i, (indices, texts, text_lengths, mels, mel_lengths, speaker_embeds, emotion_embeds) in enumerate(data_loader):
            if i >= idx:
                break

        if speaker_idx is not None:
            for i, (_, _, _, _, _, speaker_embeds, _) in enumerate(data_loader):
                if i >= idx:
                    break

        if emo_idx is not None:
            for i, (_, _, _, _, _, _, emotion_embeds) in enumerate(data_loader):
                if i >= idx:
                    break

        # Generate stop tokens for training
        stop = torch.ones(mels.shape[0], mels.shape[2])
        for j, k in enumerate(mel_lengths):
            stop[j, : k] = 0

        texts = texts.to(device)
        mels = mels.to(device)
        speaker_embeds = speaker_embeds.to(device)
        emotion_embeds = speaker_embeds.to(device)
        stop = stop.to(device)

        if sys.platform == 'darwin':
            checkpoint = torch.load(weights_fpath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(weights_fpath)

        model = MultispeakerEmotionalSynthesizer(
            embed_dims=hparams.tts_embed_dims,
            num_chars=len(symbols),
            encoder_dims=hparams.tts_encoder_dims,
            decoder_dims=hparams.tts_decoder_dims,
            n_mels=hparams.num_mels,
            fft_bins=hparams.num_mels,
            postnet_dims=hparams.tts_postnet_dims,
            encoder_K=hparams.tts_encoder_K,
            lstm_dims=hparams.tts_lstm_dims,
            postnet_K=hparams.tts_postnet_K,
            num_highways=hparams.tts_num_highways,
            dropout=hparams.tts_dropout,
            stop_threshold=hparams.tts_stop_threshold,
            speaker_embedding_size=hparams.speaker_embedding_size,
            emotion_embedding_size=hparams.speaker_embedding_size  # same size for both types of embeddings
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        m1_hat, m2_hat, attention, stop_pred = model(texts, mels, speaker_embeds, emotion_embeds)

        mel_length = int(dataset[idx][2].shape[0])
        print('mel_length:', mel_length)
        generated_mel = np_now(m2_hat[0]).T[:mel_length].T
        # gen_target_mel = np_now(mels[0]).T[:mel_length].T
    else:
        speaker_embeds = np.load(f"data/iemocap/synthesizer/speaker_enc_{env}.npy")
        emotion_embeds = np.load(f"data/iemocap/synthesizer/emotion_enc_{env}.npy")
        text = df['text'].iat[idx]
        texts = [text]
        embeds = [np.concatenate([speaker_embeds[speaker_idx], emotion_embeds[emo_idx]])]

        generated_mels = synthesizer.synthesize_spectrograms(texts, embeds)
        generated_mel = generated_mels[0]

    save_dir = Path(f'demo_outputs/{env}/{idx}_{speaker_idx}_{emo_idx}')
    save_dir.mkdir(parents=True, exist_ok=True)

    mel_norm = audio._amp_to_db(mel, hparams) - hparams.ref_level_db
    mel_norm = audio._normalize(mel_norm, hparams)

    generated_mel_norm = audio._amp_to_db(generated_mel, hparams) - hparams.ref_level_db
    generated_mel_norm = audio._normalize(generated_mel_norm, hparams)

    plot_spectrogram(generated_mel.T, f'{save_dir}/generated_mel.png', title='generated_mel',
                     target_spectrogram=mel,
                     max_len=generated_mel.size // hparams.num_mels)

    plot_spectrogram(generated_mel_norm.T, f'{save_dir}/normalized.png', title='normalized_generated_mel',
                     target_spectrogram=mel_norm,
                     max_len=generated_mel_norm.size // hparams.num_mels)

    plot_spectrogram(mel_norm, f'{save_dir}/mel_target_vs_normalized_target.png', title='mel_target_vs_normalized_target',
                     target_spectrogram=mel,
                     max_len=mel_norm.size // hparams.num_mels)

    torch.manual_seed(seed)
    vocoder.load_model("vocoder/saved_models/pretrained/pretrained.pt")

    griffin_lim_wav_from_target_mel = audio.inv_mel_spectrogram(mel_norm.T, hparams)
    vocoder_wav_from_target_mel = vocoder.infer_waveform(mel_norm.T)

    print(generated_mel_norm.shape)
    generated_wav = vocoder.infer_waveform(generated_mel_norm)
    generated_wav2 = audio.inv_mel_spectrogram(generated_mel_norm, hparams)

    audio.save_wav(griffin_lim_wav_from_target_mel, f'{save_dir}/griffin_lim_wav_from_target_mel.wav', sr=hparams.sample_rate)
    audio.save_wav(vocoder_wav_from_target_mel, f'{save_dir}/vocoder_wav_from_target_mel.wav', sr=hparams.sample_rate)
    audio.save_wav(generated_wav, f'{save_dir}/vocoder_generated_wav.wav', sr=hparams.sample_rate)
    audio.save_wav(generated_wav2, f'{save_dir}/griffin_lim_generated_wav.wav', sr=hparams.sample_rate)

    print('playing vocoder generated audio...')
    # playsound(f'{save_dir}/vocoder_generated_wav.wav')

    print('playing griffin lim generated audio...')
    # playsound(f'{save_dir}/griffin_lim_generated_wav.wav')


if __name__ == '__main__':
    synthesizer_method = 'raw'
    seed = 0

    idx = 0
    speaker_idx = 0
    emo_idx = 0

    for idx in range(1, 5):
        speaker_idx = idx
        emo_idx = idx
        main(idx, 'test', seed, synthesizer_method=synthesizer_method, speaker_idx=speaker_idx, emo_idx=emo_idx)


# python demo_cli_emo.py
