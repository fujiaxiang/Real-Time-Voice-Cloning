from functools import partial

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.emo_models import MultispeakerEmotionalSynthesizer
from synthesizer.iemocap_dataset import IemocapSynthesizerDataset, collate_fn
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import *
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import time
import platform


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def train(run_id: str, data_paths: dict, models_dir: Path, training_schedule: list, eval_every: int, save_every: int,
         backup_every: int, force_restart: bool, transfer: bool):
    writer = SummaryWriter()
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)
    
    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")

    print("Checkpoint path: {}".format(weights_fpath))

    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    # From WaveRNN/train_tacotron.py
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for session in training_schedule:
            _, _, _, batch_size, _ = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
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

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting training from scratch\n")
        model.save(weights_fpath)
    else:
        print("\nLoading weights at %s" % weights_fpath)

        if sys.platform == 'darwin':
            checkpoint = torch.load(weights_fpath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(weights_fpath)

        if transfer:
            # Make the necessary changes to reuse weights of speaker encoder for emotion encoder
            checkpoint['model_state']['step'] = checkpoint['model_state']['step'] * 0

            w = checkpoint['model_state']['encoder_proj.weight']
            w = torch.cat([w, w[:, -256:]], dim=-1)
            checkpoint['model_state']['encoder_proj.weight'] = w

            w = checkpoint['model_state']['decoder.attn_rnn.weight_ih']
            w = torch.cat([w, w[:, -256:]], dim=-1)
            checkpoint['model_state']['decoder.attn_rnn.weight_ih'] = w

            w = checkpoint['model_state']['decoder.rnn_input.weight']
            w = torch.cat([w, w[:, -256:]], dim=-1)
            checkpoint['model_state']['decoder.rnn_input.weight'] = w

            w = checkpoint['model_state']['decoder.stop_proj.weight']
            w = torch.cat([w[:, :512], w[:, 256:]], dim=-1)
            checkpoint['model_state']['decoder.stop_proj.weight'] = w

            model.load_state_dict(checkpoint["model_state"], strict=True)
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("Model weights loaded from step %d" % model.step)

    # Embeddings metadata
    char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
    with open(char_embedding_fpath, "w", encoding="utf-8") as f:
        for symbol in symbols:
            if symbol == " ":
                symbol = "\\s"  # For visual purposes, swap space with \s

            f.write("{}\n".format(symbol))

    # Initialize the dataset
    dataset = IemocapSynthesizerDataset(data_paths['train_meta'], data_paths['train_speaker_embeds'], data_paths['train_emotion_embeds'])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    for i, session in enumerate(training_schedule):
        current_step = model.get_step()

        r, lr, max_step, batch_size, batch_split = session

        training_steps = max_step - current_step

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(training_schedule) - 1:
                # We have completed training. Save the model and exit
                model.save(weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it
                continue

        model.r = r

        # Begin the training
        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        data_loader = DataLoader(dataset,
                                 collate_fn=partial(collate_fn, r=r),
                                 batch_size=batch_size // batch_split,
                                 num_workers=3 if platform.system().lower() == "linux" else 0,
                                 shuffle=True,
                                 pin_memory=True)

        total_iters = len(dataset) 
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        total, total_loss = 0, 0
        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            for i, (indices, texts, text_lengths, mels, mel_lengths, speaker_embeds, emotion_embeds) in enumerate(data_loader, 1):
                epoch_step = (i + batch_split - 1) // batch_split
                start_time = time.time()

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(mel_lengths):
                    stop[j, : k] = 0

                texts = texts.to(device)
                mels = mels.to(device)
                speaker_embeds = speaker_embeds.to(device)
                emotion_embeds = speaker_embeds.to(device)
                stop = stop.to(device)

                # Forward pass
                m1_hat, m2_hat, attention, stop_pred = model(texts, mels, speaker_embeds, emotion_embeds)
                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = (m1_loss + m2_loss + stop_loss) / batch_split

                loss.backward()
                total += texts.size(0)
                total_loss += loss.item() * texts.size(0)

                if i % batch_split == 0:
                    if hparams.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                        if np.isnan(grad_norm.cpu()):
                            print("grad_norm was NaN!")

                    optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                step = model.get_step()
                k = step // 1000

                msg = f"| Epoch: {epoch}/{epochs} ({epoch_step}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | {1./time_window.average:#.2} steps/s | Step: {k}k | "
                stream(msg)

                # Backup or save model as appropriate
                if backup_every != 0 and step % backup_every == 0 and i % batch_split == 0: 
                    backup_fpath = Path("{}/{}_{}k.pt".format(str(weights_fpath.parent), run_id, k))
                    model.save(backup_fpath, optimizer)

                if save_every != 0 and step % save_every == 0 and i % batch_split == 0: 
                    # Must save latest optimizer state to ensure that resuming training
                    # doesn't produce artifacts
                    model.save(weights_fpath, optimizer)

                # Evaluate model to generate samples
                epoch_eval = eval_every == -1 and i == len(data_loader)  # If epoch is done
                step_eval = eval_every > 0 and step % eval_every == 0  # Every N steps
                if epoch_eval or step_eval:
                    aver_loss = total_loss / total
                    total, total_loss = 0, 0
                    writer.add_scalar('Loss/train', aver_loss, step)

                    for sample_idx in range(hparams.tts_eval_num_samples):
                        # At most, generate samples equal to number in the batch
                        if sample_idx + 1 <= len(texts):
                            # Remove padding from mels using frame length in metadata
                            mel_length = int(dataset[indices[sample_idx]][2].shape[0])
                            mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                            target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                            attention_len = mel_length // model.r

                            eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                       mel_prediction=mel_prediction,
                                       target_spectrogram=target_spectrogram,
                                       input_seq=np_now(texts[sample_idx]),
                                       step=step,
                                       plot_dir=plot_dir,
                                       mel_output_dir=mel_output_dir,
                                       wav_dir=wav_dir,
                                       sample_num=sample_idx + 1,
                                       loss=loss,
                                       hparams=hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

            # Add line break after every epoch
            print("")


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))


if __name__ == "__main__":
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
    data_paths = {k: Path(v) for k, v in data_paths.items()}

    train(
        run_id="synthesizer_3",
        data_paths=data_paths,
        models_dir=Path("synthesizer/saved_models/"),
        training_schedule=[  # (r, lr, step, batch_size, batch_split)
            (2,  1e-4, 4_000,  12, 2),
            (2,  3e-5, 8_000,  12, 2),
            (2,  1e-5, 16_000,  12, 2)
        ],
        eval_every=10,
        save_every=1000,
        backup_every=500,
        force_restart=False,
        transfer=True
    )


# python -m synthesizer.train_emo

# rm synthesizer/saved_models/synthesizer_3 -r
# cp synthesizer/saved_models/pretrained/ synthesizer/saved_models/synthesizer_3 -r
# mv synthesizer/saved_models/synthesizer_3/pretrained.pt synthesizer/saved_models/synthesizer_3/synthesizer_3.pt
