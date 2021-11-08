import torch

from synthesizer.models.tacotron import Tacotron


class MultispeakerEmotionalSynthesizer(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels,
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size, emotion_embedding_size):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels,
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size + emotion_embedding_size)

    def forward(self, x, m, speaker_embedding, emotion_embedding=None):
        # Concatenating the speaker embedding and emotion_embedding,
        # and reusing existing model architecture defined in super class
        embed = torch.cat([speaker_embedding, emotion_embedding], dim=1)
        return super().forward(x, m, embed)
