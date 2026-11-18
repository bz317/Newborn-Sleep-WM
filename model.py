"""Action-conditioned EEG world model on 30-second windows."""
from __future__ import annotations

import torch
from torch import nn

from config import AUDIO_TOKENS, D_MODEL, FEAT_DIM, N_EEG, WIN


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_temporal = nn.Sequential(
            nn.Linear(WIN, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.electrode = nn.Embedding(N_EEG, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=D_MODEL * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.frame = nn.Sequential(
            nn.Linear(FEAT_DIM, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.audio_pool = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=10, stride=10)
        self.audio_time = nn.Parameter(torch.randn(AUDIO_TOKENS, D_MODEL) * 0.02)
        self.cross = nn.MultiheadAttention(D_MODEL, num_heads=4, dropout=0.1, batch_first=True)
        self.window = nn.Parameter(torch.randn(2, 1, D_MODEL) * 0.02)
        self.transition = nn.TransformerEncoder(layer, num_layers=2)
        self.forecast_mix = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Linear(D_MODEL, WIN)

    def encode_eeg(self, eeg, mean, std):
        """eeg [B, 3000, 63]. mean and std are [B, 63] from the real 60 s context."""
        series = eeg.transpose(1, 2)
        series = (series - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        tokens = self.eeg_temporal(series)
        tokens = tokens + self.electrode.weight
        return self.channel_encoder(tokens)

    def encode_audio(self, features):
        """features [B, 3000, 66], already z-scored. Causal 100 ms tokens [B, 300, 128]."""
        frames = self.frame(features).transpose(1, 2)
        tokens = self.audio_pool(frames).transpose(1, 2)
        return tokens + self.audio_time

    def _condition(self, eeg_tokens, audio_tokens):
        update, _ = self.cross(eeg_tokens, audio_tokens, audio_tokens, need_weights=False)
        return eeg_tokens + update

    def forward(self, eeg_prev, eeg_cur, audio_prev, audio_cur, audio_future, mean, std):
        """Predict the next 30 s of EEG in the original units.

        mean and std are fixed from the real 60 s context. They are reused when
        eeg_cur is a prediction, so a low-variance forecast cannot shrink the next step.
        audio_future is the real music during the window being predicted.
        """
        prev = self.encode_eeg(eeg_prev, mean, std)
        cur = self.encode_eeg(eeg_cur, mean, std)
        prev = self._condition(prev, self.encode_audio(audio_prev)) + self.window[0]
        cur = self._condition(cur, self.encode_audio(audio_cur)) + self.window[1]
        context = torch.cat([prev, cur], dim=1)
        context = self.transition(context)
        forecast = context[:, N_EEG:]
        forecast = self._condition(forecast, self.encode_audio(audio_future))
        forecast = self.forecast_mix(forecast)
        normalized = self.decoder(forecast)
        return (normalized * std.unsqueeze(-1) + mean.unsqueeze(-1)).transpose(1, 2)
