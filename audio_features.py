"""Log-mel, envelope, and onset features on the EEG time grid."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import torch

from config import AUDIO_SR, FEATURE_DIR, FEAT_DIM, HOP, N_FFT, N_MELS, ROOT

FMIN = 20.0
FMAX = 8000.0
EPS = 1e-5
NAMES = [f"audio{i:02d}" for i in range(1, 11)] + ["shf01", "shf05", "shf08", "shf10"]


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    n_freqs = n_fft // 2 + 1
    points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(points) / sr).astype(np.int64)
    bins = np.clip(bins, 0, n_freqs - 1)
    bank = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_freqs)
        for freq in range(left, center):
            if 0 <= freq < n_freqs and center != left:
                bank[i, freq] = (freq - left) / (center - left)
        for freq in range(center, right):
            if 0 <= freq < n_freqs and right != center:
                bank[i, freq] = (right - freq) / (right - center)
    return bank.astype(np.float32)


def _read_wav(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    audio = raw.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return audio / 32768.0


def _to_length(values, n_frames):
    src = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(dst, src, values).astype(np.float32)
    out = np.empty((n_frames, values.shape[1]), dtype=np.float32)
    for band in range(values.shape[1]):
        out[:, band] = np.interp(dst, src, values[:, band])
    return out


def features_for_waveform(audio, n_frames, bank):
    """Return [n_frames, 66] at the same rate as the EEG."""
    wave_t = torch.from_numpy(np.ascontiguousarray(audio))
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        wave_t,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().square().numpy().astype(np.float32)
    mel = bank @ power
    log_mel = np.log(mel + EPS).T

    sq = wave_t.square().view(1, 1, -1)
    kernel = torch.ones(1, 1, N_FFT) / N_FFT
    pad = N_FFT // 2
    pooled = torch.nn.functional.conv1d(sq, kernel, stride=HOP, padding=pad)
    envelope = torch.log(pooled.squeeze().clamp_min(0).sqrt() + EPS).numpy().astype(np.float32)

    frames = min(log_mel.shape[0], envelope.shape[0])
    log_mel = _to_length(log_mel[:frames], n_frames)
    envelope = _to_length(envelope[:frames], n_frames)
    delta = np.diff(log_mel, axis=0, prepend=log_mel[:1])
    onset = np.maximum(delta, 0).sum(axis=1).astype(np.float32)
    return np.concatenate([log_mel, envelope[:, None], onset[:, None]], axis=1)


def wav_path(name):
    return ROOT / "STIMULI" / "test_wav_final_beep" / f"{name}.wav"


def ensure_features(lengths):
    """Build one [T, 66] array per melody and z-score it with training-melody statistics.

    The same 14 files are heard by every baby, so the statistics are the training distribution.
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    bank = mel_filterbank(AUDIO_SR, N_FFT, N_MELS, FMIN, FMAX)
    raw = {}
    for name, n_frames in lengths.items():
        cache = FEATURE_DIR / f"{name}.npy"
        if cache.is_file():
            features = np.load(cache)
            if features.shape != (n_frames, FEAT_DIM):
                features = None
            else:
                raw[name] = features
                continue
        audio = _read_wav(wav_path(name))
        features = features_for_waveform(audio, n_frames, bank)
        if features.shape != (n_frames, FEAT_DIM):
            raise RuntimeError(f"{name} features {features.shape} != {(n_frames, FEAT_DIM)}")
        np.save(cache, features)
        raw[name] = features
    stacked = np.concatenate([raw[name] for name in NAMES], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-5)
    (FEATURE_DIR / "norm.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}) + "\n"
    )
    return {name: ((raw[name] - mean) / std).astype(np.float32) for name in NAMES}, mean, std
