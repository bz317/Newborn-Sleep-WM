"""Sliding 120-second examples from the Bianco newborn recordings."""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from audio_features import NAMES, ensure_features
from config import EXAMPLE, N_EEG, ROOT, SPLIT_PATH, WIN

MELODIES = [
    ("DATA_eegTRF/EEGdata_realcondition.mat", "realcondition", index, name)
    for index, name in enumerate(NAMES[:10])
] + [
    ("DATA_eegTRF/EEGdata_shufcondition.mat", "shufcondition", index, name)
    for index, name in enumerate(NAMES[10:])
]


def load_split():
    split = json.loads(SPLIT_PATH.read_text())
    return split["train_subjects"], split["test_subjects"]


def melody_lengths():
    lengths = {}
    opened = {}
    try:
        for path, key, cell, name in MELODIES:
            if path not in opened:
                opened[path] = h5py.File(ROOT / path, "r")
            ref = np.array(opened[path][key]).flatten()[cell]
            lengths[name] = int(opened[path][ref].shape[-1])
    finally:
        for handle in opened.values():
            handle.close()
    return lengths


def load_eeg(subjects):
    """EEG shaped (time, 63), keyed by (melody, subject)."""
    wanted = set(subjects)
    series = {}
    opened = {}
    try:
        for path, key, cell, name in MELODIES:
            if path not in opened:
                opened[path] = h5py.File(ROOT / path, "r")
            ref = np.array(opened[path][key]).flatten()[cell]
            block = np.asarray(opened[path][ref], dtype=np.float32)
            if block.shape[1] != N_EEG:
                raise RuntimeError(f"{name} has shape {block.shape}")
            for subject in wanted:
                series[(name, subject)] = np.ascontiguousarray(block[subject].T)
    finally:
        for handle in opened.values():
            handle.close()
    return series


class WindowDataset(Dataset):
    def __init__(self, eeg, audio, subjects, stride):
        self.eeg = eeg
        self.audio = audio
        self.index = []
        for name in NAMES:
            for subject in subjects:
                length = eeg[(name, subject)].shape[0]
                for start in range(0, length - EXAMPLE + 1, stride):
                    self.index.append((name, subject, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        name, subject, start = self.index[item]
        eeg = self.eeg[(name, subject)][start:start + EXAMPLE]
        audio = self.audio[name][start:start + EXAMPLE]
        eeg_windows = torch.from_numpy(eeg.reshape(4, WIN, N_EEG).copy())
        audio_windows = torch.from_numpy(audio.reshape(4, WIN, -1).copy())
        return eeg_windows, audio_windows
