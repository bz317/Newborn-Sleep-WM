"""Constants for the newborn action-conditioned world model."""
from pathlib import Path

ROOT = Path("/home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/WM_Biomedicine/data/sleep_data/newborn_music_bianco")
SPLIT_PATH = ROOT / "split.json"
FEATURE_DIR = ROOT / "audio_features"

FS = 100
WIN_S = 30
WIN = WIN_S * FS
N_EEG = 63
N_MELS = 64
FEAT_DIM = N_MELS + 2
D_MODEL = 128
AUDIO_TOKENS = WIN // 10
AUDIO_SR = 44100
N_FFT = 1024
HOP = 441
N_FORECASTS = 2
CONTEXT_WINDOWS = 2
EXAMPLE_WINDOWS = CONTEXT_WINDOWS + N_FORECASTS
EXAMPLE = EXAMPLE_WINDOWS * WIN
