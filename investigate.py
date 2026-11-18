"""Separate mapping errors, a shared timed response, and within-baby reproducibility.

The pooled ridge and the melody template both estimate one across-baby waveform.
This script does not refit that forecast. It checks three different questions on
the 44 training babies. The five test babies are not read.

1. Do the released onset predictors, the note table, the wav attacks, the melody
   order, and the channel labels describe the same clock? The note table is timed
   from the first note. The authors add five seconds when they place those notes
   on a trial that already contains the leading silence. Wav features already
   start at the beginning of the file, so they do not receive another five-second
   shift.
2. Do averages from disjoint groups of babies agree at the same time, for the
   same melody, after a mismatched-melody control and a shifted-time control?
3. Does a within-baby model fit on some melodies predict that baby's other
   melodies, and does a model fit on other babies do the same? The positive
   control scores those predictions against the average of the other babies, which
   is the comparison used by Bianco et al., and also against the baby's own EEG.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from audio_features import NAMES
from config import FEATURE_DIR, FS, N_EEG, ROOT, WIN
from data import MELODIES, load_split, melody_lengths
from diagnostic import DEV_SUBJECTS, weighted_template

PREDICTORS = ["Spflux", "onset", "ITI", "IOI", "Soioiratio", "Eoioiratio", "Sppitch", "Eppitch"]
LAGS = np.arange(-5, 41, dtype=np.int64)
LAMBDAS = np.logspace(-4, 8, 13)
TEMPLATE_WEIGHTS = [round(float(weight), 2) for weight in np.linspace(0.0, 1.0, 11)]
EPOCH_START = -10
EPOCH_STOP = 51
BASELINE = slice(5, 10)
P2 = slice(34, 48)
FZ_FALLBACK = 1
STIM_PATH = ROOT / "STIMULI" / "FINAL_StimMatrix_both+.mat"
CSV_PATH = ROOT / "SCRIPTS" / "allinfo_stimuli_FINAL_StimMatrix_both+.csv"
WAV_DIR = ROOT / "STIMULI" / "test_wav_final_beep"


def mean_channel_corr(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(pred * target, axis=0)
    denominator = np.sqrt(np.sum(pred * pred, axis=0) * np.sum(target * target, axis=0))
    valid = denominator > 1e-8
    if not np.any(valid):
        return 0.0
    return float(np.mean(numerator[valid] / denominator[valid]))


def one_corr(left, right):
    if left.std() < 1e-8 or right.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def lagged_matrix(features):
    """Rows are EEG times. Column blocks follow LAGS. Positive lag: sound happened earlier."""
    start = int(max(0, LAGS.max()))
    stop = int(features.shape[0] - max(0, int(-LAGS.min())))
    width = features.shape[1]
    rows = stop - start
    matrix = np.empty((rows, width * len(LAGS) + 1), dtype=np.float64)
    for index, lag in enumerate(LAGS):
        matrix[:, index * width:(index + 1) * width] = features[start - lag:stop - lag]
    matrix[:, -1] = 1.0
    return matrix, start, stop


def solve_ridge(xtx, xty, lam):
    width = xtx.shape[0]
    scale = float(np.trace(xtx) / width)
    penalty = np.eye(width)
    penalty[-1, -1] = 0.0
    system = xtx + (float(lam) * scale) * penalty
    try:
        return np.linalg.solve(system, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, xty, rcond=None)[0]


def matlab_strings(path, key):
    with h5py.File(path, "r") as handle:
        refs = np.array(handle[key]).ravel()
        labels = []
        for ref in refs:
            chars = np.array(handle[ref]).astype(np.uint16).ravel()
            labels.append("".join(chr(int(char)) for char in chars if int(char) != 0))
    return labels


def wav_clock(name):
    path = WAV_DIR / f"{name}.wav"
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = np.frombuffer(handle.readframes(frames), dtype=np.int16)
    audio = raw.reshape(-1, channels).mean(axis=1).astype(np.float32)
    hop = rate // FS
    usable = audio.size // hop
    windowed = audio[:usable * hop].reshape(usable, hop)
    rms = np.sqrt(np.mean(windowed * windowed, axis=1))
    return {
        "rms": rms,
        "wav_samples": int(frames),
        "wav_rate": int(rate),
        "expected_eeg_samples": int(round(frames / rate * FS)),
    }


def mean_at_lag(values, indices, lag):
    shifted = indices + lag
    valid = (shifted >= 0) & (shifted < len(values))
    if not np.any(valid):
        return 0.0
    return float(values[shifted[valid]].mean())


def attack_curve(rms, indices):
    """The attack peak can sit one 10 ms frame after the nominal onset.

    A raw argmax over two seconds is not the test: dense piano flux has stray
    peaks. Alignment is a local peak at 0 or +10 ms that stands above the
    surrounding two seconds and above a five-second shift.
    """
    flux = np.maximum(np.diff(rms, prepend=rms[:1]), 0.0)
    lags = np.arange(-200, 201, dtype=np.int64)
    scores = np.array([mean_at_lag(flux, indices, int(lag)) for lag in lags])
    local = scores[198:203]
    local_lag = int(np.argmax(local) - 2)
    peak = float(local.max())
    background = float(np.median(scores))
    return {
        "local_peak_lag_samples": local_lag,
        "local_peak_flux": peak,
        "flux_at_0": float(scores[200]),
        "flux_at_plus_5s": mean_at_lag(flux, indices, 500),
        "flux_at_minus_5s": mean_at_lag(flux, indices, -500),
        "median_flux_within_2s": background,
        "peak_over_median": float(peak / background) if background > 0 else None,
        "global_best_lag_samples_within_2s": int(lags[int(np.argmax(scores))]),
    }


def nearest_impulse(note_times_s, impulses, offset):
    target = np.asarray(note_times_s, dtype=np.float64) * FS + offset
    position = np.searchsorted(impulses, target)
    right = np.clip(position, 0, len(impulses) - 1)
    left = np.clip(position - 1, 0, len(impulses) - 1)
    choose_right = np.abs(impulses[right] - target) <= np.abs(impulses[left] - target)
    chosen = np.where(choose_right, impulses[right], impulses[left])
    return chosen.astype(np.int64), np.abs(chosen - target)


def normalize_predictors(stim, eeg_length):
    """RMS-normalize the released predictors and pad a short tail with zeros.

    ITI and IOI use 99 on the first note as a missing value. That single sample
    would otherwise dominate the RMS, so those two entries are set to zero.
    """
    raw = np.array(stim, dtype=np.float64, copy=True)
    sentinel = int(np.isclose(raw[:, 2], 99).sum() + np.isclose(raw[:, 3], 99).sum())
    raw[np.isclose(raw[:, 2], 99), 2] = 0.0
    raw[np.isclose(raw[:, 3], 99), 3] = 0.0
    scale = np.sqrt(np.mean(raw * raw, axis=0))
    scale = np.maximum(scale, 1e-8)
    normalized = raw / scale
    aligned = np.zeros((eeg_length, raw.shape[1]), dtype=np.float64)
    count = min(normalized.shape[0], eeg_length)
    aligned[:count] = normalized[:count]
    return aligned, sentinel


def stimulus_alignment():
    mat = scipy.io.loadmat(STIM_PATH, squeeze_me=True)
    predictor_names = [str(item) for item in np.array(mat["pred_names"]).ravel()]
    stims = list(np.array(mat["stim_all"]).ravel())
    table = np.loadtxt(CSV_PATH, delimiter=",")
    lengths = melody_lengths()
    real_path = ROOT / MELODIES[0][0]
    shuf_path = ROOT / MELODIES[10][0]
    real_names = matlab_strings(real_path, "stimnames")
    shuf_names = matlab_strings(shuf_path, "stimnames")
    real_channels = matlab_strings(real_path, "norm_chanlbls")
    shuf_channels = matlab_strings(shuf_path, "norm_chanlbls")
    saved_channels = json.loads((ROOT / "verify_pairing.json").read_text())["channels"]
    bundles = {}
    melodies = []
    order_ok = True
    attacks_ok = True
    table_ok = True
    for index, name in enumerate(NAMES):
        stim = stims[index]
        clock = wav_clock(name)
        expected = clock["expected_eeg_samples"]
        eeg_length = int(lengths[name])
        order_match = abs(eeg_length - expected) <= 1
        order_ok = order_ok and order_match
        impulses = np.flatnonzero(stim[:, 1] > 0.5)
        notes = table[np.isclose(table[:, 0], index + 1)]
        offset_distance = [
            float(nearest_impulse(notes[:, 2], impulses, offset)[1].mean())
            for offset in range(0, 1001)
        ]
        best_offset = int(np.argmin(offset_distance))
        chosen, distances = nearest_impulse(notes[:, 2], impulses, 500)
        released = stim[chosen]
        feature_error = [
            float(np.max(np.abs(notes[:, 3 + column] - released[:, 2 + column])))
            for column in range(6)
        ]
        within_one = bool(np.max(distances) <= 1.0 and max(feature_error) < 1e-4)
        attacks = attack_curve(clock["rms"], impulses)
        peak_ratio = attacks["peak_over_median"] or 0.0
        attacks_ok = attacks_ok and abs(attacks["local_peak_lag_samples"]) <= 1
        attacks_ok = attacks_ok and peak_ratio >= 3.0
        attacks_ok = attacks_ok and attacks["local_peak_flux"] > attacks["flux_at_plus_5s"]
        cached_lag = None
        cache_path = FEATURE_DIR / f"{name}.npy"
        if cache_path.is_file():
            cached = np.load(cache_path)
            cached_lag = {
                "samples": int(cached.shape[0]),
                "matches_eeg_length": bool(cached.shape[0] == eeg_length),
                "local_peak_lag_samples": None,
                "onset_feature_at_0": None,
                "onset_feature_at_plus_5s": None,
            }
            if cached.ndim == 2 and cached.shape[1] >= 66 and cached.shape[0] == eeg_length:
                onset_feature = cached[:, 65]
                local_lags = np.arange(-2, 3)
                local_scores = [mean_at_lag(onset_feature, impulses, int(lag)) for lag in local_lags]
                cached_lag["local_peak_lag_samples"] = int(local_lags[int(np.argmax(local_scores))])
                cached_lag["onset_feature_at_0"] = mean_at_lag(onset_feature, impulses, 0)
                cached_lag["onset_feature_at_plus_5s"] = mean_at_lag(onset_feature, impulses, 500)
        table_ok = table_ok and best_offset == 500 and within_one
        features, sentinel = normalize_predictors(stim, eeg_length)
        bundles[name] = {
            "features": features,
            "onsets": impulses.astype(np.int64),
            "timing_surprise": stim[impulses, 4].astype(np.float64),
            "sentinel": sentinel,
        }
        melodies.append({
            "melody": name,
            "eeg_samples": eeg_length,
            "stim_samples": int(stim.shape[0]),
            "wav_samples": clock["wav_samples"],
            "wav_rate": clock["wav_rate"],
            "expected_eeg_samples": expected,
            "eeg_minus_expected": int(eeg_length - expected),
            "eeg_minus_stim": int(eeg_length - stim.shape[0]),
            "length_identity_ok": bool(order_match),
            "csv_notes": int(len(notes)),
            "released_onsets": int(len(impulses)),
            "best_csv_offset_samples": best_offset,
            "mean_distance_at_best_offset_samples": offset_distance[best_offset],
            "max_distance_at_5s_samples": float(np.max(distances)),
            "notes_farther_than_10ms": int(np.sum(distances > 1.0)),
            "max_abs_feature_error_at_nearest_impulse": feature_error,
            "first_released_onset_s": float(impulses[0] / FS) if len(impulses) else None,
            "wav_attacks": {key: value for key, value in attacks.items()},
            "cached_world_model_onset": cached_lag,
            "missing_iti_ioi_entries_zeroed": sentinel,
        })
        del clock
    fz = real_channels.index("Fz") if "Fz" in real_channels else FZ_FALLBACK
    report = {
        "predictor_names": predictor_names,
        "predictor_names_match": predictor_names == PREDICTORS,
        "real_stimnames": real_names,
        "shuf_file_stimnames": shuf_names,
        "stimnames_match_loader": real_names == NAMES,
        "channels": real_channels,
        "shuf_channels_match_real": real_channels == shuf_channels,
        "channels_match_verify_pairing": real_channels == saved_channels,
        "fz_index": int(fz),
        "loader_reorders_channels": False,
        "native_timestamps": None,
        "eeg_file_keys": ["realcondition or shufcondition", "stimnames", "norm_chanlbls"],
        "clock": {
            "midi_table_zero": "first note, before the leading silence",
            "released_predictor_zero": "first sample of the wav and of the EEG array",
            "measured_csv_to_predictor_offset_s": 5.0 if table_ok else None,
            "wav_loader_added_offset_s": 0.0,
            "extra_5s_shift_applied": False,
        },
        "melody_order_ok": bool(order_ok and real_names == NAMES),
        "note_table_matches_released_onsets": bool(table_ok),
        "wav_attacks_match_released_onsets": bool(attacks_ok),
        "melodies": melodies,
    }
    return report, bundles, fz


def group_mean(eeg, subjects, name, window):
    total = None
    for subject in subjects:
        series = np.asarray(eeg[(name, subject)][window], dtype=np.float64)
        total = series if total is None else total + series
    return total / len(subjects)


def shifted_corr(left, right, lag):
    if lag > 0:
        return mean_channel_corr(left[lag:], right[:-lag])
    if lag < 0:
        return mean_channel_corr(left[:lag], right[-lag:])
    return mean_channel_corr(left, right)


def split_report(eeg, left_subjects, right_subjects, window):
    left = [group_mean(eeg, left_subjects, name, window) for name in NAMES]
    right = [group_mean(eeg, right_subjects, name, window) for name in NAMES]
    matched = [mean_channel_corr(left[i], right[i]) for i in range(len(NAMES))]
    mismatched = []
    for i in range(len(NAMES)):
        other = right[(i + 1) % len(NAMES)]
        count = min(len(left[i]), len(other))
        mismatched.append(mean_channel_corr(left[i][:count], other[:count]))
    fz_matched = [one_corr(left[i][:, FZ_FALLBACK], right[i][:, FZ_FALLBACK]) for i in range(len(NAMES))]
    same_shape = all(item.shape == left[0].shape for item in left + right)
    residual = None
    common = None
    if same_shape:
        left_stack = np.stack(left)
        right_stack = np.stack(right)
        common = mean_channel_corr(left_stack.mean(axis=0), right_stack.mean(axis=0))
        left_resid = left_stack - left_stack.mean(axis=0, keepdims=True)
        right_resid = right_stack - right_stack.mean(axis=0, keepdims=True)
        residual_matched = [mean_channel_corr(left_resid[i], right_resid[i]) for i in range(len(NAMES))]
        residual_mismatched = [
            mean_channel_corr(left_resid[i], right_resid[(i + 1) % len(NAMES)])
            for i in range(len(NAMES))
        ]
        residual = {
            "matched_r": float(np.mean(residual_matched)),
            "mismatched_r": float(np.mean(residual_mismatched)),
            "per_melody_matched_r": residual_matched,
        }
    lags = list(range(-200, 201, 10)) + [-500, 500]
    lag_curve = []
    for lag in lags:
        lag_curve.append({
            "lag_samples": int(lag),
            "matched_r": float(np.mean([shifted_corr(left[i], right[i], lag) for i in range(len(NAMES))])),
        })
    circular = float(np.mean([
        mean_channel_corr(left[i], np.roll(right[i], 3000, axis=0)) for i in range(len(NAMES))
    ])) if same_shape and left[0].shape[0] >= 6000 else None
    return {
        "matched_r": float(np.mean(matched)),
        "mismatched_r": float(np.mean(mismatched)),
        "fz_matched_r": float(np.mean(fz_matched)),
        "real_matched_r": float(np.mean(matched[:10])),
        "shuffled_matched_r": float(np.mean(matched[10:])),
        "common_time_from_start_r": common,
        "melody_specific": residual,
        "circular_shift_30s_r": circular,
        "lag_curve": lag_curve,
        "per_melody_matched_r": matched,
        "per_melody_mismatched_r": mismatched,
    }


def beep_profile(eeg, subjects):
    total = np.zeros(400, dtype=np.float64)
    count = 0
    for name in NAMES:
        for subject in subjects:
            total += np.asarray(eeg[(name, subject)][:400, FZ_FALLBACK], dtype=np.float64)
            count += 1
    waveform = total / count
    return {
        "fz_mean_abs_0.05_to_0.40s": float(np.mean(np.abs(waveform[5:40]))),
        "fz_mean_abs_2.50_to_4.00s": float(np.mean(np.abs(waveform[250:400]))),
        "waveform_0_to_4s": waveform.tolist(),
    }


def partitions_of(subjects):
    ordered = list(subjects)
    half = len(ordered) // 2
    permutation = np.random.default_rng(1).permutation(np.array(ordered, dtype=np.int64))
    return {
        "even_odd": (ordered[0::2], ordered[1::2]),
        "first_second": (ordered[:half], ordered[half:]),
        "seed1": (sorted(int(item) for item in permutation[:half]), sorted(int(item) for item in permutation[half:])),
    }


def add_epochs(store, key, epochs):
    if len(epochs) == 0:
        return
    store["sum"][key] += epochs.sum(axis=0)
    store["count"][key] += int(len(epochs))


def new_store(keys):
    length = EPOCH_STOP - EPOCH_START
    return {
        "sum": {key: np.zeros((length, N_EEG), dtype=np.float64) for key in keys},
        "count": {key: 0 for key in keys},
    }


def epochs_from(series, onsets):
    onsets = np.asarray(onsets, dtype=np.int64)
    eligible = (onsets >= -EPOCH_START) & (onsets + EPOCH_STOP - 1 < len(series))
    chosen = onsets[eligible]
    if len(chosen) == 0:
        empty = np.zeros((0, EPOCH_STOP - EPOCH_START, N_EEG))
        return empty, eligible, np.zeros(0, dtype=bool)
    offsets = np.arange(EPOCH_START, EPOCH_STOP)
    extracted = np.asarray(series[chosen[:, None] + offsets], dtype=np.float64)
    center = series.mean(axis=0)
    scale = np.maximum(series.std(axis=0), 1e-6)
    rejected = np.mean(np.abs(extracted - center) / scale > 2.5, axis=(1, 2)) > 0.02
    kept = extracted[~rejected]
    if len(kept):
        kept = kept - kept[:, BASELINE, :].mean(axis=1, keepdims=True)
    return kept, eligible, rejected


def finalize_erp(store):
    output = {}
    for key, total in store["sum"].items():
        count = store["count"][key]
        waveform = total / count if count else total
        output[key] = {
            "epochs": int(count),
            "fz": waveform[:, FZ_FALLBACK].tolist(),
            "p2_fz": float(waveform[P2, FZ_FALLBACK].mean()) if count else 0.0,
        }
    return output


def loo_beta(xtx, xty, holdout, lam):
    return solve_ridge(xtx.sum(axis=0) - xtx[holdout], xty.sum(axis=0) - xty[holdout], lam)


def pooled_leave_one_melody(xtx, xty):
    """Sum of every baby's gram, excluding one melody, still including every baby."""
    melody_x = sum(value for value in xtx.values())
    melody_y = sum(value for value in xty.values())
    total_x = melody_x.sum(axis=0)
    total_y = melody_y.sum(axis=0)
    return [total_x - melody_x[i] for i in range(len(NAMES))], [total_y - melody_y[i] for i in range(len(NAMES))]


def score_model(kind, subjects, xtx, xty, eeg, designs, sums, lam, peer_count, pooled):
    own = []
    peer = []
    everyone = []
    benchmark_own = []
    per_baby = []
    real = []
    shuffled = []
    maes = []
    pooled_x, pooled_y = pooled
    for subject in subjects:
        baby_own = []
        baby_peer = []
        for index, name in enumerate(NAMES):
            if kind == "within":
                beta = loo_beta(xtx[subject], xty[subject], index, lam)
            else:
                mine_x = xtx[subject].sum(axis=0) - xtx[subject][index]
                mine_y = xty[subject].sum(axis=0) - xty[subject][index]
                beta = solve_ridge(pooled_x[index] - mine_x, pooled_y[index] - mine_y, lam)
            matrix, start, stop = designs[name]
            prediction = matrix @ beta
            target = np.asarray(eeg[(name, subject)][start:stop], dtype=np.float64)
            own_r = mean_channel_corr(prediction, target)
            peer_target = (sums[name][start:stop] - target) / peer_count
            all_target = sums[name][start:stop] / (peer_count + 1)
            peer_r = mean_channel_corr(prediction, peer_target)
            all_r = mean_channel_corr(prediction, all_target)
            own.append(own_r)
            peer.append(peer_r)
            everyone.append(all_r)
            baby_own.append(own_r)
            baby_peer.append(peer_r)
            full = np.zeros((len(eeg[(name, subject)]), N_EEG), dtype=np.float64)
            full[start:stop] = prediction
            window_pred = full[2 * WIN:4 * WIN]
            window_target = np.asarray(eeg[(name, subject)][2 * WIN:4 * WIN], dtype=np.float64)
            benchmark_own.append(mean_channel_corr(window_pred, window_target))
            maes.append(float(np.mean(np.abs(window_pred - window_target))))
            if index < 10:
                real.append(own_r)
            else:
                shuffled.append(own_r)
        per_baby.append({
            "subject": int(subject),
            "own_r": float(np.mean(baby_own)),
            "peer_r": float(np.mean(baby_peer)),
        })
    return {
        "own_r": float(np.mean(own)),
        "peer_excluding_self_r": float(np.mean(peer)),
        "all_babies_average_r": float(np.mean(everyone)),
        "benchmark_minute_own_r": float(np.mean(benchmark_own)),
        "benchmark_minute_mae": float(np.mean(maes)),
        "real_melodies_own_r": float(np.mean(real)),
        "shuffled_melodies_own_r": float(np.mean(shuffled)),
        "per_baby": per_baby,
    }


def shifted_within(subjects, xtx, xty, eeg, shifted_designs, sums, lam, peer_count):
    own = []
    peer = []
    for subject in subjects:
        for index, name in enumerate(NAMES):
            beta = loo_beta(xtx[subject], xty[subject], index, lam)
            matrix, start, stop = shifted_designs[name]
            prediction = matrix @ beta
            target = np.asarray(eeg[(name, subject)][start:stop], dtype=np.float64)
            own.append(mean_channel_corr(prediction, target))
            peer_target = (sums[name][start:stop] - target) / peer_count
            peer.append(mean_channel_corr(prediction, peer_target))
    return {"own_r": float(np.mean(own)), "peer_excluding_self_r": float(np.mean(peer))}


def lambda_curve(kind, subjects, xtx, xty, eeg, designs, pooled):
    curve = []
    pooled_x, pooled_y = pooled
    for lam in LAMBDAS:
        values = []
        for subject in subjects:
            for index, name in enumerate(NAMES):
                if kind == "within":
                    beta = loo_beta(xtx[subject], xty[subject], index, float(lam))
                else:
                    mine_x = xtx[subject].sum(axis=0) - xtx[subject][index]
                    mine_y = xty[subject].sum(axis=0) - xty[subject][index]
                    beta = solve_ridge(pooled_x[index] - mine_x, pooled_y[index] - mine_y, float(lam))
                matrix, start, stop = designs[name]
                prediction = matrix @ beta
                target = np.asarray(eeg[(name, subject)][start:stop], dtype=np.float64)
                values.append(mean_channel_corr(prediction, target))
        curve.append({"lambda": float(lam), "own_r": float(np.mean(values))})
        print(f"dev {kind} lambda={lam:.3e} own_r={curve[-1]['own_r']:.5f}", flush=True)
    best = max(curve, key=lambda item: item["own_r"])
    return curve, best["lambda"]


def run_eeg(bundles, fz, output_dir):
    global FZ_FALLBACK
    FZ_FALLBACK = int(fz)
    train_subjects, test_subjects = load_split()
    del test_subjects
    train_subjects = list(train_subjects)
    dev_subjects = [subject for subject in DEV_SUBJECTS if subject in set(train_subjects)]
    fit_subjects = [subject for subject in train_subjects if subject not in set(dev_subjects)]
    eeg = {}
    xtx = {subject: [] for subject in train_subjects}
    xty = {subject: [] for subject in train_subjects}
    sums = {}
    keys = ["onset", "shifted_2s", "high_st_real", "low_st_real", "high_st_shuf", "low_st_shuf"]
    even = set(train_subjects[0::2])
    erp_all = new_store(keys)
    erp_even = new_store(keys)
    erp_odd = new_store(keys)
    considered = 0
    rejected = 0
    for path, key, cell, name in MELODIES:
        print(f"loading {name}", flush=True)
        with h5py.File(ROOT / path, "r") as handle:
            ref = np.array(handle[key]).ravel()[cell]
            block = np.asarray(handle[ref], dtype=np.float32)
        bundle = bundles[name]
        matrix, _, _ = lagged_matrix(bundle["features"])
        gram = matrix.T @ matrix
        total = None
        for subject in train_subjects:
            series = np.ascontiguousarray(block[subject].T)
            eeg[(name, subject)] = series
            response = np.asarray(series, dtype=np.float64)
            start = int(max(0, LAGS.max()))
            stop = int(len(series) - max(0, int(-LAGS.min())))
            xtx[subject].append(gram)
            xty[subject].append(matrix.T @ response[start:stop])
            total = response if total is None else total + response
            kept, eligible, bad = epochs_from(series, bundle["onsets"])
            shifted, _, _ = epochs_from(series, bundle["onsets"] + 200)
            considered += int(eligible.sum())
            rejected += int(bad.sum())
            high_epochs = kept[:0]
            low_epochs = kept[:0]
            if len(kept):
                surprise = bundle["timing_surprise"][eligible][~bad]
                high_cut = np.quantile(bundle["timing_surprise"], 0.8)
                low_cut = np.quantile(bundle["timing_surprise"], 0.2)
                high_epochs = kept[surprise >= high_cut]
                low_epochs = kept[surprise <= low_cut]
            label = "real" if name in set(NAMES[:10]) else "shuf"
            for store in (erp_all, erp_even if subject in even else erp_odd):
                add_epochs(store, "onset", kept)
                add_epochs(store, "shifted_2s", shifted)
                add_epochs(store, f"high_st_{label}", high_epochs)
                add_epochs(store, f"low_st_{label}", low_epochs)
        sums[name] = total
        del block
        print(f"stored {name} subjects={len(train_subjects)}", flush=True)
    for subject in train_subjects:
        xtx[subject] = np.stack(xtx[subject])
        xty[subject] = np.stack(xty[subject])

    grand_sum = np.zeros(N_EEG, dtype=np.float64)
    grand_count = 0
    averages = {}
    for name in NAMES:
        total = None
        for subject in fit_subjects:
            series = np.asarray(eeg[(name, subject)], dtype=np.float64)
            total = series if total is None else total + series
            grand_sum += series.sum(axis=0)
            grand_count += series.shape[0]
        averages[name] = total / len(fit_subjects)
    grand = grand_sum / grand_count
    weight_rows = []
    for weight in TEMPLATE_WEIGHTS:
        absolute = []
        square = []
        corrs = []
        for name in NAMES:
            prediction = weighted_template(averages[name][2 * WIN:4 * WIN], grand, weight)
            for subject in dev_subjects:
                target = np.asarray(eeg[(name, subject)][2 * WIN:4 * WIN], dtype=np.float64)
                difference = prediction - target
                absolute.append(float(np.mean(np.abs(difference))))
                square.append(float(np.mean(difference ** 2)))
                corrs.append(mean_channel_corr(prediction, target))
        weight_rows.append({
            "weight": float(weight),
            "mae": float(np.mean(absolute)),
            "mse": float(np.mean(square)),
            "correlation": float(np.mean(corrs)),
        })
        print(f"dev template w={weight:.2f} mse={weight_rows[-1]['mse']:.6f}", flush=True)
    best_weight = min(weight_rows, key=lambda item: item["mse"])
    (output_dir / "template_weights.json").write_text(json.dumps({
        "scored_on": "development babies, forecast minute 60-120 s",
        "fit_subjects": fit_subjects,
        "dev_subjects": dev_subjects,
        "weights": weight_rows,
        "selected": best_weight,
        "note": "w=0 is the constant channel mean. Correlation is undefined there and is reported as 0.",
    }, indent=2) + "\n")

    split_payload = {"window": {}, "full_melody": {}, "beep": {}}
    groups = partitions_of(train_subjects)
    for label, (left, right) in groups.items():
        print(f"split-half {label}", flush=True)
        split_payload["window"][label] = split_report(eeg, left, right, slice(2 * WIN, 4 * WIN))
        split_payload["full_melody"][label] = split_report(eeg, left, right, slice(None))
    split_payload["beep"] = {
        "all_training_babies": beep_profile(eeg, train_subjects),
        "even": beep_profile(eeg, train_subjects[0::2]),
        "odd": beep_profile(eeg, train_subjects[1::2]),
    }
    even_wave = np.array(split_payload["beep"]["even"]["waveform_0_to_4s"])
    odd_wave = np.array(split_payload["beep"]["odd"]["waveform_0_to_4s"])
    split_payload["beep"]["even_odd_fz_r_0_to_1s"] = one_corr(even_wave[:100], odd_wave[:100])
    split_payload["beep"]["even_odd_fz_r_2.5_to_4s"] = one_corr(even_wave[250:400], odd_wave[250:400])
    split_payload["subjects"] = {"train": train_subjects, "partitions": {key: {"left": left, "right": right} for key, (left, right) in groups.items()}}
    split_payload["headline_even_odd_forecast_minute"] = split_payload["window"]["even_odd"]
    (output_dir / "splithalf.json").write_text(json.dumps(split_payload, indent=2) + "\n")

    designs = {name: lagged_matrix(bundles[name]["features"]) for name in NAMES}
    peer_count = len(train_subjects) - 1
    pooled = pooled_leave_one_melody(xtx, xty)
    within_curve, within_lambda = lambda_curve("within", dev_subjects, xtx, xty, eeg, designs, pooled)
    across_curve, across_lambda = lambda_curve("across", dev_subjects, xtx, xty, eeg, designs, pooled)
    print("scoring fit babies", flush=True)
    within_fit = score_model("within", fit_subjects, xtx, xty, eeg, designs, sums, within_lambda, peer_count, pooled)
    across_fit = score_model("across", fit_subjects, xtx, xty, eeg, designs, sums, across_lambda, peer_count, pooled)
    shifted_designs = {}
    for name in NAMES:
        rolled = np.roll(bundles[name]["features"], 500, axis=0)
        shifted_designs[name] = lagged_matrix(rolled)
    shifted = shifted_within(fit_subjects, xtx, xty, eeg, shifted_designs, sums, within_lambda, peer_count)
    trf_payload = {
        "features": PREDICTORS,
        "lags_ms": [-50, 400],
        "lambda_grid": LAMBDAS.tolist(),
        "lambda_selected_on": "development babies, correlation with each baby's own EEG",
        "within_lambda": within_lambda,
        "across_lambda": across_lambda,
        "within_dev_curve": within_curve,
        "across_dev_curve": across_curve,
        "fit_within": within_fit,
        "fit_across": across_fit,
        "fit_within_features_shifted_plus_5s": shifted,
        "fit_subjects": fit_subjects,
        "dev_subjects": dev_subjects,
        "test_subjects_not_used": True,
    }
    (output_dir / "trf.json").write_text(json.dumps(trf_payload, indent=2) + "\n")

    erp_payload = {
        "fz_index": int(fz),
        "epoch_s": [EPOCH_START / FS, (EPOCH_STOP - 1) / FS],
        "baseline_s": [-0.05, -0.01],
        "p2_window_s": [0.24, 0.37],
        "rejection_rule": "drop an epoch if more than 2% of its samples exceed 2.5 standard deviations of that recording",
        "rejection": {"candidate_epochs": considered, "rejected_epochs": rejected},
        "all_training_babies": finalize_erp(erp_all),
        "even": finalize_erp(erp_even),
        "odd": finalize_erp(erp_odd),
    }
    even_fz = np.array(erp_payload["even"]["onset"]["fz"])
    odd_fz = np.array(erp_payload["odd"]["onset"]["fz"])
    even_shift = np.array(erp_payload["even"]["shifted_2s"]["fz"])
    odd_shift = np.array(erp_payload["odd"]["shifted_2s"]["fz"])
    erp_payload["even_odd_fz_r"] = {
        "note_onset": one_corr(even_fz, odd_fz),
        "shifted_2s": one_corr(even_shift, odd_shift),
    }
    high_real = erp_payload["all_training_babies"]["high_st_real"]["p2_fz"]
    low_real = erp_payload["all_training_babies"]["low_st_real"]["p2_fz"]
    high_shuf = erp_payload["all_training_babies"]["high_st_shuf"]["p2_fz"]
    low_shuf = erp_payload["all_training_babies"]["low_st_shuf"]["p2_fz"]
    erp_payload["p2_high_minus_low"] = {
        "real": high_real - low_real,
        "shuffled": high_shuf - low_shuf,
    }
    (output_dir / "erp.json").write_text(json.dumps(erp_payload, indent=2) + "\n")
    print("eeg checks written", flush=True)


def smoke():
    features = np.zeros((180, 8), dtype=np.float64)
    features[20::15, 1] = 1.0
    features[:, 1] /= np.sqrt(np.mean(features[:, 1] ** 2))
    eeg = np.zeros((180, N_EEG), dtype=np.float64)
    eeg[30::15, 0] = 1.0
    matrix, start, stop = lagged_matrix(features)
    beta = solve_ridge(matrix.T @ matrix, matrix.T @ eeg[start:stop], 1e-3)
    recovered = mean_channel_corr(matrix @ beta, eeg[start:stop])
    if recovered < 0.9:
        raise AssertionError(f"delayed impulse correlation {recovered}")
    lag_index = int(np.where(LAGS == 10)[0][0])
    onset_column = lag_index * 8 + 1
    if abs(beta[onset_column, 0]) <= abs(beta[0, 0]):
        raise AssertionError("ridge did not use the 100 ms lag")
    average = np.ones((4, 3))
    grand = np.array([2.0, 0.0, -1.0])
    if not np.allclose(weighted_template(average, grand, 0.0), grand):
        raise AssertionError("w=0 is not the constant mean")
    if not np.allclose(weighted_template(average, grand, 1.0), average):
        raise AssertionError("w=1 is not the template")
    raw = np.zeros((20, 8))
    raw[0, 2] = 99
    raw[0, 3] = 99
    raw[5, 1] = 1
    normalized, sentinel = normalize_predictors(raw, 25)
    if sentinel != 2 or normalized[0, 2] != 0 or normalized[0, 3] != 0:
        raise AssertionError("missing ITI/IOI sentinel was kept")
    if normalized.shape != (25, 8):
        raise AssertionError(normalized.shape)
    print(f"investigate smoke ok correlation={recovered:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/investigate"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stimulus-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, bundles, fz = stimulus_alignment()
    (args.output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "alignment",
        "order", report["melody_order_ok"],
        "table", report["note_table_matches_released_onsets"],
        "attacks", report["wav_attacks_match_released_onsets"],
        flush=True,
    )
    if args.stimulus_only:
        return
    run_eeg(bundles, fz, args.output_dir)


if __name__ == "__main__":
    main()
