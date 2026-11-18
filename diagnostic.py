"""Time-aligned audio-to-EEG ridge model and a same-melody average.

Both are fit only on the babies used for training. A development group drawn
from the original 44 chooses the ridge penalty and the melody-average
shrinkage. The five held-out babies are scored once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_features import NAMES, ensure_features
from config import EXAMPLE, N_EEG, WIN
from data import load_eeg, load_split, melody_lengths

DEV_SUBJECTS = [4, 11, 20, 33, 46]
# Stimulus time minus EEG time, in 10 ms samples. Positive: sound happened earlier.
LAGS = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=np.int64)
N_COMPACT = 10
SHRINKAGE = [0, 1, 2, 5, 10, 20, 50]


def compact_features(features):
    bands = features[:, :64].reshape(features.shape[0], 8, 8).mean(axis=2)
    return np.concatenate([bands, features[:, 64:66]], axis=1).astype(np.float64)


def design(features, lags):
    pieces = []
    width = features.shape[1]
    for lag in lags:
        if lag > 0:
            pad = np.zeros((lag, width), dtype=np.float64)
            pieces.append(np.concatenate([pad, features[:-lag]], axis=0))
        elif lag < 0:
            pad = np.zeros((-lag, width), dtype=np.float64)
            pieces.append(np.concatenate([features[-lag:], pad], axis=0))
        else:
            pieces.append(features)
    lagged = np.concatenate(pieces, axis=1)
    return np.concatenate([lagged, np.ones((lagged.shape[0], 1), dtype=np.float64)], axis=1)


def valid_bounds(length, lags):
    start = int(max(0, lags.max()))
    stop = int(length - max(0, -lags.min()))
    return start, stop


def accumulate(eeg_map, features, subjects, lags):
    width = N_COMPACT * len(lags) + 1
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros((width, N_EEG), dtype=np.float64)
    count = 0
    for name in NAMES:
        compact = compact_features(features[name])
        matrix = design(compact, lags)
        start, stop = valid_bounds(compact.shape[0], lags)
        block = matrix[start:stop]
        gram = block.T @ block
        for subject in subjects:
            series = np.asarray(eeg_map[(name, subject)][start:stop], dtype=np.float64)
            xtx += gram
            xty += block.T @ series
            count += series.shape[0]
        print(f"fit {name} samples={count}", flush=True)
    return xtx, xty, count


def solve_ridge(xtx, xty, penalty):
    scale = np.eye(xtx.shape[0])
    scale[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty * scale, xty)


def predict_window(beta, features, start, stop, lags):
    matrix = design(compact_features(features), lags)
    return matrix[start:stop] @ beta


def score(pred, target):
    diff = pred - target
    return float(np.mean(np.abs(diff))), float(np.mean(diff ** 2))


def spread(pred, target):
    pred_std = float(pred.std(axis=0).mean())
    target_std = float(target.std(axis=0).mean())
    corrs = []
    for channel in range(pred.shape[1]):
        left = pred[:, channel]
        right = target[:, channel]
        if left.std() < 1e-8 or right.std() < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(left, right)[0, 1]))
    return pred_std, target_std, float(np.mean(corrs))


def melody_averages(eeg_map, subjects):
    sums = {}
    counts = {}
    channel_sum = np.zeros(N_EEG, dtype=np.float64)
    channel_count = 0
    for name in NAMES:
        for subject in subjects:
            series = np.asarray(eeg_map[(name, subject)], dtype=np.float64)
            sums[name] = sums.get(name, 0) + series
            counts[name] = counts.get(name, 0) + 1
            channel_sum += series.sum(axis=0)
            channel_count += series.shape[0]
    averages = {name: sums[name] / counts[name] for name in NAMES}
    return averages, counts, channel_sum / channel_count


def shrunk_average(average, count, grand, shrinkage):
    weight = count / (count + shrinkage)
    return weight * average + (1.0 - weight) * grand


def evaluate_predictions(predict_fn, eeg_map, subjects):
    sums = np.zeros(6)
    shape = np.zeros(6)
    babies = {str(subject): {"n": 0, "mae": 0.0, "mse": 0.0} for subject in subjects}
    n = 0
    for name in NAMES:
        for subject in subjects:
            target = np.asarray(eeg_map[(name, subject)][2 * WIN:4 * WIN], dtype=np.float64)
            pred = predict_fn(name, subject)
            first, second = pred[:WIN], pred[WIN:]
            target1, target2 = target[:WIN], target[WIN:]
            sums += (*score(first, target1), *score(second, target2), *score(pred, target))
            shape += (*spread(first, target1), *spread(second, target2))
            babies[str(subject)]["n"] += 1
            babies[str(subject)]["mae"] += score(pred, target)[0]
            babies[str(subject)]["mse"] += score(pred, target)[1]
            n += 1
    per_baby = {}
    for subject, baby in babies.items():
        count = max(1, baby["n"])
        per_baby[subject] = {"trials": baby["n"], "mae": baby["mae"] / count, "mse": baby["mse"] / count}
    return {
        "trials": n,
        "first_30s_mae": float(sums[0] / n),
        "first_30s_mse": float(sums[1] / n),
        "second_30s_mae": float(sums[2] / n),
        "second_30s_mse": float(sums[3] / n),
        "mae": float(sums[4] / n),
        "mse": float(sums[5] / n),
        "first_30s_pred_std": float(shape[0] / n),
        "first_30s_target_std": float(shape[1] / n),
        "first_30s_correlation": float(shape[2] / n),
        "second_30s_pred_std": float(shape[3] / n),
        "second_30s_target_std": float(shape[4] / n),
        "second_30s_correlation": float(shape[5] / n),
        "per_baby": per_baby,
    }


def smoke():
    rng = np.random.default_rng(0)
    length = 500
    envelope = rng.normal(size=length)
    features = np.zeros((length, N_COMPACT))
    features[:, 8] = envelope
    eeg = np.zeros((length, N_EEG))
    eeg[10:] = envelope[:-10, None]
    matrix = design(features, LAGS)
    start, stop = valid_bounds(length, LAGS)
    beta = solve_ridge(matrix[start:stop].T @ matrix[start:stop], matrix[start:stop].T @ eeg[start:stop], 1e-3)
    pred = matrix[100:200] @ beta
    corr = np.corrcoef(pred[:, 0], eeg[100:200, 0])[0, 1]
    if corr < 0.9:
        raise AssertionError(f"synthetic TRF correlation {corr}")
    print(f"diagnostic smoke ok correlation={corr:.3f} beta={beta.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return

    train_subjects, test_subjects = load_split()
    dev_subjects = list(DEV_SUBJECTS)
    fit_subjects = [subject for subject in train_subjects if subject not in set(dev_subjects)]
    print("loading features and fit EEG", flush=True)
    features, _, _ = ensure_features(melody_lengths())
    fit_eeg = load_eeg(fit_subjects)
    xtx, xty, count = accumulate(fit_eeg, features, fit_subjects, LAGS)
    averages, counts, grand = melody_averages(fit_eeg, fit_subjects)
    del fit_eeg
    penalties = (np.trace(xtx) / xtx.shape[0]) * np.logspace(-6, 2, 17)
    print("loading development EEG", flush=True)
    dev_eeg = load_eeg(dev_subjects)

    def trf_predictor(beta):
        def predict(name, _subject):
            return predict_window(beta, features[name], 2 * WIN, 4 * WIN, LAGS)
        return predict

    def average_predictor(shrinkage):
        def predict(name, _subject):
            waveform = shrunk_average(averages[name], counts[name], grand, shrinkage)
            return waveform[2 * WIN:4 * WIN]
        return predict

    best_penalty = None
    best_dev = None
    for penalty in penalties:
        beta = solve_ridge(xtx, xty, float(penalty))
        report = evaluate_predictions(trf_predictor(beta), dev_eeg, dev_subjects)
        print(f"dev penalty={penalty:.4e} mse={report['mse']:.6f} corr={report['first_30s_correlation']:.4f}", flush=True)
        if best_dev is None or report["mse"] < best_dev:
            best_dev = report["mse"]
            best_penalty = float(penalty)
    best_shrink = None
    best_shrink_mse = None
    for shrinkage in SHRINKAGE:
        report = evaluate_predictions(average_predictor(shrinkage), dev_eeg, dev_subjects)
        print(f"dev shrinkage={shrinkage} mse={report['mse']:.6f} corr={report['first_30s_correlation']:.4f}", flush=True)
        if best_shrink_mse is None or report["mse"] < best_shrink_mse:
            best_shrink_mse = report["mse"]
            best_shrink = shrinkage

    beta = solve_ridge(xtx, xty, best_penalty)
    print("loading test EEG", flush=True)
    test_eeg = load_eeg(test_subjects)
    groups = {
        "dev": (dev_eeg, dev_subjects),
        "test": (test_eeg, test_subjects),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, (eeg_map, subjects) in groups.items():
        trf = evaluate_predictions(trf_predictor(beta), eeg_map, subjects)
        wrong = {NAMES[i]: NAMES[(i + 1) % len(NAMES)] for i in range(len(NAMES))}

        def mismatched(name, _subject, mapping=wrong):
            return predict_window(beta, features[mapping[name]], 2 * WIN, 4 * WIN, LAGS)

        mismatch = evaluate_predictions(mismatched, eeg_map, subjects)
        melody = evaluate_predictions(average_predictor(best_shrink), eeg_map, subjects)
        raw = evaluate_predictions(average_predictor(0), eeg_map, subjects)
        payload = {
            "split": split,
            "subjects": subjects,
            "trf": trf,
            "trf_mismatched_music": mismatch,
            "same_melody_average": melody,
            "same_melody_average_unshrunk": raw,
            "ridge_penalty": best_penalty,
            "melody_shrinkage": best_shrink,
            "lags_samples": LAGS.tolist(),
            "compact_features": "8 log-mel bands, envelope, onset",
            "fit_samples": int(count),
            "fit_subjects": fit_subjects,
        }
        (args.output_dir / f"{split}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(
            split,
            "trf", trf["mae"], trf["first_30s_correlation"],
            "mismatch", mismatch["mae"],
            "melody", melody["mae"], melody["first_30s_correlation"],
            flush=True,
        )


if __name__ == "__main__":
    main()
