"""Train the newborn world model and score the five held-out babies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audio_features import ensure_features
from config import EXAMPLE, N_EEG, N_FORECASTS, WIN
from data import NAMES, WindowDataset, load_eeg, load_split, melody_lengths
from model import WorldModel


def reference_stats(eeg_prev, eeg_cur):
    """Mean and standard deviation of the real 60 s context, per channel."""
    context = torch.cat([eeg_prev, eeg_cur], dim=1)
    mean = context.mean(dim=1)
    std = context.std(dim=1, unbiased=False).clamp_min(1e-5)
    return mean, std


def predict_minute(model, eeg, audio):
    """Two-step rollout. eeg is [B, 4, 3000, 63] and audio is [B, 4, 3000, 66]."""
    s0, s1 = eeg[:, 0], eeg[:, 1]
    a0, a1, a2, a3 = audio[:, 0], audio[:, 1], audio[:, 2], audio[:, 3]
    mean, std = reference_stats(s0, s1)
    first = model(s0, s1, a0, a1, a2, mean, std)
    second = model(s1, first, a1, a2, a3, mean, std)
    return first, second


def batch_loss(model, eeg, audio, teacher):
    s0, s1, s2, s3 = eeg[:, 0], eeg[:, 1], eeg[:, 2], eeg[:, 3]
    a0, a1, a2, a3 = audio[:, 0], audio[:, 1], audio[:, 2], audio[:, 3]
    mean, std = reference_stats(s0.detach(), s1.detach())
    first = model(s0, s1, a0, a1, a2, mean, std)
    second_input = s2 if teacher else first
    second = model(s1, second_input, a1, a2, a3, mean, std)
    loss1 = torch.nn.functional.mse_loss(first, s2)
    loss2 = torch.nn.functional.mse_loss(second, s3)
    return 0.5 * (loss1 + loss2), first.detach(), second.detach()


def score(pred, target):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return mae, mse


def spread(pred, target):
    """Mean per-channel standard deviation and Pearson correlation over time."""
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


def _blank_baby():
    return {"n": 0, "mae": 0.0, "mse": 0.0, "first_mae": 0.0, "second_mae": 0.0}


def evaluate(model, eeg, audio, subjects, channel_mean, device):
    model.eval()
    sums = np.zeros(10)
    shape = np.zeros(6)
    mismatch = np.zeros(4)
    babies = {str(subject): _blank_baby() for subject in subjects}
    n = 0
    wrong = {NAMES[i]: NAMES[(i + 1) % len(NAMES)] for i in range(len(NAMES))}
    with torch.no_grad():
        for name in NAMES:
            for subject in subjects:
                series = eeg[(name, subject)]
                if series.shape[0] < EXAMPLE:
                    raise RuntimeError(f"{name} subject {subject} is shorter than 120 s")
                eeg_w = torch.from_numpy(series[:EXAMPLE].reshape(1, 4, WIN, N_EEG)).to(device)
                audio_w = torch.from_numpy(audio[name][:EXAMPLE].reshape(1, 4, WIN, -1)).to(device)
                first, second = predict_minute(model, eeg_w, audio_w)
                first = first[0].cpu().numpy()
                second = second[0].cpu().numpy()
                other = audio[wrong[name]][:EXAMPLE]
                other_w = torch.from_numpy(other.reshape(1, 4, WIN, -1)).to(device)
                bad_first, bad_second = predict_minute(model, eeg_w, other_w)
                bad_first = bad_first[0].cpu().numpy()
                bad_second = bad_second[0].cpu().numpy()
                target1 = series[2 * WIN:3 * WIN]
                target2 = series[3 * WIN:4 * WIN]
                full_pred = np.concatenate([first, second], axis=0)
                full_tgt = np.concatenate([target1, target2], axis=0)
                held = np.repeat(series[2 * WIN - 1:2 * WIN], 2 * WIN, axis=0)
                mean_pred = np.broadcast_to(channel_mean, full_tgt.shape)
                sums += (
                    *score(first, target1),
                    *score(second, target2),
                    *score(full_pred, full_tgt),
                    *score(held, full_tgt),
                    *score(mean_pred, full_tgt),
                )
                shape += (*spread(first, target1), *spread(second, target2))
                mismatch += (
                    *score(bad_first, target1),
                    *score(bad_second, target2),
                )
                baby = babies[str(subject)]
                baby["n"] += 1
                baby["mae"] += score(full_pred, full_tgt)[0]
                baby["mse"] += score(full_pred, full_tgt)[1]
                baby["first_mae"] += score(first, target1)[0]
                baby["second_mae"] += score(second, target2)[0]
                n += 1
    per_baby = {}
    for subject, baby in babies.items():
        count = max(1, baby["n"])
        per_baby[subject] = {
            "trials": baby["n"],
            "mae": baby["mae"] / count,
            "mse": baby["mse"] / count,
            "first_30s_mae": baby["first_mae"] / count,
            "second_30s_mae": baby["second_mae"] / count,
        }
    return {
        "trials": n,
        "first_30s_mae": float(sums[0] / n),
        "first_30s_mse": float(sums[1] / n),
        "second_30s_mae": float(sums[2] / n),
        "second_30s_mse": float(sums[3] / n),
        "mae": float(sums[4] / n),
        "mse": float(sums[5] / n),
        "rmse": float(np.sqrt(sums[5] / n)),
        "persistence_mae": float(sums[6] / n),
        "persistence_mse": float(sums[7] / n),
        "train_mean_mae": float(sums[8] / n),
        "train_mean_mse": float(sums[9] / n),
        "first_30s_pred_std": float(shape[0] / n),
        "first_30s_target_std": float(shape[1] / n),
        "first_30s_correlation": float(shape[2] / n),
        "second_30s_pred_std": float(shape[3] / n),
        "second_30s_target_std": float(shape[4] / n),
        "second_30s_correlation": float(shape[5] / n),
        "mismatched_first_30s_mae": float(mismatch[0] / n),
        "mismatched_first_30s_mse": float(mismatch[1] / n),
        "mismatched_second_30s_mae": float(mismatch[2] / n),
        "mismatched_second_30s_mse": float(mismatch[3] / n),
        "per_baby": per_baby,
    }


def write_reports(args, epochs, n_train, fit_subjects, dev_subjects, test_subjects, dev_report, test_report):
    common = {
        "model": "newborn_world_model",
        "normalization": "fixed statistics from the real 60 s context, reused for both rollout steps",
        "music_input": True,
        "future_music_visible": True,
        "context_s": 60,
        "pred_s": 30,
        "rollouts": N_FORECASTS,
        "fit_subjects": fit_subjects,
        "dev_subjects": dev_subjects,
        "test_subjects": test_subjects,
        "train_windows": n_train,
        "epochs": epochs,
        "teacher_epochs": args.teacher_epochs,
    }
    dev_report.update(common)
    test_report.update(common)
    (args.output_dir / "dev.json").write_text(json.dumps(dev_report, indent=2) + "\n")
    (args.output_dir / "test.json").write_text(json.dumps(test_report, indent=2) + "\n")
    brief = ("mae", "mse", "first_30s_mae", "second_30s_mae", "first_30s_pred_std", "first_30s_target_std",
             "first_30s_correlation", "second_30s_pred_std", "second_30s_target_std", "second_30s_correlation",
             "mismatched_first_30s_mae", "mismatched_second_30s_mae", "train_mean_mae", "trials")
    print("DEV", json.dumps({key: dev_report[key] for key in brief}, indent=2), flush=True)
    print("TEST", json.dumps({key: test_report[key] for key in brief}, indent=2), flush=True)


def channel_mean(eeg, subjects):
    total = np.zeros(N_EEG, dtype=np.float64)
    count = 0
    for name in NAMES:
        for subject in subjects:
            series = eeg[(name, subject)]
            total += series.sum(axis=0)
            count += series.shape[0]
    return (total / count).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--teacher-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    train_subjects, test_subjects = load_split()
    if len(train_subjects) != 44 or len(test_subjects) != 5:
        raise RuntimeError(f"expected 44/5, got {len(train_subjects)}/{len(test_subjects)}")
    dev_subjects = [4, 11, 20, 33, 46]
    if any(subject not in train_subjects for subject in dev_subjects):
        raise RuntimeError(f"dev subjects are not inside the training pool: {dev_subjects}")
    fit_subjects = [subject for subject in train_subjects if subject not in set(dev_subjects)]
    print("building audio features", flush=True)
    lengths = melody_lengths()
    audio, _, _ = ensure_features(lengths)
    n_train = sum(
        len(range(0, length - EXAMPLE + 1, args.stride)) * len(fit_subjects)
        for length in lengths.values()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModel().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        saved = torch.load(args.output_dir / "last.pt", map_location=device)
        model.load_state_dict(saved["model"])
        mean = saved["channel_mean"]
        print(f"loaded epoch {saved.get('epoch')} train_mse {saved.get('train_mse')}", flush=True)
        print("loading development and test EEG", flush=True)
        heldout_eeg = load_eeg(dev_subjects + test_subjects)
        dev_report = evaluate(model, heldout_eeg, audio, dev_subjects, mean, device)
        test_report = evaluate(model, heldout_eeg, audio, test_subjects, mean, device)
        write_reports(args, saved.get("epoch", args.epochs), n_train, fit_subjects, dev_subjects, test_subjects, dev_report, test_report)
        return

    print("loading training EEG", flush=True)
    train_eeg = load_eeg(fit_subjects)
    mean = channel_mean(train_eeg, fit_subjects)
    dataset = WindowDataset(train_eeg, audio, fit_subjects, args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"windows={len(dataset)} stride_samples={args.stride}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        teacher = epoch <= args.teacher_epochs
        running = 0.0
        seen = 0
        for eeg, features in loader:
            eeg = eeg.to(device)
            features = features.to(device)
            loss, _, _ = batch_loss(model, eeg, features, teacher)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss) * eeg.shape[0]
            seen += eeg.shape[0]
        train_mse = running / max(1, seen)
        history.append({"epoch": epoch, "train_mse": train_mse, "teacher_forcing": teacher})
        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} teacher={teacher}", flush=True)
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "train_mse": train_mse, "channel_mean": mean},
            args.output_dir / "last.pt",
        )
    (args.output_dir / "train.json").write_text(json.dumps(history, indent=2) + "\n")

    del loader, dataset, train_eeg
    print("loading development and test EEG", flush=True)
    heldout_eeg = load_eeg(dev_subjects + test_subjects)
    dev_report = evaluate(model, heldout_eeg, audio, dev_subjects, mean, device)
    test_report = evaluate(model, heldout_eeg, audio, test_subjects, mean, device)
    write_reports(args, args.epochs, n_train, fit_subjects, dev_subjects, test_subjects, dev_report, test_report)


if __name__ == "__main__":
    main()
