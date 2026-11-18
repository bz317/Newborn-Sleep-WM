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


def predict_minute(model, eeg, audio):
    """Two-step rollout. eeg is [B, 4, 3000, 63] and audio is [B, 4, 3000, 66]."""
    s0, s1 = eeg[:, 0], eeg[:, 1]
    a0, a1, a2, a3 = audio[:, 0], audio[:, 1], audio[:, 2], audio[:, 3]
    first = model(s0, s1, a0, a1, a2)
    second = model(s1, first, a1, a2, a3)
    return first, second


def batch_loss(model, eeg, audio, teacher):
    s0, s1, s2, s3 = eeg[:, 0], eeg[:, 1], eeg[:, 2], eeg[:, 3]
    a0, a1, a2, a3 = audio[:, 0], audio[:, 1], audio[:, 2], audio[:, 3]
    first = model(s0, s1, a0, a1, a2)
    second_input = s2 if teacher else first
    second = model(s1, second_input, a1, a2, a3)
    loss1 = torch.nn.functional.mse_loss(first, s2)
    loss2 = torch.nn.functional.mse_loss(second, s3)
    return 0.5 * (loss1 + loss2), first.detach(), second.detach()


def score(pred, target):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return mae, mse


def evaluate(model, eeg, audio, subjects, channel_mean, device):
    model.eval()
    sums = np.zeros(10)
    n = 0
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
                n += 1
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
    }


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
    print("building audio features", flush=True)
    lengths = melody_lengths()
    audio, _, _ = ensure_features(lengths)
    n_train = sum(
        len(range(0, length - EXAMPLE + 1, args.stride)) * len(train_subjects)
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
        print("loading test EEG", flush=True)
        test_eeg = load_eeg(test_subjects)
        report = evaluate(model, test_eeg, audio, test_subjects, mean, device)
        report.update({
            "model": "newborn_world_model",
            "music_input": True,
            "future_music_visible": True,
            "context_s": 60,
            "pred_s": 30,
            "rollouts": N_FORECASTS,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_windows": n_train,
            "epochs": saved.get("epoch", args.epochs),
            "teacher_epochs": args.teacher_epochs,
        })
        (args.output_dir / "test.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TEST", json.dumps({
            key: report[key]
            for key in (
                "mae", "mse", "rmse", "first_30s_mae", "first_30s_mse",
                "second_30s_mae", "second_30s_mse", "persistence_mae", "train_mean_mae", "trials",
            )
        }, indent=2), flush=True)
        return

    print("loading training EEG", flush=True)
    train_eeg = load_eeg(train_subjects)
    mean = channel_mean(train_eeg, train_subjects)
    dataset = WindowDataset(train_eeg, audio, train_subjects, args.stride)
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
    print("loading test EEG", flush=True)
    test_eeg = load_eeg(test_subjects)
    report = evaluate(model, test_eeg, audio, test_subjects, mean, device)
    report.update({
        "model": "newborn_world_model",
        "music_input": True,
        "future_music_visible": True,
        "context_s": 60,
        "pred_s": 30,
        "rollouts": N_FORECASTS,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_windows": n_train,
        "epochs": args.epochs,
        "teacher_epochs": args.teacher_epochs,
    })
    (args.output_dir / "test.json").write_text(json.dumps(report, indent=2) + "\n")
    print("TEST", json.dumps({
        key: report[key]
        for key in (
            "mae", "mse", "rmse", "first_30s_mae", "first_30s_mse",
            "second_30s_mae", "second_30s_mse", "persistence_mae", "train_mean_mae", "trials",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
