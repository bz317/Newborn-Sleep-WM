"""Shape, causality, and gradient checks. Does not train on the corpus."""
from __future__ import annotations

import torch

from audio_features import features_for_waveform, mel_filterbank
from config import AUDIO_SR, D_MODEL, FEAT_DIM, N_EEG, N_FFT, N_MELS, WIN
from model import WorldModel
from train import reference_stats


def main():
    torch.manual_seed(0)
    model = WorldModel()
    eeg0 = torch.randn(2, WIN, N_EEG)
    eeg1 = torch.randn(2, WIN, N_EEG)
    mean, std = reference_stats(eeg0, eeg1)
    audio = torch.randn(2, 3, WIN, FEAT_DIM)
    pred = model(eeg0, eeg1, audio[:, 0], audio[:, 1], audio[:, 2], mean, std)
    assert pred.shape == (2, WIN, N_EEG), pred.shape

    class Ramp(torch.nn.Module):
        def forward(self, tokens):
            ramp = torch.linspace(-1.0, 1.0, WIN, device=tokens.device)
            return ramp.view(1, 1, -1).expand(tokens.shape[0], tokens.shape[1], WIN)

    model.decoder = Ramp()
    tiny = torch.randn(1, WIN, N_EEG) * 0.01
    ref_mean, ref_std = reference_stats(eeg0[:1], eeg1[:1])
    scaled = model(eeg0[:1], tiny, audio[:1, 0], audio[:1, 1], audio[:1, 2], ref_mean, ref_std)
    expected = float(ref_std.mean() * torch.linspace(-1.0, 1.0, WIN).std(unbiased=False))
    got = float(scaled.std(dim=1).mean())
    if abs(got - expected) / expected > 0.05:
        raise AssertionError(f"fixed scale failed: output std {got} vs reference {expected}")
    model.decoder = WorldModel().decoder

    features = torch.randn(2, WIN, FEAT_DIM)
    tokens = model.encode_audio(features)
    assert tokens.shape == (2, WIN // 10, D_MODEL), tokens.shape
    changed = features.clone()
    changed[:, 1500:] += 3
    tokens_changed = model.encode_audio(changed)
    if not torch.allclose(tokens[:, :150], tokens_changed[:, :150]):
        raise AssertionError("audio tokens depend on future frames")
    if torch.allclose(tokens[:, 150], tokens_changed[:, 150]):
        raise AssertionError("editing the current frame did not change its token")

    target = torch.randn_like(pred)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    if not grads or max(grads) == 0:
        raise AssertionError("backward produced no gradients")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = (
        torch.randn(2, WIN, N_EEG),
        torch.randn(2, WIN, N_EEG),
        torch.randn(2, WIN, FEAT_DIM),
        torch.randn(2, WIN, FEAT_DIM),
        torch.randn(2, WIN, FEAT_DIM),
        torch.randn(2, WIN, N_EEG),
    )
    stats = reference_stats(batch[0], batch[1])
    before = None
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        out = model(*batch[:5], *stats)
        loss = torch.nn.functional.mse_loss(out, batch[5])
        before = float(loss) if before is None else before
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if not torch.isfinite(loss) or float(loss) >= before:
        raise AssertionError(f"loss did not fall: {before} -> {float(loss)}")

    bank = mel_filterbank(AUDIO_SR, N_FFT, N_MELS, 20.0, 8000.0)
    tone = torch.sin(2 * torch.pi * 440 * torch.arange(AUDIO_SR) / AUDIO_SR).numpy()
    feats = features_for_waveform(tone, 100, bank)
    assert feats.shape == (100, FEAT_DIM), feats.shape
    if not (feats[:, :N_MELS].std() > 0 and feats[:, -1].std() >= 0):
        raise AssertionError("features are degenerate")
    from audio_features import _read_wav

    real = _read_wav(
        "/home/bz317/rds/rds-posecraft-5SVx1x5hoTY/bz317/WM_Biomedicine/"
        "data/sleep_data/newborn_music_bianco/STIMULI/test_wav_final_beep/audio03.wav"
    )
    aligned = features_for_waveform(real, 12956, bank)
    assert aligned.shape == (12956, FEAT_DIM), aligned.shape
    print(
        f"smoke ok pred={tuple(pred.shape)} tokens={tuple(tokens.shape)} "
        f"loss {before:.4f} -> {float(loss):.4f} features={aligned.shape}"
    )


if __name__ == "__main__":
    main()
