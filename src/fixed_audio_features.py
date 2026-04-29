import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _load_audio(path, target_sample_rate=16000):
    path = str(path)
    try:
        import torchaudio
        import torch

        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != int(target_sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(target_sample_rate))
        return waveform.squeeze(0).cpu().numpy().astype(np.float32), int(target_sample_rate)
    except Exception:
        pass

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if int(sample_rate) != int(target_sample_rate):
            from scipy.signal import resample_poly

            gcd = math.gcd(int(sample_rate), int(target_sample_rate))
            audio = resample_poly(audio, int(target_sample_rate) // gcd, int(sample_rate) // gcd).astype(np.float32)
        return audio, int(target_sample_rate)
    except Exception as exc:
        raise ImportError("AudioCaps feature extraction requires torchaudio or soundfile+scipy.") from exc


def _logmel_stats_torchaudio(audio, sample_rate, n_mels=64, n_fft=1024, hop_length=320):
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        power=2.0,
    )(waveform)
    logmel = torch.log1p(mel.squeeze(0)).cpu().numpy().astype(np.float32)
    return logmel


def _logmel_stats_numpy(audio, sample_rate, n_mels=64, n_fft=1024, hop_length=320):
    from scipy import signal

    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    _, _, stft = signal.stft(
        audio,
        fs=int(sample_rate),
        nperseg=int(n_fft),
        noverlap=max(0, int(n_fft) - int(hop_length)),
        boundary=None,
    )
    power = np.abs(stft).astype(np.float32) ** 2
    if power.shape[0] == 0:
        power = np.zeros((int(n_mels), 1), dtype=np.float32)
    # Lightweight triangular pooling over frequency bins. This is not a calibrated mel filterbank,
    # but it keeps the fallback dependency-light and stable when torchaudio is unavailable.
    splits = np.array_split(power, int(n_mels), axis=0)
    pooled = [chunk.mean(axis=0) if chunk.size else np.zeros(power.shape[1], dtype=np.float32) for chunk in splits]
    return np.log1p(np.stack(pooled, axis=0)).astype(np.float32)


def extract_logmel_stats_features(
    audio_paths,
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=320,
    max_duration_sec=10.0,
):
    features = []
    paths = [str(Path(path)) for path in audio_paths]
    max_samples = int(float(max_duration_sec) * int(sample_rate)) if max_duration_sec else None
    for path in tqdm(paths, desc="Extracting AudioCaps log-mel statistics"):
        audio, sr = _load_audio(path, target_sample_rate=int(sample_rate))
        if max_samples is not None and audio.size > max_samples:
            audio = audio[:max_samples]
        if audio.size == 0:
            audio = np.zeros(int(sample_rate), dtype=np.float32)
        try:
            logmel = _logmel_stats_torchaudio(audio, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        except Exception:
            logmel = _logmel_stats_numpy(audio, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mean = logmel.mean(axis=1)
        std = logmel.std(axis=1)
        maxv = logmel.max(axis=1)
        feat = np.concatenate([mean, std, maxv], axis=0).astype(np.float32)
        features.append(feat)
    if not features:
        raise RuntimeError("No AudioCaps audio features were extracted.")
    feature_matrix = np.stack(features, axis=0).astype(np.float32)
    info = {
        "audio_feature_mode": "logmel_stats",
        "sample_rate": int(sample_rate),
        "n_mels": int(n_mels),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "max_duration_sec": None if max_duration_sec is None else float(max_duration_sec),
        "feature_dim": int(feature_matrix.shape[1]),
    }
    return feature_matrix, info
