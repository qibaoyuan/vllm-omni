import argparse
from pathlib import Path

import librosa
import numpy as np
from pystoi import stoi

_DEFAULT_WEIGHTS = {
    "mse": 0.05,
    "snr": 0.1,
    "mel": 0.5,
    "stoi": 0.35,
}


# ---------- Basic metrics ----------
def compute_mse(x, y):
    return np.mean((x - y) ** 2)


def compute_snr(x, y):
    """SNR of reference x relative to the error signal (x - y).
    Returns -80 dB when x is silent to avoid division by zero.
    """
    signal_power = np.sum(x**2)
    noise_power = np.sum((x - y) ** 2)
    if signal_power == 0:
        return -80.0
    return 10 * np.log10(signal_power / (noise_power + 1e-8))


# ---------- Mel spectrogram difference ----------
def compute_mel_loss(x, y, sr):
    x_mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=80)
    y_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)

    x_mel = np.log(x_mel + 1e-8)
    y_mel = np.log(y_mel + 1e-8)

    return np.mean(np.abs(x_mel - y_mel))


# ---------- STOI ----------
def compute_stoi(x, y, sr):
    return stoi(x, y, sr, extended=False)


# ---------- Unified evaluation ----------
def evaluate(ref, test, sr):
    return {
        "mse": compute_mse(ref, test),
        "snr": compute_snr(ref, test),
        "mel": compute_mel_loss(ref, test, sr),
        "stoi": compute_stoi(ref, test, sr),
    }


# ---------- Quality judgement ----------
def judge_quality(metrics):
    if metrics["stoi"] < 0.9:
        return "Unacceptable (speech distortion)"

    if metrics["mel"] > 1.0:
        return "Noticeable spectral distortion"

    if metrics["snr"] < -10:
        return "Possibly misaligned or high noise"

    return "Acceptable (good quality)"


# ---------- Composite scoring ----------
def score(metrics, w):
    return (
        -w["mse"] * metrics["mse"] + w["snr"] * metrics["snr"] - w["mel"] * metrics["mel"] + w["stoi"] * metrics["stoi"]
    )


# ---------- Main workflow ----------
def compare_audio(ref_path, paths, w=None):
    if w is None:
        w = _DEFAULT_WEIGHTS

    ref, sr = librosa.load(ref_path, sr=None)

    results = {}
    for name, path in paths.items():
        y, _ = librosa.load(path, sr=sr)

        # Trim both signals to the shorter length before comparing
        min_len = min(len(ref), len(y))
        metrics = evaluate(ref[:min_len], y[:min_len], sr)
        results[name] = metrics

    scores = {k: score(v, w) for k, v in results.items()}
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return results, scores, ranking


# ---------- Example ----------
"""
=== Metrics ===
sdpa: {'mse': np.float32(0.0098172), 'snr': np.float32(-3.087422), 'mel': np.float32(0.4258594), 'stoi': np.float64(0.9681855813120915)} | quality: Acceptable (good quality)
flash: {'mse': np.float32(0.010014627), 'snr': np.float32(-3.1738937), 'mel': np.float32(0.42801365), 'stoi': np.float64(0.9689482957380713)} | quality: Acceptable (good quality)
eager: {'mse': np.float32(0.010577169), 'snr': np.float32(-3.4112406), 'mel': np.float32(0.42479107), 'stoi': np.float64(0.9690691176525283)} | quality: Acceptable (good quality)
auto: {'mse': np.float32(0.010014627), 'snr': np.float32(-3.1738937), 'mel': np.float32(0.42801365), 'stoi': np.float64(0.9689482957380713)} | quality: Acceptable (good quality)

=== Ranking ===
[('sdpa', np.float64(-0.18329784160760149)), ('flash', np.float64(-0.19276504530606714)), ('auto', np.float64(-0.19276504530606714)), ('eager', np.float64(-0.21487428742497816))]
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare audio files against a reference.")
    parser.add_argument(
        "--ref",
        default="/Users/qibaoyuan/PycharmProjects/vllm-omni-qby/examples/offline_inference/mimo_audio/freetalk_朋友_剪.wav",
        help="Path to reference wav file",
    )
    parser.add_argument(
        "--base-dir",
        default="/Users/qibaoyuan/PycharmProjects/vllm-omni-qby",
        help="Base directory for reconstructed wav files",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["sdpa", "flash", "eager", "auto"],
        help="Names of audio variants (expects reconstructed_<name>.wav in base-dir)",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    audio_paths = {name: str(base / f"reconstructed_{name}.wav") for name in args.names}

    all_results, all_scores, all_ranking = compare_audio(args.ref, audio_paths)

    print("=== Metrics ===")
    for name, metrics in all_results.items():
        print(f"{name}: {metrics} | quality: {judge_quality(metrics)}")

    print("\n=== Ranking ===")
    print(all_ranking)
