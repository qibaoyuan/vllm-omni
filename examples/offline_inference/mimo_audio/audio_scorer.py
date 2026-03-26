import librosa
import numpy as np
from pystoi import stoi


# ---------- Basic metrics ----------
def compute_mse(x, y):
    return np.mean((x - y) ** 2)


def compute_snr(x, y):
    noise = x - y
    return 10 * np.log10(np.sum(x**2) / (np.sum(noise**2) + 1e-8))


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


# ---------- Composite scoring ----------
def score(metrics, w):
    # quality = judge_quality(metrics)
    # print(quality)
    return (
        -w["mse"] * metrics["mse"]
        + w["snr"] * metrics["snr"]
        + -w["mel"] * metrics["mel"]
        + w["stoi"] * metrics["stoi"]
    )


# ---------- Main workflow ----------
def compare_audio(ref_path, paths):
    ref, sr = librosa.load(ref_path, sr=None)

    results = {}
    for name, path in paths.items():
        y, _ = librosa.load(path, sr=sr)

        # Align lengths
        min_len = min(len(ref), len(y))
        ref_cut = ref[:min_len]
        y_cut = y[:min_len]

        metrics = evaluate(ref_cut, y_cut, sr)
        results[name] = metrics

    # Weights
    w = {
        "mse": 0.05,
        "snr": 0.1,
        "mel": 0.5,
        "stoi": 0.35,
    }

    # Compute scores
    scores = {k: score(v, w) for k, v in results.items()}

    # Ranking
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return results, scores, ranking


def judge_quality(metrics):
    if metrics["stoi"] < 0.9:
        return "❌ Unacceptable (speech distortion)"

    if metrics["mel"] > 1.0:
        return "⚠️ Noticeable spectral distortion"

    if metrics["snr"] < 5:
        return "⚠️ Possibly misaligned or high noise"

    return "✅ Acceptable (good quality)"


# ---------- Example ----------
"""
=== Metrics ===
sdpa {'mse': np.float32(0.0098172), 'snr': np.float32(-3.087422), 'mel': np.float32(0.4258594), 'stoi': np.float64(0.9681855813120915)}
flash {'mse': np.float32(0.010014627), 'snr': np.float32(-3.1738937), 'mel': np.float32(0.42801365), 'stoi': np.float64(0.9689482957380713)}
eager {'mse': np.float32(0.010577169), 'snr': np.float32(-3.4112406), 'mel': np.float32(0.42479107), 'stoi': np.float64(0.9690691176525283)}
auto {'mse': np.float32(0.010014627), 'snr': np.float32(-3.1738937), 'mel': np.float32(0.42801365), 'stoi': np.float64(0.9689482957380713)}

=== Ranking ===
[('sdpa', np.float64(-0.18329784160760149)), ('flash', np.float64(-0.19276504530606714)), ('auto', np.float64(-0.19276504530606714)), ('eager', np.float64(-0.21487428742497816))]
"""
if __name__ == "__main__":
    ref_path = (
        "/Users/qibaoyuan/PycharmProjects/vllm-omni-qby/examples/offline_inference/mimo_audio/freetalk_朋友_剪.wav"
    )
    c_b_p = "/Users/qibaoyuan/PycharmProjects/vllm-omni-qby/"
    paths = {
        "sdpa": c_b_p + "reconstructed_sdpa.wav",
        "flash": c_b_p + "reconstructed_flash.wav",
        "eager": c_b_p + "reconstructed_eager.wav",
        "auto": c_b_p + "reconstructed_auto.wav",
    }

    results, scores, ranking = compare_audio(ref_path, paths)

    print("=== Metrics ===")
    for k, v in results.items():
        print(k, v)

    print("\n=== Ranking ===")
    print(ranking)
