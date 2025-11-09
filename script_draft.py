"""
vibe_matcher.py
----------------
Stylize an input audio file to match either a 'Clairo' or 'Laufey' vibe.

Usage:
    python vibe_matcher.py input.wav output.wav clairo
    python vibe_matcher.py input.mp3 output.wav laufey
"""

import sys
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from pydub import AudioSegment, effects

# ===============================
# 1. Load and save audio
# ===============================

def load_audio(path, sr=44100):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr

def save_audio(y, sr, out_path):
    sf.write(out_path, y, sr)
    print(f"[✓] Saved stylized audio to {out_path}")

# ===============================
# 2. EQ filter
# ===============================

def butter_filter(y, cutoff, sr, btype):
    b, a = butter(2, cutoff / (sr / 2), btype=btype)
    return lfilter(b, a, y)

def apply_eq(y, sr, style="clairo"):
    if style == "clairo":
        # Softer highs, warmer mids
        y = butter_filter(y, 8000, sr, 'low')
        y *= 1.1
    elif style == "laufey":
        # Slight high rolloff, boost warmth
        y = butter_filter(y, 100, sr, 'high') * 0.9
        y = butter_filter(y, 6000, sr, 'low') * 1.2
    return np.clip(y, -1.0, 1.0)

# ===============================
# 3. Add gentle reverb (simple delay-based)
# ===============================

def add_simple_reverb(y, sr, decay=0.3, delay=0.05):
    delay_samples = int(sr * delay)
    echo = np.zeros_like(y)
    for i in range(delay_samples, len(y)):
        echo[i] = y[i] + decay * y[i - delay_samples]
    echo = echo / np.max(np.abs(echo) + 1e-6)
    return echo

# ===============================
# 4. Add vibe (lo-fi or jazz ambience)
# ===============================

def add_vibe(y, sr, style="clairo"):
    # Convert NumPy → pydub segment for easy overlay
    seg = AudioSegment(
        (y * 32767).astype(np.int16).tobytes(),
        frame_rate=sr, sample_width=2, channels=1
    )

    seg = effects.normalize(seg)

    if style == "clairo":
        # Try to load vinyl crackle sample if available
        try:
            crackle = AudioSegment.from_file("samples/vinyl_noise.wav").apply_gain(-20)
            seg = seg.overlay(crackle)
            print("[info] Added vinyl crackle overlay.")
        except Exception:
            print("[warn] No vinyl crackle found; skipping.")
        seg = effects.low_pass_filter(seg, 7000)
        seg = seg.fade_in(400).fade_out(600)

    elif style == "laufey":
        seg = effects.low_pass_filter(seg, 6000)
        seg = seg.fade_in(800).fade_out(1200)
        seg = seg.apply_gain(-1)

    # Convert back to numpy
    y_out = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return np.clip(y_out, -1.0, 1.0)

# ===============================
# 5. Main pipeline
# ===============================

def stylize_audio(input_path, output_path, style="clairo"):
    print(f"[*] Loading {input_path} ...")
    y, sr = load_audio(input_path)

    print(f"[*] Applying EQ for {style} vibe ...")
    y = apply_eq(y, sr, style)

    print("[*] Adding reverb ...")
    y = add_simple_reverb(y, sr)

    print("[*] Adding vibe details ...")
    y = add_vibe(y, sr, style)

    print("[*] Exporting final audio ...")
    save_audio(y, sr, output_path)

# ===============================
# 6. CLI entry point
# ===============================

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python vibe_matcher.py <input_file> <output_file> <style>")
        print("Available styles: clairo | laufey")
        sys.exit(1)

    input_path, output_path, style = sys.argv[1], sys.argv[2], sys.argv[3].lower()
    if style not in ["clairo", "laufey"]:
        print("Error: style must be 'clairo' or 'laufey'")
        sys.exit(1)

    stylize_audio(input_path, output_path, style)
    print(f"[✓] Done! '{style}' vibe applied successfully.")
