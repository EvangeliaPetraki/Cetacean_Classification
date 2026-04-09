import os
import soundfile as sf
import numpy as np

# Path where you keep your audio files
BASE_DIR = r"E:/Cetaceos de Canarias Base/_common-frecuent"

sampling_rates = []

# Walk through all WAV files in your dataset
for root, _, files in os.walk(BASE_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            filepath = os.path.join(root, f)
            try:
                info = sf.info(filepath)
                sr = info.samplerate
                sampling_rates.append(sr)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Summary
unique_rates, counts = np.unique(sampling_rates, return_counts=True)
print("Sampling rates found (Hz):")
for rate, count in zip(unique_rates, counts):
    print(f"{rate} Hz → {count} files")

# Calculate Nyquist limits
print("\nNyquist limits (max detectable frequency):")
for rate in unique_rates:
    print(f"{rate/2/1000:.1f} kHz (for {rate} Hz files)")
