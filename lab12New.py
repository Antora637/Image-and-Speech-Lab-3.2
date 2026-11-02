
"""
Q12: Separation of Voiced, Unvoiced and Silence Regions
Method: Short-Time Energy (STE) + Zero Crossing Rate (ZCR)

Install:
pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Read audio
fs, x = wavfile.read('audio.wav')

if x.ndim > 1:
    x = x.mean(axis=1)

x = x.astype(float)
x = x / np.max(np.abs(x))
t = np.arange(len(x)) / fs

# 2. Framing
frame_len = int(0.025 * fs)
hop = int(0.010 * fs)

energy = []
zcr = []

for i in range(0, len(x) - frame_len + 1, hop):
    frame = x[i:i + frame_len]

    energy.append(np.sum(frame ** 2))
    zcr.append(0.5 * np.mean(np.abs(np.diff(np.sign(frame)))))

energy = np.array(energy)
zcr = np.array(zcr)

# 3. Threshold
E_thr = 0.02 * np.max(energy)
Z_thr = 0.15

# 4. Classification
label = []

for i in range(len(energy)):
    if energy[i] < E_thr:
        label.append(0)       # Silence
    elif zcr[i] > Z_thr:
        label.append(1)       # Unvoiced
    else:
        label.append(2)       # Voiced

# 5. Plot
plt.figure(figsize=(11, 8))

# Original signal
plt.subplot(4, 1, 1)
plt.plot(t, x, 'k')
plt.title('Original Speech Signal')
plt.ylabel('Amplitude')

# Short-time energy
plt.subplot(4, 1, 2)
plt.plot(energy)
plt.axhline(E_thr, color='r', linestyle='--',
            label='Energy Threshold')
plt.title('Short-Time Energy')
plt.legend()

# Zero crossing rate
plt.subplot(4, 1, 3)
plt.plot(zcr)
plt.axhline(Z_thr, color='r', linestyle='--',
            label='ZCR Threshold')
plt.title('Zero Crossing Rate')
plt.legend()

# Voiced, Unvoiced and Silence
plt.subplot(4, 1, 4)

# Legend: show all three labels
plt.plot([], [], color='gray', label='Silence')
plt.plot([], [], color='red', label='Unvoiced')
plt.plot([], [], color='blue', label='Voiced')

for i in range(len(label)):
    start = i * hop
    end = min(start + frame_len, len(x))

    if label[i] == 0:
        plt.plot(t[start:end], x[start:end], color='gray')

    elif label[i] == 1:
        plt.plot(t[start:end], x[start:end], color='red')

    else:
        plt.plot(t[start:end], x[start:end], color='blue')

plt.title('Voiced / Unvoiced / Silence Regions')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
