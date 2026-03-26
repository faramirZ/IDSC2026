import numpy as np


def extract_features(signals):
    features = []

    for i, lead in enumerate(signals):
        diff = np.diff(lead)

        # --- time domain ---
        features.extend([
            np.mean(lead),
            np.std(lead),
            np.min(lead),
            np.max(lead),
            np.median(lead),
            np.std(diff),
        ])

        # --- frequency domain ---
        fft = np.fft.rfft(lead)
        fft_mag = np.abs(fft)

        features.extend([
            np.mean(fft_mag),
            np.std(fft_mag),
            np.max(fft_mag),
        ])

        # --- energy ---
        energy = np.sum(lead ** 2)
        features.append(energy)

        # --- focus on V1–V3 ---
        if i in [6, 7, 8]:
            features.extend([
                np.mean(np.abs(lead)),
                np.max(np.abs(lead)),
            ])

    return np.array(features)