import numpy as np
import pandas as pd


class ECGDataset:
    def __init__(self, metadata: pd.DataFrame, loader):
        """
        metadata: dataframe from metadata.csv
        loader: ECGLoader instance
        """
        self.metadata = metadata
        self.loader = loader

        # filter usable data (binary classification only)
        self.metadata = self.metadata[self.metadata["brugada"].isin([0, 1])]

    def get_patient_ids(self):
        return self.metadata["patient_id"].astype(str).tolist()

    def get_label(self, patient_id: str):
        row = self.metadata[self.metadata["patient_id"] == int(patient_id)]
        return int(row["brugada"].values[0])

    def get_sample(self, patient_id: str):
        signals = self.loader.load_signals(patient_id, transpose=True)  # (12, 1200)
        label = self.get_label(patient_id)
        return signals, label

    def get_data(self):
        """
        Returns:
            X: (N, features)
            y: (N,)
        """
        X = []
        y = []

        for pid in self.get_patient_ids():
            signals, label = self.get_sample(pid)

            # VERY SIMPLE baseline: flatten signal
            features = signals.flatten()  # (12 * 1200 = 14400)

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)