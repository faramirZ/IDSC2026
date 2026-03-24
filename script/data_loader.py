import wfdb
import numpy as np
from pathlib import Path


class ECGLoader:
    def __init__(self, brugada_data_path: Path):
        """
        Parameters
        ----------
        brugada_data_path : Path
            Path to the root dataset folder (the one containing metadata.csv and files/)
        """
        self.data_dir = Path(brugada_data_path)
        self.files_dir = self.data_dir / "files"

    def load_record(self, patient_id: str):
        """
        Load full WFDB record (signals + metadata)
        """
        record_path = self.files_dir / patient_id / patient_id
        record = wfdb.rdrecord(str(record_path))  # wfdb prefers string

        signals = record.p_signal.astype(np.float32)  # (1200, 12)
        leads = record.sig_name
        fs = record.fs

        return {
            "patient_id": patient_id,
            "signals": signals,
            "leads": leads,
            "fs": fs
        }

    def load_signals(self, patient_id: str, transpose: bool = True):
        """
        Load only signal array

        Returns:
            (12, 1200) if transpose=True
            (1200, 12) if transpose=False
        """
        data = self.load_record(patient_id)
        signals = data["signals"]

        if transpose:
            signals = signals.T  # (12, 1200)

        return signals

    def get_available_patients(self):
        """
        List all patient IDs from files directory
        """
        return [p.name for p in self.files_dir.iterdir() if p.is_dir()]