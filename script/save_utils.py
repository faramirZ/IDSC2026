import json
import joblib
from pathlib import Path
import pandas as pd


def save_artifacts(
    model,
    model_dir: Path,
    X_train,
    X_test,
    train_ids,
    test_ids,
    y_test,
    y_pred,
    y_prob,
    extra_metadata: dict = None,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    joblib.dump(model, model_dir / "model.joblib")

    # 2. Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "input_shape": list(X_train.shape),
        "num_features": X_train.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test) if X_test is not None else None,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # 3. Save split
    split_data = {
        "train_ids": train_ids,
        "test_ids": test_ids
    }

    with open(model_dir / "split.json", "w") as f:
        json.dump(split_data, f, indent=4)

    # 4. Save predictions
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    results_df.to_csv(model_dir / "test_predictions.csv", index=False)