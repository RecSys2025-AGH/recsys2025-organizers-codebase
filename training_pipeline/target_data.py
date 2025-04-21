import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetData:
    """
    Dataclass for storing data for training and validation.
    """

    train_df: pd.DataFrame
    validation_df: pd.DataFrame

    @classmethod
    def read_from_dir(cls, target_dir: Path, i: int):
        train_df = pd.read_parquet(target_dir / f"train_target-{i}.parquet")
        validation_df = pd.read_parquet(target_dir / f"validation_target-0.parquet")
        return cls(train_df, validation_df)
