# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Validate the row/frame contract before regenerating dataset video clocks."""

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from lerobot.datasets.utils import DEFAULT_DATA_PATH


class VideoFrameRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_index: int
    file_index: int
    start_frame: int
    frame_count: int


class VideoAggregationState(BaseModel):
    chunk_index: int = 0
    file_index: int = 0
    frame_counts: dict[tuple[int, int], int] = Field(default_factory=dict)
    episodes: dict[int, VideoFrameRange] = Field(default_factory=dict)


def validate_dataset_frame_times(root: Path, fps: int, episodes: pd.DataFrame) -> None:
    """Require one row per displayed frame, in episode order, on the declared FPS clock.

    The tolerance covers only float32 timestamp serialization, not sampling jitter
    or missing frames. Custom timestamps must not be silently retimed.
    """
    if fps <= 0 or episodes.empty or episodes["episode_index"].duplicated().any():
        raise ValueError(f"{root}: expected positive FPS and unique nonempty episodes")
    for (chunk, file), metadata in episodes.groupby(["data/chunk_index", "data/file_index"]):
        path = root / DEFAULT_DATA_PATH.format(chunk_index=int(chunk), file_index=int(file))
        rows = pd.read_parquet(path, columns=["episode_index", "frame_index", "timestamp"])
        if set(rows["episode_index"]) != set(metadata["episode_index"]):
            raise ValueError(f"{path}: data episodes do not match metadata")
        for episode_id, episode in rows.groupby("episode_index", sort=False):
            length = int(metadata.loc[metadata["episode_index"] == episode_id, "length"].iloc[0])
            indices = episode["frame_index"].to_numpy()
            stamps = episode["timestamp"].to_numpy(dtype=np.float64)
            expected = np.arange(length, dtype=np.float64) / fps
            error_bound = 2 * np.finfo(np.float32).eps * np.maximum(1, np.abs(expected))
            if (
                length <= 0
                or len(episode) != length
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(indices, np.arange(length))
                or not np.all(np.abs(stamps - expected) <= error_bound)
            ):
                raise ValueError(f"{path}: episode {episode_id} rows must follow frame_index / FPS exactly")
