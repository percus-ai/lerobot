#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import numpy as np
import pandas as pd
import pytest
import torch

import lerobot.datasets.streaming_dataset as streaming_dataset_module
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.utils import safe_shard
from lerobot.utils.constants import ACTION
from tests.fixtures.constants import DUMMY_REPO_ID


@pytest.mark.parametrize(
    (
        "metadata_tolerance_s",
        "explicit_tolerance_s",
        "fps",
        "expected_delta_tolerance_s",
        "expected_video_tolerance_s",
    ),
    [
        (None, None, 30, 1e-4, 0.1),
        (None, None, 60, 1e-4, 0.05),
        (0.05, None, 30, 1e-4, 0.05),
        (0.1, 0.02, 30, 0.02, 0.02),
    ],
)
def test_streaming_dataset_resolves_video_timestamp_tolerance(
    tmp_path,
    lerobot_dataset_factory,
    info_factory,
    metadata_tolerance_s,
    explicit_tolerance_s,
    fps,
    expected_delta_tolerance_s,
    expected_video_tolerance_s,
):
    root = tmp_path / "test"
    info = info_factory(total_episodes=1, total_frames=1, total_tasks=1, fps=fps)
    if metadata_tolerance_s is not None:
        info["video_timestamp_tolerance_s"] = metadata_tolerance_s
    lerobot_dataset_factory(root=root, info=info)
    kwargs = {} if explicit_tolerance_s is None else {"tolerance_s": explicit_tolerance_s}

    dataset = StreamingLeRobotDataset(repo_id=DUMMY_REPO_ID, root=root, **kwargs)

    assert dataset.tolerance_s == expected_delta_tolerance_s
    assert dataset.video_timestamp_tolerance_s == expected_video_tolerance_s


@pytest.mark.parametrize(
    "invalid_tolerance_s",
    [None, True, "0.1", 0, -0.1, float("nan"), float("inf")],
)
def test_streaming_dataset_rejects_invalid_metadata_video_timestamp_tolerance(
    tmp_path,
    lerobot_dataset_factory,
    invalid_tolerance_s,
):
    root = tmp_path / "test"
    lerobot_dataset_factory(root=root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_timestamp_tolerance_s"] = invalid_tolerance_s
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="finite positive number"):
        StreamingLeRobotDataset(repo_id=DUMMY_REPO_ID, root=root)


@pytest.mark.parametrize(
    "invalid_tolerance_s",
    [True, "0.1", 0, -0.1, float("nan"), float("inf")],
)
def test_streaming_dataset_rejects_invalid_explicit_video_timestamp_tolerance(
    tmp_path,
    lerobot_dataset_factory,
    invalid_tolerance_s,
):
    root = tmp_path / "test"
    lerobot_dataset_factory(root=root)

    with pytest.raises((TypeError, ValueError), match="finite positive number"):
        StreamingLeRobotDataset(
            repo_id=DUMMY_REPO_ID,
            root=root,
            tolerance_s=invalid_tolerance_s,
        )


def test_streaming_video_query_adds_episode_video_offset(monkeypatch, tmp_path):
    dataset = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
    dataset.root = tmp_path
    dataset.streaming = False
    dataset.streaming_from_local = True
    dataset.video_timestamp_tolerance_s = 0.1
    dataset.video_decoder_cache = object()
    dataset.meta = type(
        "Meta",
        (),
        {
            "episodes": [{"videos/phone/from_timestamp": 12.5}],
            "get_video_file_path": staticmethod(lambda _ep_idx, _video_key: "video.mp4"),
        },
    )()
    captured = {}

    def fake_decode(video_path, timestamps, tolerance_s, decoder_cache):
        captured.update(
            video_path=video_path,
            timestamps=timestamps,
            tolerance_s=tolerance_s,
            decoder_cache=decoder_cache,
        )
        return torch.zeros((2, 3, 4, 4))

    monkeypatch.setattr(streaming_dataset_module, "decode_video_frames_torchcodec", fake_decode)

    frames = dataset._query_videos({"phone": [0.0, 1 / 30]}, ep_idx=0)

    assert frames["phone"].shape == (2, 3, 4, 4)
    assert captured == {
        "video_path": f"{tmp_path}/video.mp4",
        "timestamps": pytest.approx([12.5, 12.5 + 1 / 30]),
        "tolerance_s": 0.1,
        "decoder_cache": dataset.video_decoder_cache,
    }


def test_streaming_make_frame_queries_episode_relative_row_timestamp():
    dataset = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
    dataset.delta_indices = None
    dataset.delta_timestamps = None
    dataset.image_transforms = None
    dataset.meta = type(
        "Meta",
        (),
        {
            "fps": 30,
            "video_keys": ["phone"],
            "episodes": [
                {
                    "videos/phone/from_timestamp": 12.5,
                    "videos/phone/to_timestamp": 13.5,
                }
            ],
            "tasks": pd.DataFrame({"task_index": [0]}, index=["pick"]),
        },
    )()
    captured = {}

    def fake_query(query_timestamps, ep_idx):
        captured.update(query_timestamps=query_timestamps, ep_idx=ep_idx)
        return {"phone": torch.zeros((3, 4, 4))}

    dataset._query_videos = fake_query
    item = {
        "episode_index": 0,
        "timestamp": 0.25,
        "index": 100,
        "task_index": 0,
    }

    frame = next(dataset.make_frame(iter([item])))

    assert captured == {"query_timestamps": {"phone": [0.25]}, "ep_idx": 0}
    assert frame["task"] == "pick"


def get_frames_expected_order(streaming_ds: StreamingLeRobotDataset) -> list[int]:
    """Replicates the shuffling logic of StreamingLeRobotDataset to get the expected order of indices."""
    rng = np.random.default_rng(streaming_ds.seed)
    buffer_size = streaming_ds.buffer_size
    num_shards = streaming_ds.num_shards

    shards_indices = []
    for shard_idx in range(num_shards):
        shard = streaming_ds.hf_dataset.shard(num_shards, index=shard_idx)
        shard_indices = [item["index"] for item in shard]
        shards_indices.append(shard_indices)

    shard_iterators = {i: iter(s) for i, s in enumerate(shards_indices)}

    buffer_indices_generator = streaming_ds._iter_random_indices(rng, buffer_size)

    frames_buffer = []
    expected_indices = []

    while shard_iterators:  # While there are still available shards
        available_shard_keys = list(shard_iterators.keys())
        if not available_shard_keys:
            break

        # Call _infinite_generator_over_elements with current available shards (key difference!)
        shard_key = next(streaming_ds._infinite_generator_over_elements(rng, available_shard_keys))

        try:
            frame_index = next(shard_iterators[shard_key])

            if len(frames_buffer) == buffer_size:
                i = next(buffer_indices_generator)
                expected_indices.append(frames_buffer[i])
                frames_buffer[i] = frame_index
            else:
                frames_buffer.append(frame_index)

        except StopIteration:
            del shard_iterators[shard_key]  # Remove exhausted shard

    rng.shuffle(frames_buffer)
    expected_indices.extend(frames_buffer)

    return expected_indices


def test_single_frame_consistency(tmp_path, lerobot_dataset_factory):
    """Test if are correctly accessed"""
    ds_num_frames = 400
    ds_num_episodes = 10
    buffer_size = 100

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}"

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
    )

    streaming_ds = iter(StreamingLeRobotDataset(repo_id=repo_id, root=local_path, buffer_size=buffer_size))

    key_checks = []
    for _ in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = streaming_frame["index"]
        target_frame = ds[frame_idx]

        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                check = torch.allclose(left, right) and left.shape == right.shape

            elif isinstance(left, float):
                check = left == right.item()  # right is a torch.Tensor

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (frame_idx: {frame_idx})"
        )


@pytest.mark.parametrize(
    "shuffle",
    [False, True],
)
def test_frames_order_over_epochs(tmp_path, lerobot_dataset_factory, shuffle):
    """Test if streamed frames correspond to shuffling operations over in-memory dataset."""
    ds_num_frames = 400
    ds_num_episodes = 10
    buffer_size = 100
    seed = 42
    n_epochs = 3

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}"

    lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
    )

    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=local_path, buffer_size=buffer_size, seed=seed, shuffle=shuffle
    )

    first_epoch_indices = [frame["index"] for frame in streaming_ds]
    expected_indices = get_frames_expected_order(streaming_ds)

    assert first_epoch_indices == expected_indices, "First epoch indices do not match expected indices"

    expected_indices = get_frames_expected_order(streaming_ds)
    for _ in range(n_epochs):
        streaming_indices = [frame["index"] for frame in streaming_ds]
        frames_match = all(
            s_index == e_index for s_index, e_index in zip(streaming_indices, expected_indices, strict=True)
        )

        if shuffle:
            assert not frames_match
        else:
            assert frames_match


@pytest.mark.parametrize(
    "shuffle",
    [False, True],
)
def test_frames_order_with_shards(tmp_path, lerobot_dataset_factory, shuffle):
    """Test if streamed frames correspond to shuffling operations over in-memory dataset with multiple shards."""
    ds_num_frames = 100
    ds_num_episodes = 10
    buffer_size = 10

    seed = 42
    n_epochs = 3
    data_file_size_mb = 0.001

    chunks_size = 1

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"

    lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        data_files_size_in_mb=data_file_size_mb,
        chunks_size=chunks_size,
    )

    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id,
        root=local_path,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=shuffle,
        max_num_shards=4,
    )

    first_epoch_indices = [frame["index"] for frame in streaming_ds]
    expected_indices = get_frames_expected_order(streaming_ds)

    assert first_epoch_indices == expected_indices, "First epoch indices do not match expected indices"

    for _ in range(n_epochs):
        streaming_indices = [
            frame["index"] for frame in streaming_ds
        ]  # NOTE: this is the same as first_epoch_indices
        frames_match = all(
            s_index == e_index for s_index, e_index in zip(streaming_indices, expected_indices, strict=True)
        )
        if shuffle:
            assert not frames_match
        else:
            assert frames_match


@pytest.mark.parametrize(
    "state_deltas, action_deltas",
    [
        ([-1, -0.5, -0.20, 0], [0, 1, 2, 3]),
        ([-1, -0.5, -0.20, 0], [-1.5, -1, -0.5, -0.20, -0.10, 0]),
        ([-2, -1, -0.5, 0], [0, 1, 2, 3]),
        ([-2, -1, -0.5, 0], [-1.5, -1, -0.5, -0.20, -0.10, 0]),
    ],
)
def test_frames_with_delta_consistency(tmp_path, lerobot_dataset_factory, state_deltas, action_deltas):
    ds_num_frames = 500
    ds_num_episodes = 10
    buffer_size = 100

    seed = 42

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"
    camera_key = "phone"

    delta_timestamps = {
        camera_key: state_deltas,
        "state": state_deltas,
        ACTION: action_deltas,
    }

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        delta_timestamps=delta_timestamps,
    )

    streaming_ds = iter(
        StreamingLeRobotDataset(
            repo_id=repo_id,
            root=local_path,
            buffer_size=buffer_size,
            seed=seed,
            shuffle=False,
            delta_timestamps=delta_timestamps,
        )
    )

    for i in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = streaming_frame["index"]
        target_frame = ds[frame_idx]

        assert set(streaming_frame.keys()) == set(target_frame.keys()), (
            f"Keys differ between streaming frame and target one. Differ at: {set(streaming_frame.keys()) - set(target_frame.keys())}"
        )

        key_checks = []
        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                if (
                    key not in ds.meta.camera_keys
                    and "is_pad" not in key
                    and f"{key}_is_pad" in streaming_frame
                ):
                    # comparing frames only on non-padded regions. Padding is applied to last-valid broadcasting
                    left = left[~streaming_frame[f"{key}_is_pad"]]
                    right = right[~target_frame[f"{key}_is_pad"]]

                check = torch.allclose(left, right) and left.shape == right.shape

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (i: {i}, frame_idx: {frame_idx})"
        )


@pytest.mark.parametrize(
    "state_deltas, action_deltas",
    [
        ([-1, -0.5, -0.20, 0], [0, 1, 2, 3, 10, 20]),
        ([-1, -0.5, -0.20, 0], [-20, -1.5, -1, -0.5, -0.20, -0.10, 0]),
        ([-2, -1, -0.5, 0], [0, 1, 2, 3, 10, 20]),
        ([-2, -1, -0.5, 0], [-20, -1.5, -1, -0.5, -0.20, -0.10, 0]),
    ],
)
def test_frames_with_delta_consistency_with_shards(
    tmp_path, lerobot_dataset_factory, state_deltas, action_deltas
):
    ds_num_frames = 100
    ds_num_episodes = 10
    buffer_size = 10
    data_file_size_mb = 0.001
    chunks_size = 1

    seed = 42

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"
    camera_key = "phone"

    delta_timestamps = {
        camera_key: state_deltas,
        "state": state_deltas,
        ACTION: action_deltas,
    }

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        delta_timestamps=delta_timestamps,
        data_files_size_in_mb=data_file_size_mb,
        chunks_size=chunks_size,
    )
    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id,
        root=local_path,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=False,
        delta_timestamps=delta_timestamps,
        max_num_shards=4,
    )

    iter(streaming_ds)

    num_shards = 4
    shards_indices = []
    for shard_idx in range(num_shards):
        shard = safe_shard(streaming_ds.hf_dataset, shard_idx, num_shards)
        shard_indices = [item["index"] for item in shard]
        shards_indices.append(shard_indices)

    streaming_ds = iter(streaming_ds)

    for i in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = streaming_frame["index"]
        target_frame = ds[frame_idx]

        assert set(streaming_frame.keys()) == set(target_frame.keys()), (
            f"Keys differ between streaming frame and target one. Differ at: {set(streaming_frame.keys()) - set(target_frame.keys())}"
        )

        key_checks = []
        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                if (
                    key not in ds.meta.camera_keys
                    and "is_pad" not in key
                    and f"{key}_is_pad" in streaming_frame
                ):
                    # comparing frames only on non-padded regions. Padding is applied to last-valid broadcasting
                    left = left[~streaming_frame[f"{key}_is_pad"]]
                    right = right[~target_frame[f"{key}_is_pad"]]

                check = torch.allclose(left, right) and left.shape == right.shape

            elif isinstance(left, float):
                check = left == right.item()  # right is a torch.Tensor

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (i: {i}, frame_idx: {frame_idx})"
        )
