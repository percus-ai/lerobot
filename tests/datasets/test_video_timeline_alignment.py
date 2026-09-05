# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Image/action identity, not just successful muxing, is the retiming contract."""

import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_timeline import remux_video_files, scan_video_timeline
from tests.datasets.test_video_concatenation import h264_image_payloads, packet_timestamps

CAMERAS = ("observation.images.top", "observation.images.arm")


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("alignment-sources")
    paths = []
    with pytest.MonkeyPatch.context() as env:
        env.setenv("LEROBOT_VIDEO_VCODEC", "h264")
        for source_index, lengths in enumerate(((7, 11), (13,), (9,))):
            env.setenv("LEROBOT_VIDEO_GOP", "1" if source_index == 0 else "30")
            dataset = LeRobotDataset.create(
                repo_id=f"test/source-{source_index}",
                root=root / str(source_index),
                fps=30,
                features={
                    **{
                        key: {
                            "dtype": "video",
                            "shape": (64, 96, 3),
                            "names": ["height", "width", "channels"],
                        }
                        for key in CAMERAS
                    },
                    "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                },
            )
            for episode_index, length in enumerate(lengths):
                for frame_index in range(length):
                    identity = source_index * 60 + episode_index * 20 + frame_index
                    dataset.add_frame(
                        {
                            **{
                                key: np.full((64, 96, 3), identity + camera_index * 10, np.uint8)
                                for camera_index, key in enumerate(CAMERAS)
                            },
                            "action": np.array([identity], np.float32),
                            "observation.state": np.array([-identity], np.float32),
                            "task": "alignment",
                        }
                    )
                dataset.save_episode()
            dataset.finalize()
            paths.append(dataset.root)
    return paths


def merge(paths: list[Path], output: Path, max_video_mb: float = 200) -> None:
    aggregate_datasets(
        repo_ids=[f"test/{path.name}" for path in paths],
        roots=paths,
        aggr_repo_id="test/output",
        aggr_root=output,
        video_files_size_in_mb=max_video_mb,
        chunk_size=1,
    )


def inject_old_boundary(root: Path, defect: str) -> None:
    """Reproduce old muxable timestamps while leaving frame bytes and action rows intact."""
    for path in sorted((root / "videos").rglob("*.mp4")):
        temporary = path.with_suffix(".retimed.mp4")
        with (
            av.open(str(path)) as source,
            av.open(str(temporary), "w", options={"avoid_negative_ts": "disabled"}) as output,
        ):
            stream = output.add_stream_from_template(source.streams.video[0], opaque=True)
            count = 0
            for packet in source.demux(video=0):
                if not packet.size:
                    continue
                if count >= 7:
                    if defect == "dts-only":
                        packet.dts -= packet.duration - 1
                    else:
                        shift = 100 if defect == "pts-off-grid" else 1536
                        packet.pts += shift
                        packet.dts += shift
                packet.stream = stream
                output.mux(packet)
                count += 1
        temporary.replace(path)
    if defect != "dts-only":
        shift_s = (100 if defect == "pts-off-grid" else 1536) / 15360
        for path in (root / "meta" / "episodes").rglob("*.parquet"):
            episodes = pd.read_parquet(path)
            for key in CAMERAS:
                for endpoint in ("from_timestamp", "to_timestamp"):
                    episodes.loc[episodes["episode_index"] > 0, f"videos/{key}/{endpoint}"] += shift_s
            episodes.to_parquet(path)


def assert_rows_equal(expected_paths: list[Path], actual_path: Path, backend: str) -> None:
    actual = LeRobotDataset("test/output", root=actual_path, video_backend=backend)
    offset = 0
    for path in expected_paths:
        expected = LeRobotDataset(f"test/{path.name}", root=path, video_backend=backend)
        for index in range(len(expected)):
            before, after = expected[index], actual[offset + index]
            for key in (*CAMERAS, "action", "observation.state", "timestamp", "frame_index"):
                assert torch.equal(before[key], after[key]), (backend, offset + index, key)
        offset += len(expected)
    assert len(actual) == offset


@pytest.mark.parametrize("defect", ["dts-only", "pts-off-grid", "pts-gap"])
@pytest.mark.parametrize("rotate", [False, True])
def test_reaggregate_old_boundaries_preserves_every_action_image_pair(
    tmp_path: Path, sources: list[Path], defect: str, rotate: bool
):
    legacy = tmp_path / "legacy"
    merge(sources[:2], legacy)
    inject_old_boundary(legacy, defect)
    originals = {path: path.read_bytes() for path in legacy.rglob("*") if path.is_file()}
    output = tmp_path / "output"
    # Test both a legacy first file and a legacy file reached through file/chunk rotation.
    inputs = [sources[2], legacy] if rotate else [legacy, sources[2]]
    expected = [sources[2], *sources[:2]] if rotate else sources
    merge(inputs, output, max_video_mb=0.001 if rotate else 200)
    for backend in ("pyav", "torchcodec"):
        assert_rows_equal(expected, output, backend)
    for key in CAMERAS:
        payloads = [
            payload
            for path in expected
            for video in sorted((path / "videos" / key).rglob("*.mp4"))
            for payload in h264_image_payloads(video)
        ]
        actual_payloads = [
            payload
            for video in sorted((output / "videos" / key).rglob("*.mp4"))
            for payload in h264_image_payloads(video)
        ]
        assert payloads == actual_payloads
        for path in (output / "videos" / key).rglob("*.mp4"):
            timeline = scan_video_timeline(path)
            stamps = packet_timestamps(path)
            assert sorted(pts for pts, _, _ in stamps) == [
                i * timeline.frame_duration for i in range(len(stamps))
            ]
            assert all(a[1] < b[1] for a, b in zip(stamps, stamps[1:], strict=False))
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("defect", ["timestamp", "frame_index", "length", "video_count"])
def test_invalid_row_frame_contract_leaves_no_output(tmp_path: Path, sources: list[Path], defect: str):
    source = tmp_path / "source"
    shutil.copytree(sources[0], source)
    if defect == "video_count":
        target = next((source / "videos" / CAMERAS[0]).rglob("*.mp4"))
        shutil.copyfile(next((sources[2] / "videos" / CAMERAS[0]).rglob("*.mp4")), target)
    else:
        folder = source / "meta" / "episodes" if defect == "length" else source / "data"
        path = next(folder.rglob("*.parquet"))
        rows = pd.read_parquet(path)
        rows.at[0, defect] += 1
        rows.to_parquet(path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="rows"):
        merge([source], output)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source"]


def test_dataloader_temporal_queries_and_workers(tmp_path: Path, sources: list[Path]):
    legacy, output = tmp_path / "legacy", tmp_path / "output"
    merge(sources[:2], legacy)
    inject_old_boundary(legacy, "pts-gap")
    merge([legacy, sources[2]], output)
    for backend in ("pyav", "torchcodec"):
        dataset = LeRobotDataset(
            "test/output",
            root=output,
            video_backend=backend,
            delta_timestamps={**{key: [-1 / 30, 0, 1 / 30] for key in CAMERAS}, "action": [0, 1 / 30]},
        )
        reference = [dataset[index] for index in range(len(dataset))]
        loader = torch.utils.data.DataLoader(dataset, batch_size=7, shuffle=True, num_workers=2)
        visited = []
        for batch in loader:
            for slot, index in enumerate(batch["index"].tolist()):
                for key in (*CAMERAS, "action", "observation.state", "timestamp"):
                    assert torch.equal(batch[key][slot], reference[index][key])
                visited.append(index)
        assert sorted(visited) == list(range(len(dataset)))


def test_resume_recording_then_reaggregate_twice(
    tmp_path: Path, sources: list[Path], monkeypatch: pytest.MonkeyPatch
):
    legacy, repaired = tmp_path / "legacy", tmp_path / "repaired"
    merge(sources[:2], legacy)
    inject_old_boundary(legacy, "pts-gap")
    merge([legacy, sources[2]], repaired)
    monkeypatch.setenv("LEROBOT_VIDEO_VCODEC", "h264")
    monkeypatch.setenv("LEROBOT_VIDEO_GOP", "30")
    resumed = LeRobotDataset("test/output", root=repaired)
    for index in range(5):
        resumed.add_frame(
            {
                **{key: np.full((64, 96, 3), 180 + index, np.uint8) for key in CAMERAS},
                "action": np.array([180 + index], np.float32),
                "observation.state": np.array([-180 - index], np.float32),
                "task": "alignment",
            }
        )
    resumed.save_episode()
    resumed.finalize()
    again, final = tmp_path / "again", tmp_path / "final"
    merge([repaired], again)
    merge([again], final)
    for backend in ("pyav", "torchcodec"):
        assert_rows_equal([repaired], final, backend)


def test_video_only_correction_breaks_action_image_correspondence(tmp_path: Path, sources: list[Path]):
    legacy = tmp_path / "legacy"
    merge(sources[:2], legacy)
    inject_old_boundary(legacy, "pts-gap")
    for path in (legacy / "videos").rglob("*.mp4"):
        remux_video_files([scan_video_timeline(path)], path)
    # Negative control: a successful video rewrite is insufficient if metadata is stale.
    for backend in ("pyav", "torchcodec"):
        with pytest.raises((AssertionError, RuntimeError)):
            assert_rows_equal(sources[:2], legacy, backend)
