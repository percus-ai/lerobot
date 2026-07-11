#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import pytest

from lerobot.scripts import lerobot_dataset_viz


class _VisualizationDatasetSentinel:
    pass


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = lerobot_dataset_viz.visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()


def test_visualization_loader_uses_dataset_metadata_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = _VisualizationDatasetSentinel()

    def fake_dataset(
        repo_id: str,
        *,
        episodes: list[int],
        root: Path | None,
    ) -> _VisualizationDatasetSentinel:
        assert repo_id == "local/dataset"
        assert episodes == [3]
        assert root == tmp_path
        return sentinel

    monkeypatch.setattr(lerobot_dataset_viz, "LeRobotDataset", fake_dataset)

    dataset = lerobot_dataset_viz.load_dataset_for_visualization(
        "local/dataset",
        episode_index=3,
        root=tmp_path,
        tolerance_s=None,
    )

    assert dataset is sentinel


def test_visualization_loader_accepts_explicit_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = _VisualizationDatasetSentinel()

    def fake_dataset(
        repo_id: str,
        *,
        episodes: list[int],
        root: Path | None,
        tolerance_s: float,
    ) -> _VisualizationDatasetSentinel:
        assert repo_id == "local/dataset"
        assert episodes == [4]
        assert root == tmp_path
        assert tolerance_s == pytest.approx(0.08)
        return sentinel

    monkeypatch.setattr(lerobot_dataset_viz, "LeRobotDataset", fake_dataset)

    dataset = lerobot_dataset_viz.load_dataset_for_visualization(
        "local/dataset",
        episode_index=4,
        root=tmp_path,
        tolerance_s=0.08,
    )

    assert dataset is sentinel
