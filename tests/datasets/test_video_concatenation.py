# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import hashlib
import itertools
import shutil
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from lerobot.datasets import video_timeline
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


def encode_clip(
    path: Path,
    count: int = 31,
    gop: int = 30,
    bframes: int = 2,
    timescale: int = 15360,
    seed: int = 0,
    codec: str = "h264",
    fps: Fraction = Fraction(30),
    width: int = 96,
) -> None:
    options = {"g": str(gop), "bf": str(bframes), "crf": "18"}
    if codec == "h264":
        options["sc_threshold"] = "0"
    elif codec == "hevc":
        options["x265-params"] = f"pools=1:frame-threads=1:log-level=error:open-gop=0:bframes={bframes}"
    elif codec == "libsvtav1":
        options = {"g": str(gop), "crf": "30", "preset": "12", "svtav1-params": "lp=1"}
    with av.open(str(path), "w", options={"video_track_timescale": str(timescale)}) as container:
        stream = container.add_stream(codec, fps, options=options)
        stream.width, stream.height, stream.pix_fmt = width, 64, "yuv420p"
        stream.codec_context.thread_count = 1
        y, x = np.indices((64, width))
        for index in range(count):
            rgb = np.stack(
                ((x * 3 + index * 7 + seed) % 256, (y * 4 + index * 3) % 256, (x + y + seed * 2) % 256),
                axis=2,
            ).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, 1 / fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decoded_frames(path: Path) -> list[tuple[Fraction, bytes]]:
    with av.open(str(path)) as container:
        return [
            (frame.pts * frame.time_base, hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).digest())
            for frame in container.decode(video=0)
        ]


def packet_timestamps(path: Path) -> list[tuple[Fraction, Fraction, Fraction]]:
    with av.open(str(path)) as container:
        return [
            (packet.pts * packet.time_base, packet.dts * packet.time_base, packet.duration * packet.time_base)
            for packet in container.demux(video=0)
            if packet.size
        ]


def h264_image_payloads(path: Path) -> list[bytes]:
    """Compare encoded image NALs independently of container/parameter-set changes."""
    result = []
    with av.open(str(path)) as container:
        extra = container.streams.video[0].codec_context.extradata
        assert extra[0] == 1
        length_bytes = (extra[4] & 3) + 1
        for packet in container.demux(video=0):
            data, cursor = bytes(packet), 0
            while cursor < len(data):
                size = int.from_bytes(data[cursor : cursor + length_bytes], "big")
                cursor += length_bytes
                nal = data[cursor : cursor + size]
                assert len(nal) == size and size > 0
                if nal[0] & 31 in (1, 5):
                    result.append(hashlib.sha256(nal).digest())
                cursor += size
            assert cursor == len(data)
    return result


def assert_concatenated(paths: list[Path], output: Path, step: Fraction = Fraction(1, 30)) -> None:
    expected: list[tuple[Fraction, bytes]] = []
    boundaries = {0}
    for path in paths:
        source = decoded_frames(path)
        offset = len(expected) * step
        expected.extend((pts + offset, pixels) for pts, pixels in source)
        boundaries.update((len(expected) - 1, len(expected), len(expected) + 1))
    assert decoded_frames(output) == expected
    stamps = packet_timestamps(output)
    delay = max(-packet_timestamps(path)[0][1] for path in paths)
    assert len(stamps) == len(expected)
    assert [dts for _, dts, _ in stamps] == [i * step - delay for i in range(len(expected))]
    assert all(dts <= pts and duration == step for pts, dts, duration in stamps)
    assert get_video_duration_in_s(output) == float(len(expected) * step)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.start_time == 0
        assert stream.duration * stream.time_base == len(expected) * step
    for index in sorted(i for i in boundaries if i < len(expected)):
        timestamp, pixels = expected[index]
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
            found = False
            for frame in container.decode(video=0):
                actual = frame.pts * frame.time_base
                if actual < timestamp:
                    continue
                assert actual == timestamp
                assert hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).digest() == pixels
                found = True
                break
            assert found, f"Could not seek to frame {index}"


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("concat-clips")
    settings = {"no_b": (30, 0, 15360), "g2": (2, 3, 15360), "b2": (30, 2, 15360), "tb90": (2, 3, 90000)}
    result = {}
    for index, (name, (gop, bframes, timescale)) in enumerate(settings.items()):
        path = root / f"{name}.mp4"
        encode_clip(path, count=31 + index, gop=gop, bframes=bframes, timescale=timescale, seed=index * 21)
        result[name] = path
    assert [video_timeline.scan_video_timeline(result[name]).decode_delay for name in settings] == [
        Fraction(0),
        Fraction(1, 30),
        Fraction(2, 30),
        Fraction(1, 30),
    ]
    return result


@pytest.mark.parametrize("names", list(itertools.permutations(("no_b", "g2", "b2", "tb90"), 2)))
def test_mixed_decode_delays_and_time_bases(tmp_path: Path, clips: dict[str, Path], names: tuple[str, str]):
    paths = [clips[name] for name in names]
    output = tmp_path / "merged.mp4"
    concatenate_video_files(paths, output)
    assert_concatenated(paths, output)
    assert h264_image_payloads(output) == [payload for path in paths for payload in h264_image_payloads(path)]
    with av.open(str(output)) as container:
        assert container.streams.video[0].codec_context.codec_tag == "avc3"


def test_incremental_in_place_equals_one_shot(tmp_path: Path, clips: dict[str, Path]):
    paths = [clips[name] for name in ("no_b", "g2", "b2", "tb90", "no_b")]
    direct, incremental = tmp_path / "direct.mp4", tmp_path / "incremental.mp4"
    concatenate_video_files(paths, direct)
    shutil.copyfile(paths[0], incremental)
    for path in paths[1:]:
        concatenate_video_files([incremental, path], incremental)
    assert_concatenated(paths, incremental)
    assert packet_timestamps(direct) == packet_timestamps(incremental)
    assert decoded_frames(direct) == decoded_frames(incremental)
    assert h264_image_payloads(direct) == h264_image_payloads(incremental)


def test_short_clips(tmp_path: Path):
    paths = [tmp_path / f"short-{count}.mp4" for count in (1, 2, 3)]
    for count, path in enumerate(paths, start=1):
        encode_clip(path, count=count, seed=count)
    output = tmp_path / "merged.mp4"
    concatenate_video_files(paths, output)
    assert_concatenated(paths, output)


@pytest.mark.parametrize("codec", ["hevc", "libsvtav1"])
def test_other_dataset_codecs(tmp_path: Path, codec: str):
    paths = [tmp_path / f"input-{i}.mp4" for i in range(2)]
    for index, path in enumerate(paths):
        encode_clip(path, count=9 + index, gop=2, bframes=index * 2, seed=index, codec=codec)
    output = tmp_path / "merged.mp4"
    concatenate_video_files(paths, output)
    assert_concatenated(paths, output)


def test_fractional_fps(tmp_path: Path):
    paths = [tmp_path / f"input-{i}.mp4" for i in range(2)]
    for index, path in enumerate(paths):
        encode_clip(
            path, count=9, fps=Fraction(30000, 1001), timescale=30000 * (index + 1), bframes=index * 2
        )
    output = tmp_path / "merged.mp4"
    concatenate_video_files(paths, output)
    assert_concatenated(paths, output, step=Fraction(1001, 30000))


@pytest.mark.parametrize(
    "defect, message",
    [
        ("pts", "requires PTS, DTS"),
        ("dts", "requires PTS, DTS"),
        ("duration", "positive duration"),
        ("dts_step", "nonuniform DTS"),
        ("duration_step", "nonuniform DTS or duration"),
        ("off_grid", "off-grid PTS"),
        ("duplicate", "duplicate PTS"),
        ("origin", "starting at zero"),
        ("keyframe", "start with a keyframe"),
        ("empty", "no video packets"),
    ],
)
def test_invalid_packets_fail_before_overwrite(
    tmp_path: Path, clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch, defect: str, message: str
):
    original_packets = video_timeline._video_packets

    def corrupt_packets(container: av.container.InputContainer) -> Iterator[av.Packet]:
        for index, packet in enumerate(original_packets(container)):
            if defect == "empty":
                continue
            if defect == "origin":
                packet.pts += packet.duration
            if index == 0:
                if defect in {"pts", "dts"}:
                    setattr(packet, defect, None)
                elif defect == "duration":
                    packet.duration = 0
                elif defect == "keyframe":
                    packet.is_keyframe = False
            if index == 1:
                if defect == "dts_step":
                    packet.dts += 1
                elif defect == "duration_step":
                    packet.duration += 1
                elif defect == "off_grid":
                    packet.pts += 1
                elif defect == "duplicate":
                    packet.pts = 0
            yield packet

    monkeypatch.setattr(video_timeline, "_video_packets", corrupt_packets)
    output = tmp_path / "existing.mp4"
    shutil.copyfile(clips["b2"], output)
    before = output.read_bytes()
    with pytest.raises(ValueError, match=message):
        concatenate_video_files([output, clips["g2"]], output)
    with pytest.raises(ValueError, match=message):
        get_video_duration_in_s(output)
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


def test_mux_failure_preserves_original_and_cleans_temporary_files(
    tmp_path: Path, clips: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    original_packets = video_timeline._video_packets
    calls = 0

    def fail_during_remux(container: av.container.InputContainer) -> Iterator[av.Packet]:
        nonlocal calls
        calls += 1
        for index, packet in enumerate(original_packets(container)):
            if calls == 3 and index == 2:
                raise RuntimeError("injected remux failure")
            yield packet

    output = tmp_path / "existing.mp4"
    shutil.copyfile(clips["no_b"], output)
    before = output.read_bytes()
    monkeypatch.setattr(video_timeline, "_video_packets", fail_during_remux)
    with pytest.raises(RuntimeError, match="injected remux failure"):
        concatenate_video_files([output, clips["b2"]], output)
    assert calls == 3
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("width,fps", [(128, 30), (96, 25)])
def test_incompatible_inputs(tmp_path: Path, clips: dict[str, Path], width: int, fps: int):
    other, output = tmp_path / "other.mp4", tmp_path / "output.mp4"
    encode_clip(other, count=10, width=width, fps=Fraction(fps), timescale=90000)
    with pytest.raises(ValueError, match="must match"):
        concatenate_video_files([clips["no_b"], other], output)
    assert not output.exists()


def test_empty_inputs_and_overwrite_flag(tmp_path: Path, clips: dict[str, Path]):
    output = tmp_path / "output.mp4"
    with pytest.raises(FileNotFoundError, match="No input video paths"):
        concatenate_video_files([], output)
    shutil.copyfile(clips["b2"], output)
    before = output.read_bytes()
    concatenate_video_files([tmp_path / "missing.mp4"], output, overwrite=False)
    assert output.read_bytes() == before
    concatenate_video_files([str(clips["b2"])], output)
    assert_concatenated([clips["b2"]], output)


def test_common_timescale_overflow_is_rejected(tmp_path: Path):
    paths = [tmp_path / f"clock-{timescale}.mp4" for timescale in (1000003, 1000033)]
    for path, timescale in zip(paths, (1000003, 1000033), strict=True):
        encode_clip(path, count=3, fps=Fraction(1), timescale=timescale)
    output = tmp_path / "output.mp4"
    with pytest.raises(ValueError, match="32-bit limit"):
        concatenate_video_files(paths, output)
    assert not output.exists()


def test_additional_streams_are_not_silently_discarded(tmp_path: Path, clips: dict[str, Path]):
    multiple, output_path = tmp_path / "two-streams.mp4", tmp_path / "output.mp4"
    with av.open(str(clips["no_b"])) as source, av.open(str(multiple), "w") as output:
        streams = [output.add_stream_from_template(source.streams.video[0], opaque=True) for _ in range(2)]
        for packet in source.demux(video=0):
            if not packet.size:
                continue
            for stream in streams:
                copied = av.Packet(bytes(packet))
                copied.pts, copied.dts, copied.duration = packet.pts, packet.dts, packet.duration
                copied.time_base, copied.stream = packet.time_base, stream
                output.mux(copied)
    with pytest.raises(ValueError, match="exactly one video stream"):
        concatenate_video_files([multiple], output_path)
    assert not output_path.exists()


def test_external_ffmpeg_decode(tmp_path: Path, clips: dict[str, Path]):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return pytest.skip("System ffmpeg is needed for an independent decoder check")
    output = tmp_path / "merged.mp4"
    concatenate_video_files(list(clips.values()), output)
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("video_backend", ["pyav", None], ids=["pyav", "default-decoder"])
def test_record_and_aggregate_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, video_backend: str | None
):
    """Exercise real add_frame/save_episode and aggregate paths, including dataset reads."""
    key = "observation.images.camera"
    lengths = ((7, 11), (13, 17))
    sources = []
    monkeypatch.setenv("LEROBOT_VIDEO_VCODEC", "h264")
    for source_index, episode_lengths in enumerate(lengths):
        monkeypatch.setenv("LEROBOT_VIDEO_GOP", "1" if source_index == 0 else "30")
        dataset = LeRobotDataset.create(
            repo_id=f"test/concat-{source_index}",
            root=tmp_path / f"source-{source_index}",
            fps=30,
            features={
                key: {"dtype": "video", "shape": (64, 96, 3), "names": ["height", "width", "channels"]}
            },
            video_backend=video_backend,
        )
        for episode_index, length in enumerate(episode_lengths):
            for index in range(length):
                pixels: NDArray[np.uint8] = np.full(
                    (64, 96, 3), source_index * 80 + episode_index * 20 + index, dtype=np.uint8
                )
                dataset.add_frame({key: pixels, "task": "concatenate videos"})
            dataset.save_episode()
        dataset.finalize()
        sources.append(dataset)

    source_paths = [source.root / source.meta.get_video_file_path(0, key) for source in sources]
    # Change only the second file's MP4 clock, as with independently generated datasets.
    rescaled = tmp_path / "rescaled.mp4"
    with (
        av.open(str(source_paths[1])) as source,
        av.open(
            str(rescaled), "w", options={"video_track_timescale": "90000", "avoid_negative_ts": "disabled"}
        ) as output,
    ):
        stream = output.add_stream_from_template(source.streams.video[0], opaque=True)
        for packet in source.demux(video=0):
            if packet.size:
                packet.stream = stream
                output.mux(packet)
    rescaled.replace(source_paths[1])
    timelines = [video_timeline.scan_video_timeline(path) for path in source_paths]
    assert timelines[0].decode_delay == 0 < timelines[1].decode_delay
    assert [timeline.time_base for timeline in timelines] == [Fraction(1, 15360), Fraction(1, 90000)]

    aggregate_datasets(
        repo_ids=[source.repo_id for source in sources],
        roots=[source.root for source in sources],
        aggr_repo_id="test/concat-merged",
        aggr_root=tmp_path / "merged",
    )
    merged = LeRobotDataset("test/concat-merged", root=tmp_path / "merged", video_backend=video_backend)
    output_path = merged.root / merged.meta.get_video_file_path(0, key)
    assert_concatenated(source_paths, output_path)
    offset = 0
    for episode, length in zip(merged.meta.episodes, itertools.chain.from_iterable(lengths), strict=True):
        assert episode[f"videos/{key}/from_timestamp"] == pytest.approx(offset / 30)
        assert episode[f"videos/{key}/to_timestamp"] == pytest.approx((offset + length) / 30)
        offset += length
    offset = 0
    for source in sources:
        loaded = LeRobotDataset(source.repo_id, root=source.root, video_backend=video_backend)
        for index in range(len(loaded)):
            assert torch.equal(merged[offset + index][key], loaded[index][key])
        offset += len(loaded)
    assert len(merged) == offset
