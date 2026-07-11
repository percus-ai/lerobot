import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Event, Thread

import av
import numpy as np
import pytest
import torch

import lerobot.datasets.video_utils as video_utils_module
from lerobot.datasets.video_utils import (
    VIDEO_QUERY_TIMESTAMP_SOURCE_FRAME_INDEX,
    FrameTimestampError,
    VideoDecoderCache,
    decode_video_frames_torchcodec,
    decode_video_frames_torchvision,
    resolve_video_query_timestamp_source,
)


@dataclass
class _FrameBatch:
    data: torch.Tensor
    pts_seconds: torch.Tensor
    duration_seconds: torch.Tensor


@dataclass
class _Metadata:
    begin_stream_seconds: float
    end_stream_seconds: float


class _Decoder:
    def __init__(
        self,
        pts_seconds: list[float],
        last_duration_s: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.pts_seconds = torch.tensor(pts_seconds, dtype=dtype)
        self.duration_seconds = torch.tensor(
            [
                *(pts_seconds[index + 1] - pts_seconds[index] for index in range(len(pts_seconds) - 1)),
                last_duration_s,
            ],
            dtype=dtype,
        )
        self.data = (
            torch.tensor(pts_seconds, dtype=dtype).mul(100).to(torch.uint8)[:, None, None, None]
        )
        self.metadata = _Metadata(pts_seconds[0], pts_seconds[-1] + last_duration_s)
        self.timestamp_queries: list[list[float]] = []

    def get_frames_played_at(self, seconds: list[float]) -> _FrameBatch:
        if any(timestamp >= self.metadata.end_stream_seconds for timestamp in seconds):
            raise AssertionError("end_stream_seconds is not queryable")
        self.timestamp_queries.append(seconds)
        indices = torch.searchsorted(
            self.pts_seconds,
            torch.tensor(seconds, dtype=self.pts_seconds.dtype),
            right=True,
        ).sub(1)
        indices.clamp_(0, len(self.pts_seconds) - 1)
        return _FrameBatch(
            data=self.data[indices],
            pts_seconds=self.pts_seconds[indices],
            duration_seconds=self.duration_seconds[indices],
        )

    def get_frames_at(self, indices: list[int]) -> _FrameBatch:
        raise AssertionError(f"frame-index decoding must not be used: {indices}")


class _DecoderCache:
    def __init__(self, decoder: _Decoder) -> None:
        self.decoder = decoder

    @contextmanager
    def lease(self, video_path: str):
        yield self.decoder, self.decoder.pts_seconds[-1].item()


def _decode(decoder: _Decoder, timestamps: list[float], tolerance_s: float) -> torch.Tensor:
    return decode_video_frames_torchcodec(
        "video.mp4",
        timestamps,
        tolerance_s,
        decoder_cache=_DecoderCache(decoder),
    )


def _decoded_values(frames: torch.Tensor) -> list[int]:
    return frames[:, 0, 0, 0].mul(255).round().to(torch.int).tolist()


def test_video_query_timestamp_source_defaults_to_persisted_timestamp() -> None:
    assert (
        resolve_video_query_timestamp_source(
            metadata_value=None,
            metadata_value_present=False,
        )
        is False
    )


def test_video_query_timestamp_source_accepts_exact_frame_index_contract() -> None:
    assert (
        resolve_video_query_timestamp_source(
            metadata_value=VIDEO_QUERY_TIMESTAMP_SOURCE_FRAME_INDEX,
            metadata_value_present=True,
        )
        is True
    )


def test_video_query_timestamp_source_rejects_unknown_contract() -> None:
    with pytest.raises(ValueError, match="video_query_timestamp_source"):
        resolve_video_query_timestamp_source(
            metadata_value="unknown",
            metadata_value_present=True,
        )


def test_torchcodec_uses_pts_nearest_for_cfr_video() -> None:
    decoder = _Decoder([0.0, 0.1, 0.2])

    frames = _decode(decoder, [0.04, 0.06, 0.14, 0.16], tolerance_s=0.05)

    assert _decoded_values(frames) == [0, 10, 10, 20]
    assert len(decoder.timestamp_queries) == 2


def test_torchcodec_uses_pts_nearest_for_vfr_video() -> None:
    decoder = _Decoder([0.0, 0.02, 0.2, 0.21])

    frames = _decode(decoder, [0.1, 0.16, 0.205], tolerance_s=0.11)

    assert _decoded_values(frames) == [2, 20, 20]


def test_torchcodec_preserves_duplicate_queries() -> None:
    decoder = _Decoder([0.0, 0.02, 0.2])

    frames = _decode(decoder, [0.16, 0.16], tolerance_s=0.05)

    assert _decoded_values(frames) == [20, 20]


def test_torchcodec_does_not_decode_successors_for_exact_pts() -> None:
    decoder = _Decoder([0.0, 0.1, 0.2])

    frames = _decode(decoder, [0.0, 0.1, 0.2], tolerance_s=1e-4)

    assert _decoded_values(frames) == [0, 10, 20]
    assert decoder.timestamp_queries == [[0.0, 0.1, 0.2]]


def test_torchcodec_clamps_decode_queries_at_video_boundaries() -> None:
    decoder = _Decoder([0.1, 0.2])

    frames = _decode(decoder, [0.08, 0.31], tolerance_s=0.12)

    assert _decoded_values(frames) == [10, 20]
    assert decoder.timestamp_queries[0][0] == pytest.approx(0.1)
    assert decoder.timestamp_queries[0][1] == pytest.approx(0.2)
    assert all(
        timestamp < decoder.metadata.end_stream_seconds
        for query in decoder.timestamp_queries
        for timestamp in query
    )


def test_torchcodec_does_not_query_a_successor_after_the_last_frame() -> None:
    decoder = _Decoder([0.0, 0.2], last_duration_s=1e-7)

    frames = _decode(decoder, [0.2, 0.21], tolerance_s=0.02)

    assert _decoded_values(frames) == [20, 20]
    assert decoder.timestamp_queries == [[0.2, 0.2]]


def test_torchcodec_accepts_nearest_pts_at_inclusive_tolerance_boundary() -> None:
    decoder = _Decoder([0.0, 0.2])

    frames = _decode(decoder, [0.1], tolerance_s=0.1)

    assert _decoded_values(frames) == [0]


def test_torchcodec_rejects_nearest_pts_outside_tolerance() -> None:
    decoder = _Decoder([0.0, 0.3])

    with pytest.raises(FrameTimestampError, match="violate the tolerance"):
        _decode(decoder, [0.15], tolerance_s=0.1)


def test_torchcodec_rejects_float32_pts_for_long_vfr_timeline() -> None:
    decoder = _Decoder([3600.03328, 3600.03338], dtype=torch.float32)

    with pytest.raises(FrameTimestampError, match="float64 PTS"):
        _decode(decoder, [108_001 / 30], tolerance_s=0.001)


class _FakeFileHandle:
    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeOpenFile:
    def __init__(self, file_handle: _FakeFileHandle) -> None:
        self.file_handle = file_handle

    def __enter__(self) -> _FakeFileHandle:
        return self.file_handle


class _FakeVideoDecoder:
    def __init__(self, file_handle: _FakeFileHandle, seek_mode: str) -> None:
        assert seek_mode == "exact"
        self.file_handle = file_handle

    def __len__(self) -> int:
        return 1

    def get_frames_at(self, indices: list[int]) -> _FrameBatch:
        assert indices == [0]
        return _FrameBatch(
            data=torch.zeros((1, 3, 1, 1), dtype=torch.uint8),
            pts_seconds=torch.tensor([0.0], dtype=torch.float64),
            duration_seconds=torch.tensor([0.1], dtype=torch.float64),
        )


def _install_fake_torchcodec(monkeypatch) -> dict[str, list[_FakeFileHandle]]:
    handles: dict[str, list[_FakeFileHandle]] = {}

    def fake_open(path: str) -> _FakeOpenFile:
        handle = _FakeFileHandle(path)
        handles.setdefault(path, []).append(handle)
        return _FakeOpenFile(handle)

    fake_torchcodec = types.ModuleType("torchcodec")
    fake_torchcodec.__path__ = []
    fake_decoders = types.ModuleType("torchcodec.decoders")
    fake_decoders.VideoDecoder = _FakeVideoDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", fake_torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_decoders)
    monkeypatch.setattr(video_utils_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(video_utils_module.fsspec, "open", fake_open)
    return handles


def test_video_decoder_cache_is_bounded_lru_and_closes_evicted_handles(monkeypatch) -> None:
    handles = _install_fake_torchcodec(monkeypatch)

    cache = VideoDecoderCache(capacity=2)
    with cache.lease("first.mp4") as (first_decoder, _):
        pass
    with cache.lease("second.mp4") as (second_decoder, _):
        pass
    with cache.lease("first.mp4") as (cached_first_decoder, _):
        assert cached_first_decoder is first_decoder

    with cache.lease("third.mp4"):
        pass

    assert not handles["first.mp4"][0].closed
    assert handles["second.mp4"][0].closed
    assert not handles["third.mp4"][0].closed
    assert cache.size() == 2

    with cache.lease("second.mp4") as (recreated_second_decoder, _):
        assert recreated_second_decoder is not second_decoder
    assert len(handles["second.mp4"]) == 2
    assert handles["first.mp4"][0].closed

    cache.clear()
    assert handles["second.mp4"][1].closed
    assert handles["third.mp4"][0].closed
    assert cache.size() == 0


def test_video_decoder_cache_does_not_evict_a_leased_decoder(monkeypatch) -> None:
    handles = _install_fake_torchcodec(monkeypatch)

    cache = VideoDecoderCache(capacity=1)
    attempted = Event()
    finished = Event()

    def add_second_decoder() -> None:
        attempted.set()
        with cache.lease("second.mp4"):
            pass
        finished.set()

    with cache.lease("first.mp4"):
        thread = Thread(target=add_second_decoder)
        thread.start()
        assert attempted.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        assert not handles["first.mp4"][0].closed

    thread.join(timeout=1)
    assert not thread.is_alive()
    assert finished.is_set()
    assert handles["first.mp4"][0].closed


def test_video_decoder_cache_decodes_different_entries_in_parallel(monkeypatch) -> None:
    _install_fake_torchcodec(monkeypatch)
    cache = VideoDecoderCache(capacity=2)
    with cache.lease("first.mp4"):
        pass
    with cache.lease("second.mp4"):
        pass

    first_entered = Event()
    second_entered = Event()
    release = Event()

    def hold_lease(path: str, entered: Event) -> None:
        with cache.lease(path):
            entered.set()
            assert release.wait(timeout=1)

    first_thread = Thread(target=hold_lease, args=("first.mp4", first_entered))
    second_thread = Thread(target=hold_lease, args=("second.mp4", second_entered))
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=1)
    assert second_entered.wait(timeout=1)
    release.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    cache.clear()


def test_video_decoder_cache_serializes_the_same_entry(monkeypatch) -> None:
    _install_fake_torchcodec(monkeypatch)
    cache = VideoDecoderCache(capacity=1)
    first_entered = Event()
    second_attempted = Event()
    second_entered = Event()
    release_first = Event()

    def hold_first() -> None:
        with cache.lease("video.mp4"):
            first_entered.set()
            assert release_first.wait(timeout=1)

    def hold_second() -> None:
        second_attempted.set()
        with cache.lease("video.mp4"):
            second_entered.set()

    first_thread = Thread(target=hold_first)
    second_thread = Thread(target=hold_second)
    first_thread.start()
    assert first_entered.wait(timeout=1)
    second_thread.start()
    assert second_attempted.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    assert second_entered.wait(timeout=1)
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    cache.clear()


def test_video_decoder_cache_rejects_clear_while_leased(monkeypatch) -> None:
    handles = _install_fake_torchcodec(monkeypatch)
    cache = VideoDecoderCache(capacity=1)

    with cache.lease("video.mp4"):
        with pytest.raises(RuntimeError, match="while decoders are leased"):
            cache.clear()
        assert not handles["video.mp4"][0].closed

    cache.clear()
    assert handles["video.mp4"][0].closed


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.5])
def test_video_decoder_cache_rejects_invalid_capacity(capacity) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        VideoDecoderCache(capacity=capacity)


class _Reader:
    def __init__(self) -> None:
        self.container = self

    def seek(self, timestamp: float, keyframes_only: bool) -> None:
        pass

    def __iter__(self):
        yield {"pts": 0.0, "data": torch.zeros((3, 2, 2), dtype=torch.uint8)}
        yield {"pts": 0.2, "data": torch.full((3, 2, 2), 200, dtype=torch.uint8)}

    def close(self) -> None:
        pass


class _LongTimelineReader(_Reader):
    def __iter__(self):
        center = 108_001 / 30
        yield {
            "pts": center - 0.00005,
            "data": torch.zeros((3, 2, 2), dtype=torch.uint8),
        }
        yield {
            "pts": center + 0.00005,
            "data": torch.full((3, 2, 2), 200, dtype=torch.uint8),
        }


def test_pyav_uses_the_same_inclusive_tolerance_contract(monkeypatch) -> None:
    monkeypatch.setattr(video_utils_module.torchvision, "set_video_backend", lambda _backend: None)
    monkeypatch.setattr(video_utils_module.torchvision.io, "VideoReader", lambda *_args: _Reader())

    frame = decode_video_frames_torchvision("video.mp4", [0.1], tolerance_s=0.1)

    assert frame.shape == (1, 3, 2, 2)
    with pytest.raises(FrameTimestampError, match="violate the tolerance"):
        decode_video_frames_torchvision("video.mp4", [0.1], tolerance_s=0.09)


def test_pyav_preserves_float64_query_precision_on_long_vfr_timeline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(video_utils_module.torchvision, "set_video_backend", lambda _backend: None)
    monkeypatch.setattr(
        video_utils_module.torchvision.io,
        "VideoReader",
        lambda *_args: _LongTimelineReader(),
    )

    frame = decode_video_frames_torchvision(
        "video.mp4",
        [108_001 / 30],
        tolerance_s=0.001,
    )

    assert torch.count_nonzero(frame) == 0


def _write_cfr_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 30)
    stream.codec_context.time_base = Fraction(1, 30)
    try:
        for index in range(4):
            image = np.full((24, 32, 3), index * 50, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _remux_with_pts(source: Path, destination: Path, pts_seconds: list[float]) -> None:
    time_base = Fraction(1, 90_000)
    input_container = av.open(str(source), mode="r")
    output_container = av.open(str(destination), mode="w")
    try:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream_from_template(input_stream)
        output_stream.time_base = time_base
        packet_index = 0
        for packet in input_container.demux(input_stream):
            if packet.dts is None:
                continue
            pts = round(pts_seconds[packet_index] * time_base.denominator)
            packet.pts = pts
            packet.dts = pts
            packet.time_base = time_base
            packet.stream = output_stream
            if packet_index + 1 < len(pts_seconds):
                next_pts = round(pts_seconds[packet_index + 1] * time_base.denominator)
                packet.duration = next_pts - pts
            output_container.mux(packet)
            packet_index += 1
        assert packet_index == len(pts_seconds)
    finally:
        input_container.close()
        output_container.close()


def _assert_real_torchcodec_frames(
    video_path: Path,
    *,
    timestamps: list[float],
    expected_indices: list[int],
    tolerance_s: float,
) -> None:
    pytest.importorskip("torchcodec")
    from torchcodec.decoders import VideoDecoder

    expected_decoder = VideoDecoder(str(video_path), seek_mode="exact")
    expected = expected_decoder.get_frames_at(indices=expected_indices).data.float() / 255
    cache = VideoDecoderCache()
    try:
        actual = decode_video_frames_torchcodec(
            video_path,
            timestamps,
            tolerance_s,
            decoder_cache=cache,
        )
    finally:
        cache.clear()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(importlib.util.find_spec("torchcodec") is None, reason="torchcodec is unavailable")
def test_real_torchcodec_reads_existing_cfr_timeline_by_nearest_pts(tmp_path: Path) -> None:
    video_path = tmp_path / "cfr.mp4"
    _write_cfr_video(video_path)

    _assert_real_torchcodec_frames(
        video_path,
        timestamps=[0.0, 0.04, 0.06, 0.1],
        expected_indices=[0, 1, 2, 3],
        tolerance_s=0.02,
    )


@pytest.mark.skipif(importlib.util.find_spec("torchcodec") is None, reason="torchcodec is unavailable")
def test_real_torchcodec_reads_vfr_timeline_by_nearest_pts(tmp_path: Path) -> None:
    cfr_path = tmp_path / "source.mp4"
    vfr_path = tmp_path / "vfr.mp4"
    _write_cfr_video(cfr_path)
    _remux_with_pts(cfr_path, vfr_path, [0.0, 0.02, 0.2, 0.21])

    _assert_real_torchcodec_frames(
        vfr_path,
        timestamps=[0.0, 0.1, 0.16, 0.205, 0.214],
        expected_indices=[0, 1, 2, 2, 3],
        tolerance_s=0.1,
    )
