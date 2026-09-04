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

"""Exact, zero-origin CFR timelines for lossless dataset video concatenation."""

import math
import tempfile
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path

import av
from pydantic import BaseModel, ConfigDict


class VideoTimeline(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    time_base: Fraction
    frame_ticks: int
    first_dts: int
    frame_count: int
    codec: str
    width: int
    height: int
    pixel_format: str
    extradata: bytes

    @property
    def frame_duration(self) -> Fraction:
        return self.frame_ticks * self.time_base

    @property
    def duration(self) -> Fraction:
        return self.frame_count * self.frame_duration

    @property
    def decode_delay(self) -> Fraction:
        return -self.first_dts * self.time_base


def _video_packets(container: av.container.InputContainer) -> Iterator[av.Packet]:
    for packet in container.demux(video=0):
        # PyAV emits an empty, untimestamped flush sentinel at EOF, not a video frame.
        if packet.size == 0 and packet.pts is None and packet.dts is None:
            continue
        yield packet


def scan_video_timeline(path: Path | str) -> VideoTimeline:
    """Validate the actual PTS/DTS lattice, without trusting declared duration/FPS."""
    path = Path(path)
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream and no other streams")
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None or time_base <= 0:
            raise ValueError(f"{path}: missing or invalid video time base")
        frame_ticks = first_dts = None
        count = next_pts_index = 0
        pending_pts: set[int] = set()
        for packet in _video_packets(container):
            pts, dts, duration = packet.pts, packet.dts, packet.duration
            if pts is None or dts is None or duration is None or duration <= 0:
                raise ValueError(f"{path}: packet {count} requires PTS, DTS and a positive duration")
            if packet.time_base != time_base:
                raise ValueError(f"{path}: packet {count} has an inconsistent time base")
            if first_dts is None or frame_ticks is None:
                if not packet.is_keyframe or dts > 0:
                    raise ValueError(f"{path}: must start with a keyframe and nonpositive DTS")
                first_dts, frame_ticks = dts, duration
            if duration != frame_ticks or dts != first_dts + count * frame_ticks:
                raise ValueError(f"{path}: packet {count} has nonuniform DTS or duration; expected CFR")
            if dts > pts or pts < 0 or pts % frame_ticks:
                raise ValueError(f"{path}: packet {count} has invalid PTS/DTS or an off-grid PTS")
            pts_index = pts // frame_ticks
            if pts_index < next_pts_index or pts_index in pending_pts:
                raise ValueError(f"{path}: packet {count} has a duplicate PTS")
            pending_pts.add(pts_index)
            # Retain only the reorder window, not a timestamp list for the entire video.
            while next_pts_index in pending_pts:
                pending_pts.remove(next_pts_index)
                next_pts_index += 1
            count += 1
        if first_dts is None or frame_ticks is None:
            raise ValueError(f"{path}: no video packets")
        if pending_pts:
            raise ValueError(f"{path}: PTS must cover consecutive frames starting at zero")
        if stream.codec_context.format is None or not stream.codec_context.extradata:
            raise ValueError(f"{path}: missing video format or codec initialization data")
        return VideoTimeline(
            path=path,
            time_base=time_base,
            frame_ticks=frame_ticks,
            first_dts=first_dts,
            frame_count=count,
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            pixel_format=stream.codec_context.format.name,
            extradata=stream.codec_context.extradata,
        )


def remux_video_files(input_paths: Sequence[Path | str], output_path: Path) -> None:
    """Remux prevalidated CFR clips, preserving PTS and sharing one decode delay."""
    clips = [scan_video_timeline(path) for path in input_paths]
    first = clips[0]
    for clip in clips[1:]:
        if (clip.codec, clip.width, clip.height, clip.pixel_format, clip.frame_duration) != (
            first.codec,
            first.width,
            first.height,
            first.pixel_format,
            first.frame_duration,
        ):
            raise ValueError(f"{clip.path}: codec, dimensions, pixel format and frame duration must match")
        if first.codec not in {"h264", "hevc"} and clip.extradata != first.extradata:
            raise ValueError(
                f"{clip.path}: differing codec initialization data is unsupported for {clip.codec}"
            )

    timescale = math.lcm(*(clip.time_base.denominator for clip in clips))
    if timescale > 2**31 - 1:
        raise ValueError("Input time bases have no exact common MP4 timescale within the signed 32-bit limit")
    time_base = Fraction(1, timescale)
    delay = max(clip.decode_delay for clip in clips)
    frame_ticks = int(first.frame_duration / time_base)
    delay_ticks = int(delay / time_base)
    bitstream_filter = {"h264": "h264_mp4toannexb", "hevc": "hevc_mp4toannexb"}.get(first.codec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original intact on failure, including when it is also an input.
    # A sibling temporary file allows an atomic replacement on the same filesystem.
    with tempfile.TemporaryDirectory(prefix="lerobot-concat-", dir=output_path.parent) as tmp_dir:
        tmp_path = Path(tmp_dir) / "video.mp4"
        with av.open(
            str(tmp_path),
            mode="w",
            options={
                "movflags": "faststart",
                "video_track_timescale": str(timescale),
                "avoid_negative_ts": "disabled",
            },
        ) as output:
            with av.open(str(first.path)) as template:
                stream = output.add_stream_from_template(template.streams.video[0], opaque=True)
                if bitstream_filter is not None:
                    # Initialize output extradata too. Each clip's parameter sets must
                    # travel with its packets when encoder settings differ (e.g. B-frames).
                    av.bitstream.BitStreamFilterContext(bitstream_filter, template.streams.video[0], stream)
                    stream.codec_context.codec_tag = "avc3" if first.codec == "h264" else "hev1"
            stream.time_base = time_base
            output.start_encoding()
            if stream.time_base != time_base:
                raise ValueError("MP4 muxer changed the requested exact video time base")
            offset = 0
            for clip in clips:
                scale = int(clip.time_base / time_base)
                clip_delay = int(clip.decode_delay / time_base)
                with av.open(str(clip.path)) as source:
                    packet_filter = (
                        av.bitstream.BitStreamFilterContext(bitstream_filter, source.streams.video[0])
                        if bitstream_filter is not None
                        else None
                    )
                    count = 0
                    for packet in _video_packets(source):
                        pts, dts = packet.pts, packet.dts
                        if (
                            pts is None
                            or dts is None
                            or packet.duration != clip.frame_ticks
                            or packet.time_base != clip.time_base
                            or dts != clip.first_dts + count * clip.frame_ticks
                        ):
                            raise ValueError(f"{clip.path}: packet timeline changed after validation")
                        # R = max(input decode delays), C = preceding frame duration.
                        # PTS' = PTS + C; DTS' = DTS + C - (R - input delay).
                        # Thus DTS' = C - R + j*T. Boundaries are exactly T apart,
                        # and DTS only moves earlier, preserving DTS <= PTS.
                        pts = pts * scale + offset
                        dts = dts * scale + offset - (delay_ticks - clip_delay)
                        if packet_filter is not None:
                            filtered = packet_filter.filter(packet)
                            if len(filtered) != 1:
                                raise ValueError(f"{clip.path}: bitstream filter must preserve packet count")
                            packet = filtered[0]
                        packet.pts, packet.dts, packet.duration = pts, dts, frame_ticks
                        packet.time_base = time_base
                        packet.stream = stream
                        output.mux(packet)
                        count += 1
                    if count != clip.frame_count:
                        raise ValueError(f"{clip.path}: packet count changed after validation")
                    if packet_filter is not None and packet_filter.filter(None):
                        raise ValueError(f"{clip.path}: unexpected buffered packets in bitstream filter")
                offset += clip.frame_count * frame_ticks
        tmp_path.replace(output_path)
