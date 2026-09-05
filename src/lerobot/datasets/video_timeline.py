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

"""Frame-order timelines for lossless CFR dataset video concatenation."""

import logging
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
    frame_count: int
    presentation_indices: tuple[int, ...]
    source_pts: tuple[int, ...]
    source_dts: tuple[int, ...]
    codec_tag: str
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
        return max(0, max(i - rank for i, rank in enumerate(self.presentation_indices))) * self.frame_duration


def _video_packets(container: av.container.InputContainer) -> Iterator[av.Packet]:
    for packet in container.demux(video=0):
        # PyAV emits an empty, untimestamped flush sentinel at EOF, not a video frame.
        if packet.size == 0 and packet.pts is None and packet.dts is None:
            continue
        yield packet


def scan_video_timeline(path: Path | str) -> VideoTimeline:
    """Validate complete CFR segments and map packets to decoded presentation order.

    Old joins may have nonuniform DTS or a PTS offset at a new keyframe. Neither
    changes frame identity. COPY_OPAQUE proves the packet/frame correspondence;
    packet order is never confused with presentation order. Within each segment
    displayed frames must remain exactly one frame duration apart.
    """
    path = Path(path)
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream and no other streams")
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None or time_base <= 0:
            raise ValueError(f"{path}: missing or invalid video time base")
        frame_ticks = first_dts = previous_dts = previous_pts = None
        source_pts: list[int] = []
        source_dts: list[int] = []
        ranks: dict[int, int] = {}
        decoder = stream.codec_context
        decoder.flags |= av.codec.context.Flags.copy_opaque

        def accept_frames(frames: list[av.VideoFrame]) -> None:
            nonlocal previous_pts
            for frame in frames:
                index = frame.opaque
                if not isinstance(index, int) or index in ranks or not 0 <= index < len(source_pts):
                    raise ValueError(f"{path}: requires exactly one decoded frame per packet")
                pts = frame.pts
                if pts is None or pts != source_pts[index]:
                    raise ValueError(f"{path}: decoded frame has missing or inconsistent PTS")
                if previous_pts is None:
                    if pts != 0:
                        raise ValueError(f"{path}: PTS must start at zero")
                elif pts <= previous_pts:
                    raise ValueError(f"{path}: duplicate or reversed presentation timestamps")
                elif pts - previous_pts != frame_ticks and not frame.key_frame:
                    raise ValueError(f"{path}: nonuniform PTS within a segment; expected CFR")
                ranks[index] = len(ranks)
                previous_pts = pts

        for packet in _video_packets(container):
            count = len(source_pts)
            pts, dts, duration = packet.pts, packet.dts, packet.duration
            if pts is None or dts is None or duration is None or duration <= 0:
                raise ValueError(f"{path}: packet {count} requires PTS, DTS and a positive duration")
            if packet.time_base != time_base:
                raise ValueError(f"{path}: packet {count} has an inconsistent time base")
            if first_dts is None or frame_ticks is None:
                if not packet.is_keyframe or dts > 0:
                    raise ValueError(f"{path}: must start with a keyframe and nonpositive DTS")
                first_dts, frame_ticks = dts, duration
            if duration != frame_ticks:
                raise ValueError(f"{path}: packet {count} has nonuniform duration; expected CFR")
            if dts > pts or pts < 0 or (previous_dts is not None and dts <= previous_dts):
                raise ValueError(f"{path}: packet {count} has invalid PTS/DTS")
            source_pts.append(pts)
            source_dts.append(dts)
            packet.opaque = count
            previous_dts = dts
            accept_frames(decoder.decode(packet))
        if first_dts is None or frame_ticks is None:
            raise ValueError(f"{path}: no video packets")
        accept_frames(decoder.decode(None))
        count = len(source_pts)
        if len(ranks) != count:
            raise ValueError(f"{path}: decoded frame count does not match packet count")
        if stream.codec_context.format is None or not stream.codec_context.extradata:
            raise ValueError(f"{path}: missing video format or codec initialization data")
        return VideoTimeline(
            path=path,
            time_base=time_base,
            frame_ticks=frame_ticks,
            frame_count=count,
            presentation_indices=tuple(ranks[i] for i in range(count)),
            source_pts=tuple(source_pts),
            source_dts=tuple(source_dts),
            codec_tag=stream.codec_context.codec_tag,
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            pixel_format=stream.codec_context.format.name,
            extradata=stream.codec_context.extradata,
        )


def remux_video_files(clips: Sequence[VideoTimeline], output_path: Path) -> None:
    """Generate one CFR clock from verified frame order, without recompression."""
    if not clips:
        raise ValueError("At least one validated video timeline is required")
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
    inline_headers = bitstream_filter is not None and (
        any(clip.extradata != first.extradata for clip in clips)
        or any(clip.codec_tag in {"avc3", "hev1"} for clip in clips)
    )

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
                if inline_headers:
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
                if any(
                    pts != rank * clip.frame_ticks
                    for pts, rank in zip(clip.source_pts, clip.presentation_indices, strict=True)
                ):
                    logging.info(
                        "Normalizing presentation timestamps from verified frame order: %s", clip.path
                    )
                with av.open(str(clip.path)) as source:
                    packet_filter = (
                        av.bitstream.BitStreamFilterContext(bitstream_filter, source.streams.video[0])
                        if inline_headers
                        else None
                    )
                    count = 0
                    for packet in _video_packets(source):
                        pts, dts = packet.pts, packet.dts
                        if (
                            pts is None
                            or dts is None
                            or count >= clip.frame_count
                            or packet.duration != clip.frame_ticks
                            or packet.time_base != clip.time_base
                            or pts != clip.source_pts[count]
                            or dts != clip.source_dts[count]
                        ):
                            raise ValueError(f"{clip.path}: packet timeline changed after validation")
                        # r = decoded presentation rank; j = unchanged packet order.
                        # R = max((j-r)*T) over all inputs. Thus DTS is strictly
                        # increasing and DTS <= PTS, including old/new boundaries.
                        pts = clip.presentation_indices[count] * frame_ticks + offset
                        dts = count * frame_ticks + offset - delay_ticks
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
