"""GrootConfig.dtype (torch.dtype) を config.json とコマンドラインで読み書きできるようにする。

本家 LeRobot は lerobot.configs.parser でこの変換を登録している。この LeRobot には無いので、
GR00T の設定を読み込むときに登録する (ほかのポリシーの dtype は文字列なので影響しない)。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import draccus
import torch


@draccus.decode.register(torch.dtype)
def _decode_dtype(value: Any, path: Sequence[str] = ()) -> torch.dtype:
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype {value!r}: expected a torch dtype name.")
    return dtype


@draccus.encode.register(torch.dtype)
def _encode_dtype(value: torch.dtype, declared_type: type | None = None) -> str:
    return str(value).removeprefix("torch.")
