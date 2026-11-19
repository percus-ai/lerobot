"""GR00T N1.7 の実装が使う共通部品のうち、この LeRobot (0.4.3 系) に無いものを補う。

本家 LeRobot (0.7 系) の GR00T 実装は、新しい共通部品 (lerobot.configs の再輸出、
lerobot.lerobot_types、require_package、チャンク推論の宣言など) を前提にしている。
本家ごと上げると π0 / π0.5 側の修正に影響するので、GR00T のフォルダの中だけで補う。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, TypedDict

from torch import Tensor

from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, PolicyAction, TransitionKey
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.import_utils import is_package_available
from lerobot.utils.utils import get_safe_torch_device

__all__ = [
    "EnvTransition",
    "RTCActionSelectKwargs",
    "TransitionKey",
    "_datasets_available",
    "_diffusers_available",
    "_transformers_available",
    "get_safe_torch_device",
    "load_pretrained_policy_processors",
    "policy_action_to_transition",
    "require_package",
    "require_qwen3_vl",
    "transition_to_policy_action",
]

_transformers_available = is_package_available("transformers")
_diffusers_available = is_package_available("diffusers")
_datasets_available = is_package_available("datasets")


def require_package(pkg_name: str, extra: str, import_name: str | None = None) -> None:
    """必要なパッケージが無ければ、入れ方を添えて ImportError にする。"""
    if importlib.util.find_spec(import_name or pkg_name.replace("-", "_")) is None:
        raise ImportError(
            f"'{pkg_name}' is required but not installed. Install it with: "
            f"pip install 'lerobot[{extra}]' (or uv pip install 'lerobot[{extra}]')"
        )


def require_qwen3_vl(*classes: Any) -> None:
    """GR00T N1.7 の土台 (Qwen3-VL 系) のクラスを読めなかった環境なら、必要な版を添えてエラーにする。

    classes には、呼び出し側が `from transformers import ...` で読もうとしたクラスを渡す
    (読めなかった物は None になっている)。
    """
    require_package("transformers", extra="groot")
    if any(cls is None for cls in classes):
        import transformers

        raise ImportError(
            f"GR00T N1.7 requires transformers>=4.57 (found {transformers.__version__}): "
            "its Cosmos-Reason2 / Qwen3-VL backbone classes are missing. Use the GR00T environment."
        )


class RTCActionSelectKwargs(TypedDict, total=False):
    """Real-Time Chunking が `predict_action_chunk` に渡す引数。"""

    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def load_pretrained_policy_processors(
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """保存済みの前処理・後処理の組を読み込み、保存では残らない相互の参照を結び直す。"""
    from .relative_actions import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        revision=revision,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        revision=revision,
    )
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                step.relative_step = relative_step
    return preprocessor, postprocessor
