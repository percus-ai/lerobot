#!/usr/bin/env python
"""GR00T N1.7 の学習済みチェックポイントの持ち方 (この LeRobot で足した部分)。

- チェックポイントに N1.7 本体の構成を同梱し、読み込みでベースモデルを取りに行かない
- 推論でメモリを半分にするために、重みを bfloat16 で持てる
- Real-Time Chunking の設定を config に持てる
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.groot import modeling_groot  # noqa: E402
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.groot_n1_7 import GR00TN17  # noqa: E402
from lerobot.policies.groot.modeling_groot import GROOT_MODEL_CONFIG_FILENAME, GrootPolicy  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

NATIVE_CONFIG = {"model_type": "Gr00tN1d7", "action_horizon": 40, "max_action_dim": 132}


class _DummyModel(nn.Module):
    """N1.7 本体の代わり。重み 1 個と、保存できる構成だけを持つ。"""

    def __init__(self, native_config: dict | None = None):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        native = dict(native_config or NATIVE_CONFIG)
        self.config = SimpleNamespace(to_dict=lambda: dict(native), **native)


def _config(**overrides) -> GrootConfig:
    return GrootConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        use_bf16=False,
        **overrides,
    )


def _policy_from_base(monkeypatch, **overrides) -> GrootPolicy:
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyModel()))
    return GrootPolicy(_config(**overrides))


def test_save_writes_the_native_model_config_next_to_the_weights(tmp_path, monkeypatch):
    policy = _policy_from_base(monkeypatch)

    policy.save_pretrained(tmp_path)

    assert (tmp_path / "model.safetensors").is_file()
    assert json.loads((tmp_path / GROOT_MODEL_CONFIG_FILENAME).read_text()) == NATIVE_CONFIG


def test_trained_checkpoint_loads_without_touching_the_base_model(tmp_path, monkeypatch):
    policy = _policy_from_base(monkeypatch)
    with torch.no_grad():
        policy._groot_model.linear.weight.fill_(0.25)
    policy.save_pretrained(tmp_path)

    built_from: list[dict] = []

    def build(self, native_config, model_kwargs):
        built_from.append(dict(native_config))
        return _DummyModel(native_config)

    def fail(cls, **kwargs):
        raise AssertionError("the base model must not be loaded for a self-contained checkpoint")

    monkeypatch.setattr(GrootPolicy, "_build_model_from_native_config", build)
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(fail))

    loaded = GrootPolicy.from_pretrained(tmp_path)

    assert built_from == [NATIVE_CONFIG]
    torch.testing.assert_close(loaded._groot_model.linear.weight, torch.full((4, 4), 0.25))


def test_checkpoint_without_the_native_config_loads_the_base_model_first(tmp_path, monkeypatch):
    # 本家 LeRobot で学習したチェックポイントは構成を同梱していない。本家と同じ経路で読む。
    policy = _policy_from_base(monkeypatch)
    policy.save_pretrained(tmp_path)
    (tmp_path / GROOT_MODEL_CONFIG_FILENAME).unlink()

    base_loads: list[str] = []

    def from_base(cls, **kwargs):
        base_loads.append(kwargs["pretrained_model_name_or_path"])
        return _DummyModel()

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(from_base))

    GrootPolicy.from_pretrained(tmp_path)

    assert base_loads == [policy.config.base_model_path]


def test_native_config_builds_the_model_without_backbone_weights(monkeypatch):
    built: dict = {}

    class _FakeN17(_DummyModel):
        def __init__(self, config, load_backbone_weights=True):
            super().__init__()
            built["config"] = config
            built["load_backbone_weights"] = load_backbone_weights
            self.config = config
            self.backbone = SimpleNamespace(set_trainable_parameters=lambda **kw: built.update(backbone=kw))
            self.action_head = SimpleNamespace(set_trainable_parameters=lambda **kw: built.update(head=kw))

    monkeypatch.setattr(modeling_groot, "GR00TN17", _FakeN17)

    GrootPolicy(
        _config(tune_visual=True, tune_diffusion_model=False, num_inference_timesteps=8),
        groot_model_config={**NATIVE_CONFIG, "num_inference_timesteps": 4, "tune_top_llm_layers": 2},
    )

    assert built["load_backbone_weights"] is False
    assert built["config"].action_horizon == 40
    # config で指定した値はチェックポイントの値より優先する (ベースモデルから読む経路と同じ)。
    assert built["config"].num_inference_timesteps == 8
    assert built["config"].tune_visual is True
    assert built["backbone"] == {"tune_visual": True, "tune_llm": False, "tune_top_llm_layers": 0}
    assert built["head"] == {"tune_projector": True, "tune_diffusion_model": False, "tune_vlln": True}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parameters_are_held_in_the_configured_dtype(monkeypatch, dtype):
    policy = _policy_from_base(monkeypatch, dtype=dtype)

    assert {p.dtype for p in policy.parameters()} == {dtype}


def test_bfloat16_checkpoint_loads_weights_saved_in_float32(tmp_path, monkeypatch):
    policy = _policy_from_base(monkeypatch)
    with torch.no_grad():
        policy._groot_model.linear.weight.fill_(0.5)
    policy.save_pretrained(tmp_path)
    monkeypatch.setattr(
        GrootPolicy, "_build_model_from_native_config", lambda self, native, kwargs: _DummyModel(native)
    )

    config = _config(dtype=torch.bfloat16)
    loaded = GrootPolicy.from_pretrained(tmp_path, config=config)

    weight = loaded._groot_model.linear.weight
    assert weight.dtype == torch.bfloat16
    torch.testing.assert_close(weight.float(), torch.full((4, 4), 0.5))
    assert loaded._groot_model.scale.dtype == torch.bfloat16


def test_unsupported_parameter_dtype_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Expected torch.float32 or torch.bfloat16"):
        _policy_from_base(monkeypatch, dtype=torch.float16)


def test_config_roundtrips_dtype_and_rtc_settings(tmp_path):
    config = _config(dtype=torch.bfloat16, rtc_config=RTCConfig(enabled=True, execution_horizon=13))

    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == "bfloat16"
    assert isinstance(loaded, GrootConfig)
    assert loaded.dtype == torch.bfloat16
    assert loaded.rtc_config.enabled is True
    assert loaded.rtc_config.execution_horizon == 13
