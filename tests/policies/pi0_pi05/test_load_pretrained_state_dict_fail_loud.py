"""load_pretrained_state_dict: 重みが読めなければ例外、結合重みの片方欠けだけ許し、他の不足・過剰は例外 (2026-11-05)。"""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.pretrained import load_pretrained_state_dict, tied_missing_keys


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 3)
        self.head = nn.Linear(3, 4, bias=False)
        self.head.weight = self.embed.weight  # 結合重み (lm_head と embed_tokens と同じ形)
        self.proj = nn.Linear(3, 3)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


def _save(dirpath: Path, state: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(state, str(dirpath / "model.safetensors"))


def _full_state(policy: nn.Module) -> dict:
    # safetensors の保存と同じく共有テンソルは片方だけ (head.weight を落とす)
    sd = {k: v.clone() for k, v in policy.state_dict().items() if k != "model.head.weight"}
    return sd


def test_tied_missing_keys_detects_shared_tensor() -> None:
    policy = _Policy()
    assert tied_missing_keys(policy, ["model.head.weight"], ["model.embed.weight"]) == {"model.head.weight"}
    assert tied_missing_keys(policy, ["model.proj.weight"], ["model.embed.weight"]) == set()


def test_missing_weights_file_raises(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not load weights .*Refusing to return an untrained model"):
        load_pretrained_state_dict(_Policy(), tmp_path)


def test_tied_key_missing_in_file_is_accepted_and_values_load(tmp_path: Path) -> None:
    src = _Policy()
    with torch.no_grad():
        src.model.embed.weight.fill_(0.5)
        src.model.proj.weight.fill_(-1.0)
    _save(tmp_path, _full_state(src))
    dst = _Policy()
    load_pretrained_state_dict(dst, tmp_path)
    assert torch.equal(dst.model.embed.weight, src.model.embed.weight)
    assert torch.equal(dst.model.head.weight, src.model.embed.weight)  # 結合先にも入っている
    assert torch.equal(dst.model.proj.weight, src.model.proj.weight)


def test_real_missing_key_raises(tmp_path: Path) -> None:
    sd = _full_state(_Policy())
    sd.pop("model.proj.weight")
    _save(tmp_path, sd)
    with pytest.raises(RuntimeError, match="does not match .*missing=1 \\['model.proj.weight'\\]"):
        load_pretrained_state_dict(_Policy(), tmp_path)


def test_unexpected_key_raises_but_strict_false_only_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sd = _full_state(_Policy())
    sd["model.extra.weight"] = torch.zeros(2)
    _save(tmp_path, sd)
    with pytest.raises(RuntimeError, match="unexpected=1"):
        load_pretrained_state_dict(_Policy(), tmp_path)
    load_pretrained_state_dict(_Policy(), tmp_path, strict=False)
    assert "Warning: state dict mismatch (strict=False)" in capsys.readouterr().out


def test_keys_without_model_prefix_are_remapped(tmp_path: Path) -> None:
    sd = {k[len("model."):]: v for k, v in _full_state(_Policy()).items()}
    _save(tmp_path, sd)
    load_pretrained_state_dict(_Policy(), tmp_path, fix_keys=lambda d: d)
