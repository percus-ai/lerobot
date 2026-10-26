"""load_safetensors_into_model: テンソル単位で読み、CPU でモデル側の dtype に変換してから直接写す。"""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.utils import load_safetensors_into_model


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bf = nn.Linear(4, 3)          # 推論で bfloat16 になる層の代わり
        self.fp = nn.Linear(3, 2)          # float32 のまま残る射影層の代わり
        self.register_buffer("steps", torch.zeros(1, dtype=torch.int64))
        self.bf.to(torch.bfloat16)


def _saved(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "bf.weight": torch.randn(3, 4) * 0.37,     # float32 保存 → bfloat16 に丸めて写す
        "bf.bias": torch.randn(3) * 0.37,
        "fp.weight": torch.randn(2, 3) * 0.37,     # float32 のまま
        "fp.bias": torch.randn(2) * 0.37,
        "steps": torch.tensor([7]),                # 整数はそのまま
    }


def test_matches_load_state_dict_bit_for_bit(tmp_path):
    saved = _saved()
    path = tmp_path / "model.safetensors"
    save_file(saved, str(path))

    reference = _Tiny()
    reference.load_state_dict(saved)               # 従来経路: load_state_dict が写すときに丸める
    model = _Tiny()
    missing, unexpected = load_safetensors_into_model(path, model, map_key=lambda name: name)

    assert missing == [] and unexpected == []
    for key, ref in reference.state_dict().items():
        got = model.state_dict()[key]
        assert got.dtype == ref.dtype, key
        if got.dtype == torch.bfloat16:
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), key
        else:
            assert torch.equal(got, ref), key
    assert model.bf.weight.dtype == torch.bfloat16 and model.fp.weight.dtype == torch.float32


def test_map_key_renames_and_drops_and_reports_missing_unexpected(tmp_path):
    saved = _saved(1)
    saved["old_name.weight"] = saved.pop("fp.weight")   # 名前変換の対象
    saved["dropped.weight"] = torch.randn(2)            # map_key が None を返す = 捨てる
    saved["stray.weight"] = torch.randn(2)              # モデルに無い = unexpected
    del saved["fp.bias"]                                # 無い = missing
    path = tmp_path / "model.safetensors"
    save_file(saved, str(path))

    def map_key(name: str) -> str | None:
        if name == "dropped.weight":
            return None
        return name.replace("old_name.", "fp.")

    model = _Tiny()
    missing, unexpected = load_safetensors_into_model(path, model, map_key=map_key, strict=False)
    assert missing == ["fp.bias"]
    assert unexpected == ["stray.weight"]
    assert torch.equal(model.fp.weight, saved["old_name.weight"])

    with pytest.raises(RuntimeError, match="Missing key"):
        load_safetensors_into_model(path, _Tiny(), map_key=map_key, strict=True)


def test_shape_mismatch_is_an_error(tmp_path):
    saved = _saved(2)
    saved["bf.weight"] = torch.randn(3, 5)
    path = tmp_path / "model.safetensors"
    save_file(saved, str(path))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_safetensors_into_model(path, _Tiny(), map_key=lambda name: name)
