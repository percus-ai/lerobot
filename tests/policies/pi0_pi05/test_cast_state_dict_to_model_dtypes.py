"""cast_state_dict_to_model_dtypes: 重みをモデルへ写す前に CPU 上でモデル側の dtype に揃える。"""

import torch
from torch import nn

from lerobot.policies.utils import cast_state_dict_to_model_dtypes, load_safetensors_casted_to_model


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bf = nn.Linear(4, 3)          # 推論で bfloat16 になる層の代わり
        self.fp = nn.Linear(3, 2)          # float32 のまま残る射影層の代わり
        self.register_buffer("steps", torch.zeros(1, dtype=torch.int64))
        self.bf.to(torch.bfloat16)


def _state_dict_fp32(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: (v.float() if v.is_floating_point() else v.clone()) for k, v in model.state_dict().items()}


def test_casts_only_mismatched_floats_and_matches_load_state_dict_rounding():
    torch.manual_seed(0)
    model = _Tiny()
    sd = _state_dict_fp32(model)
    sd["bf.weight"] = torch.randn(3, 4) * 0.37          # 丸めが起きる値
    sd["fp.weight"] = torch.randn(2, 3) * 0.37
    sd["extra.unknown"] = torch.randn(2)                  # モデルに無いキーは触らない
    model.load_state_dict({k: v for k, v in sd.items() if k != "extra.unknown"})  # 従来経路の丸め
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    casted = cast_state_dict_to_model_dtypes(sd, model)

    assert casted == 2                                     # bf.weight と bf.bias(float32 → bfloat16)だけ
    assert sd["bf.weight"].dtype == torch.bfloat16
    assert sd["bf.bias"].dtype == torch.bfloat16
    assert sd["fp.weight"].dtype == torch.float32          # float32 の層はそのまま
    assert sd["steps"].dtype == torch.int64                # 整数バッファはそのまま
    assert sd["extra.unknown"].dtype == torch.float32      # 未知のキーはそのまま
    # 先に CPU で丸めた値は、load_state_dict が写すときに丸めた値とビット単位で同じ
    assert torch.equal(sd["bf.weight"].view(torch.int16), reference["bf.weight"].view(torch.int16))
    assert torch.equal(sd["fp.weight"], reference["fp.weight"])


def test_noop_when_dtypes_already_match():
    model = _Tiny().float()                                # 学習時(全部 float32)の形
    sd = _state_dict_fp32(model)
    assert cast_state_dict_to_model_dtypes(sd, model) == 0
    assert all(v.dtype == torch.float32 for k, v in sd.items() if v.is_floating_point())


def test_streaming_loader_casts_per_tensor_and_keeps_file_keys(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(1)
    model = _Tiny()
    saved = {
        "bf.weight": torch.randn(3, 4) * 0.37,     # float32 → bfloat16 に変換される
        "bf.bias": torch.randn(3) * 0.37,
        "fp.weight": torch.randn(2, 3) * 0.37,     # float32 のまま
        "fp.bias": torch.randn(2) * 0.37,
        "steps": torch.tensor([7]),                # 整数はそのまま
        "unused.weight": torch.randn(2),           # モデルに無いキー
    }
    path = tmp_path / "model.safetensors"
    save_file(saved, str(path))
    model.load_state_dict({k: v for k, v in saved.items() if k != "unused.weight"})  # 従来経路の丸め
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    loaded = load_safetensors_casted_to_model(path, model, map_key=lambda name: name)

    assert set(loaded) == set(saved)                       # ファイル内のキーのまま返す
    assert loaded["bf.weight"].dtype == torch.bfloat16
    assert loaded["fp.weight"].dtype == torch.float32
    assert loaded["steps"].dtype == torch.int64 and int(loaded["steps"]) == 7
    assert loaded["unused.weight"].dtype == torch.float32
    assert torch.equal(loaded["bf.weight"].view(torch.int16), reference["bf.weight"].view(torch.int16))
    assert torch.equal(loaded["fp.weight"], reference["fp.weight"])
