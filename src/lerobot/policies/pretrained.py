# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import builtins
import logging
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, TypeVar

import packaging
import safetensors
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # Create base kwargs
        kwargs = {"strict": strict}

        # Add device parameter for newer versions that support it
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # Load the model with appropriate kwargs
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # For older versions, manually move to device if needed
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

        Child classes using action chunking should use this method within `select_action` to form the action chunk
        cached for selection.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
    ):
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        # Push the files to the repo in a single commit
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id

            self.save_pretrained(saved_path)  # Calls _save_pretrained and stores model tensors

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            cfg.save_pretrained(saved_path)  # Calls _save_pretrained and stores train config

            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload policy weights, train config and readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"Model pushed to {commit_info.repo_url.url}")

    def generate_model_card(
        self, dataset_repo_id: str, model_type: str, license: str | None, tags: list[str] | None
    ) -> ModelCard:
        base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None  # Set a base model

        card_data = ModelCardData(
            license=license or "apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
            model_name=model_type,
            datasets=dataset_repo_id,
            base_model=base_model,
        )

        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
        )
        card = ModelCard.from_template(card_data, template_str=template_card)
        card.validate()
        return card


def tied_missing_keys(model: nn.Module, missing_keys, loaded_keys) -> set[str]:
    """load_state_dict(strict=False) が「無い」と言ったキーのうち、結合重み (同じテンソルを複数の名前で
    持つ。PaliGemma の lm_head と embed_tokens 等) で、読み込んだ側の名前と同じテンソルを指すものを返す。
    safetensors は共有テンソルを 1 つしか保存しないので、片方のキーは必ず「無い」扱いになるが、
    実体は読み込まれている。"""
    named = dict(model.named_parameters(remove_duplicate=False))
    named.update(dict(model.named_buffers(remove_duplicate=False)))
    loaded_ids = {id(named[k]) for k in loaded_keys if k in named}
    return {k for k in missing_keys if k in named and id(named[k]) in loaded_ids}


def load_pretrained_state_dict(
    model: nn.Module,
    pretrained_name_or_path: str | Path,
    *,
    strict: bool = True,
    fix_keys=None,
    cache_dir=None,
    force_download: bool = False,
    resume_download=None,
    proxies=None,
    token=None,
    revision=None,
    local_files_only: bool = False,
) -> nn.Module:
    """model.safetensors を読んで model に載せる。読めなければ例外 (未学習の重みのまま返さない)。
    fix_keys(state_dict) でキーを直した後、"model." 接頭辞を補い、strict=False で載せてから
    足りない / 余るキーを自分で検査する: 結合重みで片方だけ保存されているキーは許し、それ以外の
    不足・過剰は strict=True なら RuntimeError (strict=False なら警告だけ)。

    2026-11-05: アップロード失敗で重みの無いモデルディレクトリが登録され、旧実装 (読めなければ
    警告だけ出して未学習のモデルを返す / 読み込み中の例外を握りつぶす) がそれを「読み込めた」
    ことにして推論に使われ、指令が発散した。"""
    from safetensors.torch import load_file

    print(f"Loading model from: {pretrained_name_or_path}")
    try:
        local_dir = Path(pretrained_name_or_path)
        if local_dir.is_dir():
            # ローカルディレクトリは直接見る (Hub の解決は要らない)
            resolved_file = local_dir / SAFETENSORS_SINGLE_FILE
            if not resolved_file.is_file() or resolved_file.stat().st_size == 0:
                raise FileNotFoundError(f"{resolved_file} is missing or empty")
            resolved_file = str(resolved_file)
        else:
            from transformers.utils import cached_file

            resolved_file = cached_file(
                pretrained_name_or_path,
                SAFETENSORS_SINGLE_FILE,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            if not resolved_file:
                raise FileNotFoundError(f"{SAFETENSORS_SINGLE_FILE} not found in {pretrained_name_or_path}")
        original_state_dict = load_file(resolved_file)
    except Exception as e:
        raise RuntimeError(
            f"Could not load weights ({SAFETENSORS_SINGLE_FILE}) from {pretrained_name_or_path}: {e}. "
            "Refusing to return an untrained model."
        ) from e
    print(f"✓ Loaded state dict from {SAFETENSORS_SINGLE_FILE} ({len(original_state_dict)} tensors)")

    fixed_state_dict = fix_keys(original_state_dict) if fix_keys is not None else original_state_dict
    remapped_state_dict = {}
    remap_count = 0
    for key, value in fixed_state_dict.items():
        if not key.startswith("model."):
            remapped_state_dict[f"model.{key}"] = value
            remap_count += 1
        else:
            remapped_state_dict[key] = value
    if remap_count > 0:
        print(f"Remapped {remap_count} state dict keys")

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    loaded_keys = [k for k in remapped_state_dict if k not in unexpected_keys]
    tied = tied_missing_keys(model, missing_keys, loaded_keys)
    real_missing = [k for k in missing_keys if k not in tied]
    if tied:
        print(f"Tied keys not stored separately (ok): {sorted(tied)}")
    if real_missing or unexpected_keys:
        detail = (
            f"missing={len(real_missing)} {real_missing[:5]}{' ...' if len(real_missing) > 5 else ''} "
            f"unexpected={len(unexpected_keys)} {list(unexpected_keys)[:5]}{' ...' if len(unexpected_keys) > 5 else ''}"
        )
        if strict:
            raise RuntimeError(
                f"state dict of {pretrained_name_or_path} does not match {type(model).__name__}: {detail}"
            )
        print(f"Warning: state dict mismatch (strict=False): {detail}")
    else:
        print("All keys loaded successfully!")
    return model
