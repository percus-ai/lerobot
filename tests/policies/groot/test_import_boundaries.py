#!/usr/bin/env python

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

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_python_without_transformers(script: str) -> None:
    project_root = Path(__file__).resolve().parents[3]
    pythonpath_entries = [str(project_root / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    bootstrap = """
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec

        def find_spec_without_transformers(
            name: str,
            package: str | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if name == "transformers" or name.startswith("transformers."):
                return None
            return real_find_spec(name, package)

        class BlockTransformers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "transformers" or fullname.startswith("transformers."):
                    raise ModuleNotFoundError("No module named 'transformers'")
                return None

        sys.modules.pop("transformers", None)
        importlib.util.find_spec = find_spec_without_transformers
        sys.meta_path.insert(0, BlockTransformers())
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap + "\n" + script)],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_act_policy_config_does_not_require_groot_transformers_dependency() -> None:
    _run_python_without_transformers(
        """
        from lerobot.policies.factory import make_policy_config

        config = make_policy_config("act")
        assert config.__class__.__name__ == "ACTConfig"
        """
    )


def test_groot_processor_requires_transformers_only_when_processor_is_built() -> None:
    _run_python_without_transformers(
        """
        from lerobot.policies.factory import make_policy_config
        from lerobot.policies.groot.processor_groot import _build_n1_7_processor

        config = make_policy_config("groot")
        assert config.__class__.__name__ == "GrootConfig"
        try:
            _build_n1_7_processor()
        except ImportError as exc:
            assert "'transformers' is required but not installed" in str(exc)
        else:
            raise AssertionError("GR00T N1.7 processor did not report the missing transformers dependency")
        """
    )


def test_factory_imports_when_transformers_predates_qwen3_vl() -> None:
    """π0 / π0.5 の実行環境 (transformers 4.53 系) には Qwen3-VL のクラスが無い。

    lerobot.policies.factory は GR00T のパッケージを必ず読み込むので、そこで落ちると
    ほかのポリシーが全部使えなくなる。読み込みは通り、GR00T を作るときだけエラーになること。
    """
    project_root = Path(__file__).resolve().parents[3]
    script = """
        import transformers

        # 4.57 より前の transformers を再現する: Qwen3-VL の名前を取り出せなくする。
        blocked = {
            "Qwen3VLConfig",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLProcessor",
            "Qwen3VLVideoProcessor",
        }
        lazy_module = type(transformers)
        real_getattr = lazy_module.__getattr__

        def getattr_without_qwen3_vl(self, name):
            if name in blocked:
                raise AttributeError(name)
            return real_getattr(self, name)

        lazy_module.__getattr__ = getattr_without_qwen3_vl
        for name in blocked:
            transformers.__dict__.pop(name, None)
        assert not hasattr(transformers, "Qwen3VLForConditionalGeneration")

        from lerobot.policies.factory import get_policy_class, make_policy_config

        assert make_policy_config("act").__class__.__name__ == "ACTConfig"
        assert make_policy_config("groot").__class__.__name__ == "GrootConfig"
        assert get_policy_class("groot").__name__ == "GrootPolicy"

        from lerobot.policies.groot.processor_groot import _build_n1_7_processor

        try:
            _build_n1_7_processor()
        except ImportError as exc:
            assert "requires transformers>=4.57" in str(exc), str(exc)
        else:
            raise AssertionError("GR00T N1.7 did not report the outdated transformers")
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
