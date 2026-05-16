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
        from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep

        step = GrootEagleEncodeStep()
        try:
            step.proc
        except ImportError as exc:
            assert "GROOT Eagle processor requires the transformers dependency" in str(exc)
        else:
            raise AssertionError("GROOT Eagle processor did not report the missing transformers dependency")
        """
    )
