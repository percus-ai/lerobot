"""Regression tests for PI05 layer checkpointing, using small real transformers."""

import copy
import unittest
from collections import Counter
from unittest.mock import patch

import torch
from torch import Tensor, nn
from transformers import GemmaConfig as HFGemmaConfig, PaliGemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from lerobot.policies.pi05 import modeling_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import GemmaConfig, PaliGemmaWithExpertModel, PI05Pytorch


def small_gemma_config(variant: str) -> GemmaConfig:
    if variant not in ("gemma_2b", "gemma_300m"):
        raise ValueError(variant)
    return GemmaConfig(width=64, depth=3, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=8)


class SmallJointModel(PaliGemmaWithExpertModel):
    def __init__(
        self,
        vlm_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        use_adarms: list[bool],
        precision: str,
    ) -> None:
        nn.Module.__init__(self)
        text_config = HFGemmaConfig(
            vocab_size=32,
            hidden_size=vlm_config.width,
            intermediate_size=vlm_config.mlp_dim,
            num_hidden_layers=vlm_config.depth,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=8,
            hidden_activation="gelu_pytorch_tanh",
            use_adarms=use_adarms[0],
        )
        config = PaliGemmaConfig(
            text_config=text_config.to_dict(),
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 8,
                "patch_size": 4,
            },
            projection_dim=64,
            hidden_size=64,
            vocab_size=32,
            image_token_index=31,
        )
        self.paligemma = PaliGemmaForConditionalGeneration(config)
        self.gemma_expert = GemmaForCausalLM(
            HFGemmaConfig(
                vocab_size=32,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_hidden_layers=action_expert_config.depth,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=8,
                hidden_activation="gelu_pytorch_tanh",
                use_adarms=use_adarms[1],
                adarms_cond_dim=action_expert_config.width,
            )
        )
        self.gemma_expert.model.embed_tokens = None
        if precision != "float32":
            raise ValueError("This CPU regression test requires float32")
        self.float()


class TestPI05Checkpointing(unittest.TestCase):
    def test_checkpointing_preserves_gradients_without_replaying_layers_twice(self) -> None:
        torch.manual_seed(19)
        config = PI05Config(
            device="cpu",
            dtype="float32",
            max_action_dim=8,
            chunk_size=4,
            n_action_steps=4,
            tokenizer_max_length=6,
            image_resolution=(8, 8),
        )
        with (
            patch.object(modeling_pi05, "get_gemma_config", small_gemma_config),
            patch.object(modeling_pi05, "PaliGemmaWithExpertModel", SmallJointModel),
        ):
            reference = PI05Pytorch(config).train()
        candidate = copy.deepcopy(reference)
        candidate.gradient_checkpointing_enable()
        images = [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]
        image_masks = [torch.tensor([True, True]), torch.tensor([False, True])]
        tokens = torch.randint(0, 31, (2, 6))
        masks = torch.tensor([[True, True, False, False, False, False], [True] * 6])
        actions = torch.randn(2, 4, 8)
        noise = torch.randn_like(actions)
        time = torch.tensor([0.2, 0.8])
        expected = reference(images, image_masks, tokens, masks, actions, noise=noise, time=time)
        expected.mean().backward()

        # Count real layer evaluations, including backward recomputation. This
        # catches the performance regression even if the loss still matches.
        with patch.object(
            modeling_pi05,
            "compute_layer_complete",
            wraps=modeling_pi05.compute_layer_complete,
        ) as layers:
            actual = candidate(images, image_masks, tokens, masks, actions, noise=noise, time=time)
            actual.mean().backward()
        counts = Counter(call.args[0] for call in layers.call_args_list)
        self.assertEqual(counts, {0: 2, 1: 2, 2: 2})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (name, parameter), (expected_name, original) in zip(
            candidate.named_parameters(),
            reference.named_parameters(),
            strict=True,
        ):
            self.assertEqual(name, expected_name)
            self.assertEqual(parameter.requires_grad, original.requires_grad)
            if original.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0, msg=name)

        # The same configured model must use the ordinary path in evaluation.
        candidate.eval()
        with (
            torch.no_grad(),
            patch.object(
                modeling_pi05, "compute_layer_complete", wraps=modeling_pi05.compute_layer_complete
            ) as layers,
        ):
            evaluation: Tensor = candidate(
                images, image_masks, tokens, masks, actions, noise=noise, time=time
            )
        self.assertEqual(Counter(call.args[0] for call in layers.call_args_list), {0: 1, 1: 1, 2: 1})
        torch.testing.assert_close(evaluation, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
